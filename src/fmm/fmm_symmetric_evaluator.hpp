#pragma once

#include <Eigen/Core>
#include <limits>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/fmm/interpolator_configuration.hpp>
#include <polatory/types.hpp>
#include <scalfmm/algorithms/fmm.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/container/particle_container.hpp>
#include <scalfmm/interpolation/interpolation.hpp>
#include <scalfmm/operators/fmm_operators.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/cell.hpp>
#include <scalfmm/tree/for_each.hpp>
#include <scalfmm/tree/group_tree_view.hpp>
#include <scalfmm/tree/leaf_view.hpp>
#include <scalfmm/utils/sort.hpp>
#include <tuple>
#include <unordered_map>

#include "fmm_accuracy_estimator.hpp"
#include "full_direct.hpp"
#include "utility.hpp"

namespace polatory::fmm {

template <class Rbf, class Kernel>
class FmmGenericSymmetricEvaluator<Rbf, Kernel>::Impl {
  static constexpr int kDim{Rbf::kDim};
  using Bbox = geometry::Bbox<kDim>;
  using Points = geometry::Points<kDim>;

  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using Particle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn,
      /* variables */ Index>;

  using Container = scalfmm::container::particle_container<Particle>;

  using NearField = scalfmm::operators::near_field_operator<Kernel>;
  using Interpolator = scalfmm::interpolation::interpolator<double, kDim, Kernel,
                                                            scalfmm::options::modified_uniform_>;
  using FarField = scalfmm::operators::far_field_operator<Interpolator>;
  using FmmOperator = scalfmm::operators::fmm_operators<NearField, FarField>;
  using Position = typename Particle::position_type;
  using Box = scalfmm::component::box<Position>;
  using Cell = scalfmm::component::cell<typename Interpolator::storage_type>;
  using Leaf = scalfmm::component::leaf_view<Particle>;
  using Tree = scalfmm::component::group_tree_view<Cell, Leaf, Box>;

 public:
  Impl(const Rbf& rbf, const Bbox& bbox)
      : rbf_(rbf),
        bbox_(bbox),
        box_(make_box<Rbf, Box>(rbf, bbox)),
        kernel_(rbf),
        near_field_(kernel_) {}

  VecX evaluate() const {
    using namespace scalfmm::algorithms;

    prepare();

    if (tree_height_ > 0) {
      tree_->reset_multipoles();
      tree_->reset_locals();
      tree_->reset_outputs();
      if (!tree_->is_interaction_m2l_lists_built()) {
        scalfmm::list::omp::build_m2l_interaction_list(*tree_, *tree_, 1);
      }
      if (!tree_->is_interaction_p2p_lists_built()) {
        scalfmm::list::omp::build_p2p_interaction_list(*tree_, *tree_, 1, true);
      }
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)]  //
          (*tree_, *fmm_operator_, p2m | m2m | m2l | l2l | l2p | p2p);
    } else {
      particles_.reset_outputs();
      full_direct(particles_, kernel_);
    }

    handle_self_interaction();

    auto result = potentials();

    // Release memory held by the tree.
    tree_.reset(nullptr);

    return result;
  }

  void set_accuracy(double accuracy) {
    accuracy_ = accuracy;

    best_config_.clear();
  }

  void set_points(const Points& points) {
    n_points_ = points.rows();

    particles_.resize(n_points_);

    auto a = rbf_.anisotropy();
    for (Index idx = 0; idx < n_points_; idx++) {
      auto p = particles_.at(idx);
      auto ap = geometry::transform_point<kDim>(a, points.row(idx));
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = ap(i);
      }
      p.variables(idx);
    }

    sorted_level_ = 0;
    tree_.reset(nullptr);
    best_config_.clear();
  }

  void set_weights(const Eigen::Ref<const VecX>& weights) {
    POLATORY_ASSERT(weights.rows() == km * n_points_);

    if (!tree_) {
      for (Index idx = 0; idx < n_points_; idx++) {
        auto p = particles_.at(idx);
        auto orig_idx = std::get<0>(p.variables());
        for (auto i = 0; i < km; i++) {
          p.inputs(i) = weights(km * orig_idx + i);
        }
      }
    } else {
      scalfmm::component::for_each_leaf(std::begin(*tree_), std::end(*tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename Leaf::proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            for (auto i = 0; i < km; i++) {
                                              p.inputs(i) = weights(km * idx + i);
                                            }
                                          }
                                        });
    }

    // NOTE: If weights are changed significantly, the best configuration must be recomputed.
  }

 private:
  InterpolatorConfiguration find_best_configuration(int tree_height) const {
    auto it = best_config_.find(tree_height);
    if (it != best_config_.end()) {
      return it->second;
    }

    auto config = FmmAccuracyEstimator<Rbf, Kernel>::find_best_configuration(
        rbf_, accuracy_, particles_, box_, tree_height);
    return best_config_[tree_height] = config;
  }

  void handle_self_interaction() const {
    if (n_points_ == 0) {
      return;
    }

    scalfmm::container::point<double, kDim> x{};
    auto k = kernel_.evaluate(x, x);

    if (tree_height_ > 0) {
      scalfmm::component::for_each_leaf(std::begin(*tree_), std::end(*tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename Leaf::proxy_type(p_ref);
                                            for (auto i = 0; i < kn; i++) {
                                              for (auto j = 0; j < km; j++) {
                                                p.outputs(i) += p.inputs(j) * k.at(km * i + j);
                                              }
                                            }
                                          }
                                        });
    } else {
      for (Index idx = 0; idx < n_points_; idx++) {
        auto p = particles_.at(idx);
        for (auto i = 0; i < kn; i++) {
          for (auto j = 0; j < km; j++) {
            p.outputs(i) += p.inputs(j) * k.at(km * i + j);
          }
        }
      }
    }
  }

  VecX potentials() const {
    VecX potentials = VecX::Zero(kn * n_points_);

    if (tree_height_ > 0) {
      scalfmm::component::for_each_leaf(std::cbegin(*tree_), std::cend(*tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename Leaf::const_proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            for (auto i = 0; i < kn; i++) {
                                              potentials(kn * idx + i) = p.outputs(i);
                                            }
                                          }
                                        });
    } else {
      for (auto idx = 0; idx < n_points_; idx++) {
        const auto p = particles_.at(idx);
        auto orig_idx = std::get<0>(p.variables());
        for (auto i = 0; i < kn; i++) {
          potentials(kn * orig_idx + i) = p.outputs(i);
        }
      }
    }

    return potentials;
  }

  void prepare() const {
    if (n_points_ < 1024) {
      interpolator_.reset(nullptr);
      far_field_.reset(nullptr);
      fmm_operator_.reset(nullptr);
      tree_.reset(nullptr);
      tree_height_ = 0;
      return;
    }

    auto tree_height = fmm_tree_height<kDim>(n_points_);

    if (sorted_level_ < tree_height - 1) {
      scalfmm::utils::sort_container(box_, tree_height - 1, particles_);
      sorted_level_ = tree_height - 1;
    }

    auto config = find_best_configuration(tree_height);

    if (tree_height_ != tree_height || config_ != config) {
      interpolator_ = std::make_unique<Interpolator>(kernel_, config.order, tree_height,
                                                     box_.width(0), config.d);
      far_field_ = std::make_unique<FarField>(*interpolator_);
      fmm_operator_ = std::make_unique<FmmOperator>(near_field_, *far_field_);
      tree_.reset(nullptr);
      tree_height_ = tree_height;
      config_ = config;
    }

    if (!tree_) {
      tree_ = std::make_unique<Tree>(tree_height, config.order, box_, 10, 10, particles_, true);
    }
  }

  const Rbf& rbf_;
  const Bbox bbox_;
  const Box box_;
  const Kernel kernel_;
  const NearField near_field_;

  double accuracy_{std::numeric_limits<double>::infinity()};
  Index n_points_{};
  mutable Container particles_;
  mutable int sorted_level_{};
  mutable int tree_height_{};
  mutable InterpolatorConfiguration config_{};
  mutable std::unique_ptr<Interpolator> interpolator_;
  mutable std::unique_ptr<FarField> far_field_;
  mutable std::unique_ptr<FmmOperator> fmm_operator_;
  mutable std::unique_ptr<Tree> tree_;
  mutable std::unordered_map<int, InterpolatorConfiguration> best_config_;
};

template <class Rbf, class Kernel>
FmmGenericSymmetricEvaluator<Rbf, Kernel>::FmmGenericSymmetricEvaluator(const Rbf& rbf,
                                                                        const Bbox& bbox)
    : impl_(std::make_unique<Impl>(rbf, bbox)) {}

template <class Rbf, class Kernel>
FmmGenericSymmetricEvaluator<Rbf, Kernel>::~FmmGenericSymmetricEvaluator() = default;

template <class Rbf, class Kernel>
VecX FmmGenericSymmetricEvaluator<Rbf, Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Rbf, class Kernel>
void FmmGenericSymmetricEvaluator<Rbf, Kernel>::set_accuracy(double accuracy) {
  impl_->set_accuracy(accuracy);
}

template <class Rbf, class Kernel>
void FmmGenericSymmetricEvaluator<Rbf, Kernel>::set_points(const Points& points) {
  impl_->set_points(points);
}

template <class Rbf, class Kernel>
void FmmGenericSymmetricEvaluator<Rbf, Kernel>::set_weights(const Eigen::Ref<const VecX>& weights) {
  impl_->set_weights(weights);
}

#define IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF)                 \
  template class FmmGenericSymmetricEvaluator<RBF, Kernel<RBF>>; \
  template class FmmGenericSymmetricEvaluator<RBF, HessianKernel<RBF>>;

#define IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(RBF_NAME) \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<1>);  \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<2>);  \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<3>);

}  // namespace polatory::fmm
