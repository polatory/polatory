#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/inverse_multiquadric.hpp>
#include <polatory/rbf/multiquadric.hpp>
#include <polatory/rbf/polyharmonic_even.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <scalfmm/algorithms/fmm.hpp>
#include <scalfmm/algorithms/full_direct.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/interpolation/interpolation.hpp>
#include <scalfmm/operators/fmm_operators.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/cell.hpp>
#include <scalfmm/tree/for_each.hpp>
#include <scalfmm/tree/group_tree_view.hpp>
#include <scalfmm/tree/leaf_view.hpp>
#include <tuple>
#include <type_traits>
#include <vector>

#include "utility.hpp"

namespace polatory::fmm {

template <class Model, class Kernel>
class fmm_generic_symmetric_evaluator<Model, Kernel>::impl {
  static constexpr int kDim{Model::kDim};
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using Particle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn,
      /* variables */ index_t>;

  using NearField = scalfmm::operators::near_field_operator<Kernel>;
  using Interpolator = scalfmm::interpolation::interpolator<
      double, kDim, Kernel,
      std::conditional_t<std::is_same_v<Kernel, kernel<typename Model::rbf_type>>,
                         scalfmm::options::chebyshev_<>, scalfmm::options::uniform_<>>>;
  using FarField = scalfmm::operators::far_field_operator<Interpolator>;
  using FmmOperator = scalfmm::operators::fmm_operators<NearField, FarField>;
  using Position = typename Particle::position_type;
  using Box = scalfmm::component::box<Position>;
  using Cell = scalfmm::component::cell<typename Interpolator::storage_type>;
  using Leaf = scalfmm::component::leaf_view<Particle>;
  using Tree = scalfmm::component::group_tree_view<Cell, Leaf, Box>;

 public:
  impl(const Model& model, const Bbox& bbox, precision prec)
      : model_(model),
        kernel_(model.rbf()),
        order_(static_cast<int>(prec)),
        box_(make_box<Model, Box>(model, bbox)),
        near_field_(kernel_) {}

  common::valuesd evaluate() const {
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
          (*tree_, *fmm_operator_, p2m);
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)]  //
          (*tree_, *fmm_operator_, m2m);
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)]  //
          (*tree_, *fmm_operator_, m2l | p2p);
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)]  //
          (*tree_, *fmm_operator_, l2l);
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)]  //
          (*tree_, *fmm_operator_, l2p);
    } else {
      for (auto& p : particles_) {
        for (auto i = 0; i < kn; i++) {
          p.outputs(i) = 0.0;
        }
      }
      scalfmm::algorithms::full_direct(particles_, kernel_);
    }

    handle_self_interaction();

    auto result = potentials();

    // Release some memory.
    reset_tree();

    return result;
  }

  void set_points(const Points& points) {
    n_points_ = points.rows();

    particles_.resize(n_points_);

    auto a = model_.rbf().anisotropy();
    for (index_t idx = 0; idx < n_points_; idx++) {
      auto& p = particles_.at(idx);
      auto ap = geometry::transform_point<kDim>(a, points.row(idx));
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = ap(i);
      }
      p.variables(idx);
    }

    tree_.reset(nullptr);
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    POLATORY_ASSERT(weights.rows() == km * n_points_);

    if (!tree_) {
      for (index_t idx = 0; idx < n_points_; idx++) {
        auto& p = particles_.at(idx);
        for (auto i = 0; i < km; i++) {
          p.inputs(i) = weights(km * idx + i);
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
  }

 private:
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
      for (auto& p : particles_) {
        for (auto i = 0; i < kn; i++) {
          for (auto j = 0; j < km; j++) {
            p.outputs(i) += p.inputs(j) * k.at(km * i + j);
          }
        }
      }
    }
  }

  common::valuesd potentials() const {
    common::valuesd potentials = common::valuesd::Zero(kn * n_points_);

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
        const auto& p = particles_.at(idx);
        for (auto i = 0; i < kn; i++) {
          potentials(kn * idx + i) = p.outputs(i);
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
      reset_tree();
      tree_height_ = 0;
      return;
    }

    auto tree_height = fmm_tree_height<kDim>(n_points_);
    if (tree_height_ != tree_height) {
      interpolator_ = std::make_unique<Interpolator>(kernel_, order_, tree_height, box_.width(0));
      far_field_ = std::make_unique<FarField>(*interpolator_);
      fmm_operator_ = std::make_unique<FmmOperator>(near_field_, *far_field_);
      reset_tree();
      tree_height_ = tree_height;
    }

    if (!tree_) {
      tree_ = std::make_unique<Tree>(tree_height, order_, box_, 10, 10, particles_);
      particles_.clear();
      particles_.shrink_to_fit();
    }
  }

  void reset_tree() const {
    if (!tree_) {
      return;
    }

    particles_.resize(n_points_);

    scalfmm::component::for_each_leaf(std::begin(*tree_), std::end(*tree_), [&](const auto& leaf) {
      for (auto p_ref : leaf) {
        auto p = typename Leaf::proxy_type(p_ref);
        auto idx = std::get<0>(p.variables());
        auto& new_p = particles_.at(idx);
        for (auto i = 0; i < kDim; i++) {
          new_p.position(i) = p.position(i);
        }
        for (auto i = 0; i < km; i++) {
          new_p.inputs(i) = p.inputs(i);
        }
        new_p.variables(idx);
      }
    });

    tree_.reset(nullptr);
  }

  const Model& model_;
  const Kernel kernel_;
  const int order_;
  const Box box_;
  const NearField near_field_;

  index_t n_points_{};
  mutable std::vector<Particle> particles_;
  mutable int tree_height_{};
  mutable std::unique_ptr<Interpolator> interpolator_;
  mutable std::unique_ptr<FarField> far_field_;
  mutable std::unique_ptr<FmmOperator> fmm_operator_;
  mutable std::unique_ptr<Tree> tree_;
};

template <class Model, class Kernel>
fmm_generic_symmetric_evaluator<Model, Kernel>::fmm_generic_symmetric_evaluator(const Model& model,
                                                                                const Bbox& bbox,
                                                                                precision prec)
    : impl_(std::make_unique<impl>(model, bbox, prec)) {}

template <class Model, class Kernel>
fmm_generic_symmetric_evaluator<Model, Kernel>::~fmm_generic_symmetric_evaluator() = default;

template <class Model, class Kernel>
common::valuesd fmm_generic_symmetric_evaluator<Model, Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Model, class Kernel>
void fmm_generic_symmetric_evaluator<Model, Kernel>::set_points(const Points& points) {
  impl_->set_points(points);
}

template <class Model, class Kernel>
void fmm_generic_symmetric_evaluator<Model, Kernel>::set_weights(
    const Eigen::Ref<const common::valuesd>& weights) {
  impl_->set_weights(weights);
}

#define IMPLEMENT_MODEL(MODEL)                                                             \
  template class fmm_generic_symmetric_evaluator<MODEL, kernel<typename MODEL::rbf_type>>; \
  template class fmm_generic_symmetric_evaluator<MODEL, hessian_kernel<typename MODEL::rbf_type>>;

#define IMPLEMENT_DIM(DIM)                                 \
  IMPLEMENT_MODEL(model<rbf::biharmonic2d<DIM>>);          \
  IMPLEMENT_MODEL(model<rbf::biharmonic3d<DIM>>);          \
  IMPLEMENT_MODEL(model<rbf::cov_exponential<DIM>>);       \
  IMPLEMENT_MODEL(model<rbf::cov_spheroidal3<DIM>>);       \
  IMPLEMENT_MODEL(model<rbf::cov_spheroidal5<DIM>>);       \
  IMPLEMENT_MODEL(model<rbf::cov_spheroidal7<DIM>>);       \
  IMPLEMENT_MODEL(model<rbf::cov_spheroidal9<DIM>>);       \
  IMPLEMENT_MODEL(model<rbf::inverse_multiquadric1<DIM>>); \
  IMPLEMENT_MODEL(model<rbf::multiquadric1<DIM>>);         \
  IMPLEMENT_MODEL(model<rbf::multiquadric3<DIM>>);         \
  IMPLEMENT_MODEL(model<rbf::triharmonic2d<DIM>>);         \
  IMPLEMENT_MODEL(model<rbf::triharmonic3d<DIM>>);

}  // namespace polatory::fmm
