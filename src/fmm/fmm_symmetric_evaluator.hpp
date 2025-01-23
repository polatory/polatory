#pragma once

#include <Eigen/Core>
#include <limits>
#include <memory>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
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
#include <tuple>
#include <unordered_map>
#include <utility>

#include "fmm_accuracy_estimator.hpp"
#include "full_direct.hpp"
#include "interpolator_configuration.hpp"
#include "utility.hpp"

namespace polatory::fmm {

template <class Kernel>
class FmmGenericSymmetricEvaluator<Kernel>::Impl {
  static constexpr int kDim{Kernel::kDim};
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
    auto particles = prepare();

    if (config_.tree_height == 0) {
      full_direct(particles, kernel_);
      handle_self_interaction(particles);
      return potentials(particles);
    }

    Tree tree(config_.tree_height, config_.order, box_, 10, 10, particles, true);
    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)](tree, *fmm_operator_);
    handle_self_interaction(tree);
    return potentials(tree);
  }

  void set_accuracy(double accuracy) {
    accuracy_ = accuracy;

    best_config_.clear();
  }

  void set_resource(const Resource& resource) {
    resource_ = &resource;
    best_config_.clear();
  }

 private:
  InterpolatorConfiguration find_best_configuration(int tree_height,
                                                    const Container& particles) const {
    auto [it, inserted] = best_config_.try_emplace(tree_height);
    if (inserted) {
      auto config = FmmAccuracyEstimator<Kernel>::find_best_configuration(
          rbf_, accuracy_, particles, box_, tree_height);
      it->second = config;
    }

    return it->second;
  }

  void handle_self_interaction(Container& particles) const {
    auto size = resource_->size();
    if (size == 0) {
      return;
    }

    scalfmm::container::point<double, kDim> x{};
    auto k = kernel_.evaluate(x, x);

    for (Index idx = 0; idx < size; idx++) {
      auto p = particles.at(idx);
      for (auto i = 0; i < kn; i++) {
        for (auto j = 0; j < km; j++) {
          p.outputs(i) += p.inputs(j) * k.at(km * i + j);
        }
      }
    }
  }

  void handle_self_interaction(Tree& tree) const {
    auto size = resource_->size();
    if (size == 0) {
      return;
    }

    scalfmm::container::point<double, kDim> x{};
    auto k = kernel_.evaluate(x, x);

    scalfmm::component::for_each_leaf(tree.begin(), tree.end(), [&](const auto& leaf) {
      for (auto p_ref : leaf) {
        auto p = typename Leaf::proxy_type(p_ref);
        for (auto i = 0; i < kn; i++) {
          for (auto j = 0; j < km; j++) {
            p.outputs(i) += p.inputs(j) * k.at(km * i + j);
          }
        }
      }
    });
  }

  VecX potentials(const Container& particles) const {
    auto size = resource_->size();
    VecX potentials = VecX::Zero(kn * size);

    for (auto idx = 0; idx < size; idx++) {
      const auto p = particles.at(idx);
      auto orig_idx = std::get<0>(p.variables());
      for (auto i = 0; i < kn; i++) {
        potentials(kn * orig_idx + i) = p.outputs(i);
      }
    }

    return potentials;
  }

  VecX potentials(const Tree& tree) const {
    auto size = resource_->size();
    VecX potentials = VecX::Zero(kn * size);

    scalfmm::component::for_each_leaf(tree.begin(), tree.end(), [&](const auto& leaf) {
      for (auto p_ref : leaf) {
        auto p = typename Leaf::const_proxy_type(p_ref);
        auto idx = std::get<0>(p.variables());
        for (auto i = 0; i < kn; i++) {
          potentials(kn * idx + i) = p.outputs(i);
        }
      }
    });

    return potentials;
  }

  Container prepare() const {
    auto size = resource_->size();

    if (size < 1024) {
      auto particles = resource_->template get_particles<Container, true>(0);

      interpolator_.reset(nullptr);
      far_field_.reset(nullptr);
      fmm_operator_.reset(nullptr);
      config_ = {.tree_height = 0};

      return std::move(particles);
    }

    auto tree_height = fmm_tree_height<kDim>(size);
    auto particles = resource_->template get_particles<Container, true>(tree_height - 1);

    auto config = find_best_configuration(tree_height, particles);
    if (config != config_) {
      interpolator_ = std::make_unique<Interpolator>(kernel_, config.order, tree_height,
                                                     box_.width(0), config.d);

      far_field_ = std::make_unique<FarField>(*interpolator_);
      fmm_operator_ = std::make_unique<FmmOperator>(near_field_, *far_field_);
      config_ = config;
    }

    return std::move(particles);
  }

  const Rbf& rbf_;
  const Bbox bbox_;
  const Box box_;
  const Kernel kernel_;
  const NearField near_field_;

  double accuracy_{std::numeric_limits<double>::infinity()};
  const Resource* resource_{nullptr};
  mutable InterpolatorConfiguration config_{};
  mutable std::unique_ptr<Interpolator> interpolator_;
  mutable std::unique_ptr<FarField> far_field_;
  mutable std::unique_ptr<FmmOperator> fmm_operator_;
  mutable std::unordered_map<int, InterpolatorConfiguration> best_config_;
};

template <class Kernel>
FmmGenericSymmetricEvaluator<Kernel>::FmmGenericSymmetricEvaluator(const Rbf& rbf, const Bbox& bbox)
    : impl_(std::make_unique<Impl>(rbf, bbox)) {}

template <class Kernel>
FmmGenericSymmetricEvaluator<Kernel>::~FmmGenericSymmetricEvaluator() = default;

template <class Kernel>
VecX FmmGenericSymmetricEvaluator<Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Kernel>
void FmmGenericSymmetricEvaluator<Kernel>::set_accuracy(double accuracy) {
  impl_->set_accuracy(accuracy);
}

template <class Kernel>
void FmmGenericSymmetricEvaluator<Kernel>::set_resource(const Resource& resource) {
  impl_->set_resource(resource);
}

#define IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF)            \
  template class FmmGenericSymmetricEvaluator<Kernel<RBF>>; \
  template class FmmGenericSymmetricEvaluator<HessianKernel<RBF>>;

#define IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(RBF_NAME) \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<1>);  \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<2>);  \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<3>);

}  // namespace polatory::fmm
