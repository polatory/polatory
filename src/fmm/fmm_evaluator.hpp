#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <memory>
#include <polatory/fmm/fmm_evaluator.hpp>
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
#include "lru_cache.hpp"
#include "utility.hpp"

namespace polatory::fmm {

template <class Kernel>
class FmmGenericEvaluator<Kernel>::Impl {
  static constexpr int kDim{Kernel::kDim};
  using Bbox = geometry::Bbox<kDim>;
  using Points = geometry::Points<kDim>;

  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using SourceParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn,  // should be 0
      /* variables */ Index>;

  using TargetParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,  // should be 0
      /* outputs */ double, kn,
      /* variables */ Index>;

  using SourceContainer = scalfmm::container::particle_container<SourceParticle>;
  using TargetContainer = scalfmm::container::particle_container<TargetParticle>;

  using NearField = scalfmm::operators::near_field_operator<Kernel>;
  using Interpolator = scalfmm::interpolation::interpolator<double, kDim, Kernel,
                                                            scalfmm::options::modified_uniform_>;
  using FarField = scalfmm::operators::far_field_operator<Interpolator>;
  using FmmOperator = scalfmm::operators::fmm_operators<NearField, FarField>;
  using Position = typename SourceParticle::position_type;
  using Box = scalfmm::component::box<Position>;
  using Cell = scalfmm::component::cell<typename Interpolator::storage_type>;
  using SourceLeaf = scalfmm::component::leaf_view<SourceParticle>;
  using TargetLeaf = scalfmm::component::leaf_view<TargetParticle>;
  using SourceTree = scalfmm::component::group_tree_view<Cell, SourceLeaf, Box>;
  using TargetTree = scalfmm::component::group_tree_view<Cell, TargetLeaf, Box>;

 public:
  Impl(const Rbf& rbf, const Bbox& bbox)
      : rbf_(rbf),
        bbox_(bbox),
        box_(make_box<Rbf, Box>(rbf, bbox)),
        kernel_(rbf),
        near_field_(kernel_, false) {}

  VecX evaluate() const {
    auto [src_particles, trg_particles] = prepare();

    if (config_.tree_height == 0) {
      full_direct(src_particles, trg_particles, kernel_);
      return potentials(trg_particles);
    }

    SourceTree src_tree(config_.tree_height, config_.order, box_, 10, 10, src_particles, true);
    TargetTree trg_tree(config_.tree_height, config_.order, box_, 10, 10, trg_particles, true);
    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)](src_tree, trg_tree,
                                                                          *fmm_operator_);
    return potentials(trg_tree);
  }

  void set_accuracy(double accuracy) {
    accuracy_ = accuracy;
    best_config_.clear();
  }

  void set_source_resource(const SourceResource& resource) {
    src_resource_ = &resource;
    best_config_.clear();
  }

  void set_target_resource(const TargetResource& resource) { trg_resource_ = &resource; }

 private:
  InterpolatorConfiguration find_best_configuration(int tree_height,
                                                    SourceContainer& src_particles) const {
    auto [it, inserted] = best_config_.try_emplace(tree_height);
    if (inserted) {
      auto config = FmmAccuracyEstimator<Kernel>::find_best_configuration(
          rbf_, accuracy_, src_particles, box_, tree_height);
      it->second = config;
    }

    return it->second;
  }

  VecX potentials(const TargetContainer& particles) const {
    auto size = trg_resource_->size();
    VecX potentials = VecX::Zero(kn * size);

    for (Index idx = 0; idx < size; idx++) {
      const auto p = particles.at(idx);
      auto orig_idx = std::get<0>(p.variables());
      for (auto i = 0; i < kn; i++) {
        potentials(kn * orig_idx + i) = p.outputs(i);
      }
    }

    return potentials;
  }

  VecX potentials(const TargetTree& tree) const {
    auto size = trg_resource_->size();
    VecX potentials = VecX::Zero(kn * size);

    scalfmm::component::for_each_leaf(tree.begin(), tree.end(), [&](const auto& leaf) {
      for (auto p_ref : leaf) {
        auto p = typename TargetLeaf::const_proxy_type(p_ref);
        auto idx = std::get<0>(p.variables());
        for (auto i = 0; i < kn; i++) {
          potentials(kn * idx + i) = p.outputs(i);
        }
      }
    });

    return potentials;
  }

  std::pair<SourceContainer, TargetContainer> prepare() const {
    auto src_size = src_resource_->size();
    auto trg_size = trg_resource_->size();

    if (src_size * trg_size < 1024 * 1024) {
      auto src_particles = src_resource_->template get_particles<SourceContainer, true>(0);
      auto trg_particles = trg_resource_->template get_particles<TargetContainer, false>(0);

      far_field_.reset(nullptr);
      fmm_operator_.reset(nullptr);
      config_ = {.tree_height = 0};

      return {std::move(src_particles), std::move(trg_particles)};
    }

    auto tree_height = fmm_tree_height<kDim>(std::max(src_size, trg_size));
    auto src_particles =
        src_resource_->template get_particles<SourceContainer, true>(tree_height - 1);
    auto trg_particles =
        trg_resource_->template get_particles<TargetContainer, false>(tree_height - 1);

    auto config = find_best_configuration(tree_height, src_particles);
    if (config != config_) {
      auto [it, inserted] = interpolator_cache_.try_emplace(config, kernel_, config.order,
                                                            tree_height, box_.width(0), config.d);
      interpolator_cache_.touch(it);

      far_field_ = std::make_unique<FarField>(it->second);
      fmm_operator_ = std::make_unique<FmmOperator>(near_field_, *far_field_);
      config_ = config;
    }

    return {std::move(src_particles), std::move(trg_particles)};
  }

  const Rbf& rbf_;
  const Bbox bbox_;
  const Box box_;
  const Kernel kernel_;
  const NearField near_field_;

  double accuracy_{std::numeric_limits<double>::infinity()};
  const SourceResource* src_resource_{nullptr};
  const TargetResource* trg_resource_{nullptr};
  mutable InterpolatorConfiguration config_{};
  mutable LruCache<InterpolatorConfiguration, Interpolator> interpolator_cache_{2};
  mutable std::unique_ptr<FarField> far_field_;
  mutable std::unique_ptr<FmmOperator> fmm_operator_;
  mutable std::unordered_map<int, InterpolatorConfiguration> best_config_;
};

template <class Kernel>
FmmGenericEvaluator<Kernel>::FmmGenericEvaluator(const Rbf& rbf, const Bbox& bbox)
    : impl_(std::make_unique<Impl>(rbf, bbox)) {}

template <class Kernel>
FmmGenericEvaluator<Kernel>::~FmmGenericEvaluator() = default;

template <class Kernel>
VecX FmmGenericEvaluator<Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Kernel>
void FmmGenericEvaluator<Kernel>::set_accuracy(double accuracy) {
  impl_->set_accuracy(accuracy);
}

template <class Kernel>
void FmmGenericEvaluator<Kernel>::set_source_resource(const SourceResource& resource) {
  impl_->set_source_resource(resource);
}

template <class Kernel>
void FmmGenericEvaluator<Kernel>::set_target_resource(const TargetResource& resource) {
  impl_->set_target_resource(resource);
}

#define IMPLEMENT_FMM_EVALUATORS_(RBF)                              \
  template class FmmGenericEvaluator<Kernel<RBF>>;                  \
  template class FmmGenericEvaluator<GradientKernel<RBF>>;          \
  template class FmmGenericEvaluator<GradientTransposeKernel<RBF>>; \
  template class FmmGenericEvaluator<HessianKernel<RBF>>;

#define IMPLEMENT_FMM_EVALUATORS(RBF_NAME) \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<1>);  \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<2>);  \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<3>);

}  // namespace polatory::fmm
