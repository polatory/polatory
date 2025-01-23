#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/types.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/container/particle_container.hpp>
#include <vector>

namespace polatory::fmm {

template <class Kernel>
class FmmGenericEvaluator<Kernel>::Impl {
  static constexpr int kDim{Kernel::kDim};
  using Bbox = geometry::Bbox<kDim>;
  using Point = geometry::Point<kDim>;
  using Points = geometry::Points<kDim>;

  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using SourceParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn,
      /* variables */ Index>;  // should be 0

  using TargetParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,  // should be 0
      /* outputs */ double, kn,
      /* variables */ Index>;

  using SourceContainer = scalfmm::container::particle_container<SourceParticle>;
  using TargetContainer = scalfmm::container::particle_container<TargetParticle>;

 public:
  Impl(const Rbf& rbf, const Bbox& /*bbox*/) : rbf_(rbf), kernel_(rbf) {}

  VecX evaluate() const {
    auto src_size = src_resource_->size();
    auto trg_size = trg_resource_->size();

    auto src_particles = src_resource_->template get_particles<SourceContainer, true>(0);
    auto trg_particles = trg_resource_->template get_particles<TargetContainer, false>(0);

    std::vector<Index> to_idx(src_size);
    for (Index idx = 0; idx < src_size; idx++) {
      const auto p = src_particles.at(idx);
      auto orig_idx = std::get<0>(p.variables());
      to_idx.at(orig_idx) = idx;
    }

    auto radius = rbf_.support_radius_isotropic();
    std::vector<Index> indices;
    std::vector<double> distances;

#pragma omp parallel for schedule(guided) private(indices, distances)
    for (Index trg_idx = 0; trg_idx < trg_size; trg_idx++) {
      auto p = trg_particles.at(trg_idx);
      Point point;
      for (auto i = 0; i < kDim; i++) {
        point(i) = p.position(i);
      }
      kdtree_->radius_search(point, radius, indices, distances);
      for (auto src_orig_idx : indices) {
        auto src_idx = to_idx.at(src_orig_idx);
        const auto q = src_particles.at(src_idx);
        auto k = kernel_.evaluate(p.position(), q.position());
        for (auto i = 0; i < kn; i++) {
          for (auto j = 0; j < km; j++) {
            p.outputs(i) += q.inputs(j) * k.at(km * i + j);
          }
        }
      }
    }

    return potentials(trg_particles);
  }

  void set_accuracy(double /*accuracy*/) {
    // Do nothing.
  }

  void set_source_resource(const SourceResource& resource) {
    src_resource_ = &resource;
    kdtree_ = resource.get_kdtree();
  }

  void set_target_resource(const TargetResource& resource) { trg_resource_ = &resource; }

 private:
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

  const Rbf& rbf_;
  const Kernel kernel_;

  const SourceResource* src_resource_{nullptr};
  const TargetResource* trg_resource_{nullptr};
  std::unique_ptr<point_cloud::KdTree<kDim>> kdtree_;
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
