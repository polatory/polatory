#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/types.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/container/particle_container.hpp>
#include <vector>

namespace polatory::fmm {

template <class Kernel>
class FmmGenericSymmetricEvaluator<Kernel>::Impl {
  static constexpr int kDim{Kernel::kDim};
  using Bbox = geometry::Bbox<kDim>;
  using Point = geometry::Point<kDim>;
  using Points = geometry::Points<kDim>;

  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using Particle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn,
      /* variables */ Index>;

  using Container = scalfmm::container::particle_container<Particle>;

 public:
  Impl(const Rbf& rbf, const Bbox& /*bbox*/) : rbf_(rbf), kernel_(rbf) {}

  VecX evaluate() const {
    auto size = resource_->size();

    auto particles = resource_->template get_particles<Container, true>(0);
    std::vector<Index> to_idx(size);
    for (Index idx = 0; idx < size; idx++) {
      const auto p = particles.at(idx);
      auto orig_idx = std::get<0>(p.variables());
      to_idx.at(orig_idx) = idx;
    }

    auto radius = rbf_.support_radius_isotropic();
    std::vector<Index> indices;
    std::vector<double> distances;

#pragma omp parallel for schedule(guided) private(indices, distances)
    for (Index trg_idx = 0; trg_idx < size; trg_idx++) {
      auto p = particles.at(trg_idx);
      Point point;
      for (auto i = 0; i < kDim; i++) {
        point(i) = p.position(i);
      }
      kdtree_->radius_search(point, radius, indices, distances);
      for (auto src_orig_idx : indices) {
        auto src_idx = to_idx.at(src_orig_idx);
        if (src_idx == trg_idx) {
          continue;
        }
        const auto q = particles.at(src_idx);
        auto k = kernel_.evaluate(p.position(), q.position());
        for (auto i = 0; i < kn; i++) {
          for (auto j = 0; j < km; j++) {
            p.outputs(i) += q.inputs(j) * k.at(km * i + j);
          }
        }
      }
    }

    handle_self_interaction(particles);

    return potentials(particles);
  }

  void set_accuracy(double /*accuracy*/) {
    // Do nothing.
  }

  void set_resource(const Resource& resource) {
    resource_ = &resource;
    kdtree_ = resource.get_kdtree();
  }

 private:
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

  VecX potentials(const Container& particles) const {
    auto size = resource_->size();
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

  const Resource* resource_{nullptr};
  std::unique_ptr<point_cloud::KdTree<kDim>> kdtree_;
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
