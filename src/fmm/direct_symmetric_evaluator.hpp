#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/types.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/container/particle_container.hpp>
#include <vector>

namespace polatory::fmm {

template <class Rbf, class Kernel>
class FmmGenericSymmetricEvaluator<Rbf, Kernel>::Impl {
  static constexpr int kDim{Rbf::kDim};
  using Bbox = geometry::Bbox<kDim>;
  using Point = geometry::Point<kDim>;
  using Points = geometry::Points<kDim>;

  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using Particle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn>;

  using Container = scalfmm::container::particle_container<Particle>;

 public:
  Impl(const Rbf& rbf, const Bbox& /*bbox*/) : rbf_(rbf), kernel_(rbf) {}

  VecX evaluate() const {
    particles_.reset_outputs();

    auto radius = rbf_.support_radius_isotropic();
    std::vector<Index> indices;
    std::vector<double> distances;

#pragma omp parallel for private(indices, distances)
    for (Index trg_idx = 0; trg_idx < n_points_; trg_idx++) {
      auto p = particles_.at(trg_idx);
      Point point;
      for (auto i = 0; i < kDim; i++) {
        point(i) = p.position(i);
      }
      kdtree_->radius_search(point, radius, indices, distances);
      for (auto src_idx : indices) {
        if (src_idx == trg_idx) {
          continue;
        }
        const auto q = particles_.at(src_idx);
        auto k = kernel_.evaluate(p.position(), q.position());
        for (auto i = 0; i < kn; i++) {
          for (auto j = 0; j < km; j++) {
            p.outputs(i) += q.inputs(j) * k.at(km * i + j);
          }
        }
      }
    }

    handle_self_interaction();

    return potentials();
  }

  void set_accuracy(double /*accuracy*/) {
    // Do nothing.
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
    }

    Points apoints = geometry::transform_points<kDim>(a, points);
    kdtree_ = std::make_unique<point_cloud::KdTree<kDim>>(apoints);
  }

  void set_weights(const Eigen::Ref<const VecX>& weights) {
    POLATORY_ASSERT(weights.rows() == km * n_points_);

    for (Index idx = 0; idx < n_points_; idx++) {
      auto p = particles_.at(idx);
      for (auto i = 0; i < km; i++) {
        p.inputs(i) = weights(km * idx + i);
      }
    }
  }

 private:
  void handle_self_interaction() const {
    if (n_points_ == 0) {
      return;
    }

    scalfmm::container::point<double, kDim> x{};
    auto k = kernel_.evaluate(x, x);

    for (Index idx = 0; idx < n_points_; idx++) {
      auto p = particles_.at(idx);
      for (auto i = 0; i < kn; i++) {
        for (auto j = 0; j < km; j++) {
          p.outputs(i) += p.inputs(j) * k.at(km * i + j);
        }
      }
    }
  }

  VecX potentials() const {
    VecX potentials = VecX::Zero(kn * n_points_);

    for (auto idx = 0; idx < n_points_; idx++) {
      const auto p = particles_.at(idx);
      for (auto i = 0; i < kn; i++) {
        potentials(kn * idx + i) = p.outputs(i);
      }
    }

    return potentials;
  }

  const Rbf& rbf_;
  const Kernel kernel_;

  Index n_points_{};
  mutable Container particles_;
  std::unique_ptr<point_cloud::KdTree<kDim>> kdtree_;
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
