#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/types.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/container/particle_container.hpp>
#include <vector>

namespace polatory::fmm {

template <class Rbf, class Kernel>
class fmm_generic_evaluator<Rbf, Kernel>::impl {
  static constexpr int kDim{Rbf::kDim};
  using Bbox = geometry::bboxNd<kDim>;
  using Point = geometry::pointNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using SourceParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn>;  // should be 0

  using TargetParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,  // should be 0
      /* outputs */ double, kn>;

  using SourceContainer = scalfmm::container::particle_container<SourceParticle>;
  using TargetContainer = scalfmm::container::particle_container<TargetParticle>;

 public:
  impl(const Rbf& rbf, const Bbox& /*bbox*/) : rbf_(rbf), kernel_(rbf) {}

  vectord evaluate() const {
    trg_particles_.reset_outputs();

    auto radius = rbf_.support_radius_isotropic();
    std::vector<index_t> indices;
    std::vector<double> distances;

#pragma omp parallel for private(indices, distances)
    for (index_t trg_idx = 0; trg_idx < n_trg_points_; trg_idx++) {
      auto p = trg_particles_.at(trg_idx);
      Point point;
      for (auto i = 0; i < kDim; i++) {
        point(i) = p.position(i);
      }
      kdtree_->radius_search(point, radius, indices, distances);
      for (auto src_idx : indices) {
        const auto q = src_particles_.at(src_idx);
        auto k = kernel_.evaluate(p.position(), q.position());
        for (auto i = 0; i < kn; i++) {
          for (auto j = 0; j < km; j++) {
            p.outputs(i) += q.inputs(j) * k.at(km * i + j);
          }
        }
      }
    }

    return potentials();
  }

  void set_accuracy(double /*accuracy*/) {
    // Do nothing.
  }

  void set_source_points(const Points& points) {
    n_src_points_ = points.rows();

    src_particles_.resize(n_src_points_);

    auto a = rbf_.anisotropy();
    for (index_t idx = 0; idx < n_src_points_; idx++) {
      auto p = src_particles_.at(idx);
      auto ap = geometry::transform_point<kDim>(a, points.row(idx));
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = ap(i);
      }
    }

    Points apoints = geometry::transform_points<kDim>(a, points);
    kdtree_ = std::make_unique<point_cloud::kdtree<kDim>>(apoints);
  }

  void set_target_points(const Points& points) {
    n_trg_points_ = points.rows();

    trg_particles_.resize(n_trg_points_);

    auto a = rbf_.anisotropy();
    for (index_t idx = 0; idx < n_trg_points_; idx++) {
      auto p = trg_particles_.at(idx);
      auto ap = geometry::transform_point<kDim>(a, points.row(idx));
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = ap(i);
      }
    }
  }

  void set_weights(const Eigen::Ref<const vectord>& weights) {
    POLATORY_ASSERT(weights.rows() == km * n_src_points_);

    for (index_t idx = 0; idx < n_src_points_; idx++) {
      auto p = src_particles_.at(idx);
      for (auto i = 0; i < km; i++) {
        p.inputs(i) = weights(km * idx + i);
      }
    }
  }

 private:
  vectord potentials() const {
    vectord potentials = vectord::Zero(kn * n_trg_points_);

    for (index_t idx = 0; idx < n_trg_points_; idx++) {
      const auto p = trg_particles_.at(idx);
      for (auto i = 0; i < kn; i++) {
        potentials(kn * idx + i) = p.outputs(i);
      }
    }

    return potentials;
  }

  const Rbf& rbf_;
  const Kernel kernel_;

  index_t n_src_points_{};
  index_t n_trg_points_{};
  mutable SourceContainer src_particles_;
  mutable TargetContainer trg_particles_;
  std::unique_ptr<point_cloud::kdtree<kDim>> kdtree_;
};

template <class Rbf, class Kernel>
fmm_generic_evaluator<Rbf, Kernel>::fmm_generic_evaluator(const Rbf& rbf, const Bbox& bbox)
    : impl_(std::make_unique<impl>(rbf, bbox)) {}

template <class Rbf, class Kernel>
fmm_generic_evaluator<Rbf, Kernel>::~fmm_generic_evaluator() = default;

template <class Rbf, class Kernel>
vectord fmm_generic_evaluator<Rbf, Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Rbf, class Kernel>
void fmm_generic_evaluator<Rbf, Kernel>::set_accuracy(double accuracy) {
  impl_->set_accuracy(accuracy);
}

template <class Rbf, class Kernel>
void fmm_generic_evaluator<Rbf, Kernel>::set_source_points(const Points& points) {
  impl_->set_source_points(points);
}

template <class Rbf, class Kernel>
void fmm_generic_evaluator<Rbf, Kernel>::set_target_points(const Points& points) {
  impl_->set_target_points(points);
}

template <class Rbf, class Kernel>
void fmm_generic_evaluator<Rbf, Kernel>::set_weights(const Eigen::Ref<const vectord>& weights) {
  impl_->set_weights(weights);
}

#define IMPLEMENT_FMM_EVALUATORS_(RBF)                                       \
  template class fmm_generic_evaluator<RBF, kernel<RBF>>;                    \
  template class fmm_generic_evaluator<RBF, gradient_kernel<RBF>>;           \
  template class fmm_generic_evaluator<RBF, gradient_transpose_kernel<RBF>>; \
  template class fmm_generic_evaluator<RBF, hessian_kernel<RBF>>;

#define IMPLEMENT_FMM_EVALUATORS(RBF_NAME) \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<1>);  \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<2>);  \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<3>);

}  // namespace polatory::fmm
