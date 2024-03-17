#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/common/types.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <scalfmm/algorithms/full_direct.hpp>
#include <scalfmm/container/particle.hpp>
#include <vector>

namespace polatory::fmm {

template <class Rbf, class Kernel>
class fmm_generic_symmetric_evaluator<Rbf, Kernel>::impl {
  static constexpr int kDim{Rbf::kDim};
  using Bbox = geometry::bboxNd<kDim>;
  using Point = geometry::pointNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using Particle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn,
      /* variables */ index_t>;

 public:
  impl(const Rbf& rbf, const Bbox& /*bbox*/, int /*order*/) : rbf_(rbf), kernel_(rbf) {}

  common::valuesd evaluate() const {
    for (auto& p : particles_) {
      for (auto i = 0; i < kn; i++) {
        p.outputs(i) = 0.0;
      }
    }

    auto radius = rbf_.support_radius_isotropic();
    std::vector<index_t> indices;
    std::vector<double> distances;

#pragma omp parallel for private(indices, distances)
    for (auto trg_idx = 0; trg_idx < n_points_; trg_idx++) {
      auto& p = particles_.at(trg_idx);
      Point point;
      for (auto i = 0; i < kDim; i++) {
        point(i) = p.position(i);
      }
      kdtree_->radius_search(point, radius, indices, distances);
      for (auto src_idx : indices) {
        if (src_idx == trg_idx) {
          continue;
        }
        const auto& q = particles_.at(src_idx);
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

  void set_points(const Points& points) {
    n_points_ = points.rows();

    particles_.resize(n_points_);

    auto a = rbf_.anisotropy();
    for (index_t idx = 0; idx < n_points_; idx++) {
      auto& p = particles_.at(idx);
      auto ap = geometry::transform_point<kDim>(a, points.row(idx));
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = ap(i);
      }
      p.variables(idx);
    }

    Points apoints = geometry::transform_points<kDim>(a, points);
    kdtree_ = std::make_unique<point_cloud::kdtree<kDim>>(apoints);
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    POLATORY_ASSERT(weights.rows() == km * n_points_);

    for (index_t idx = 0; idx < n_points_; idx++) {
      auto& p = particles_.at(idx);
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

    for (auto& p : particles_) {
      for (auto i = 0; i < kn; i++) {
        for (auto j = 0; j < km; j++) {
          p.outputs(i) += p.inputs(j) * k.at(km * i + j);
        }
      }
    }
  }

  common::valuesd potentials() const {
    common::valuesd potentials = common::valuesd::Zero(kn * n_points_);

    for (auto idx = 0; idx < n_points_; idx++) {
      const auto& p = particles_.at(idx);
      for (auto i = 0; i < kn; i++) {
        potentials(kn * idx + i) = p.outputs(i);
      }
    }

    return potentials;
  }

  const Rbf& rbf_;
  const Kernel kernel_;

  index_t n_points_{};
  mutable std::vector<Particle> particles_;
  std::unique_ptr<point_cloud::kdtree<kDim>> kdtree_;
};

template <class Rbf, class Kernel>
fmm_generic_symmetric_evaluator<Rbf, Kernel>::fmm_generic_symmetric_evaluator(const Rbf& rbf,
                                                                              const Bbox& bbox,
                                                                              int order)
    : impl_(std::make_unique<impl>(rbf, bbox, order)) {}

template <class Rbf, class Kernel>
fmm_generic_symmetric_evaluator<Rbf, Kernel>::~fmm_generic_symmetric_evaluator() = default;

template <class Rbf, class Kernel>
common::valuesd fmm_generic_symmetric_evaluator<Rbf, Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Rbf, class Kernel>
void fmm_generic_symmetric_evaluator<Rbf, Kernel>::set_points(const Points& points) {
  impl_->set_points(points);
}

template <class Rbf, class Kernel>
void fmm_generic_symmetric_evaluator<Rbf, Kernel>::set_weights(
    const Eigen::Ref<const common::valuesd>& weights) {
  impl_->set_weights(weights);
}

#define IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF)                    \
  template class fmm_generic_symmetric_evaluator<RBF, kernel<RBF>>; \
  template class fmm_generic_symmetric_evaluator<RBF, hessian_kernel<RBF>>;

#define IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(RBF_NAME) \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<1>);  \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<2>);  \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<3>);

}  // namespace polatory::fmm
