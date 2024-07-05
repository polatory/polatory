#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/types.hpp>
#include <tuple>

namespace polatory::interpolation {

template <int Dim>
class rbf_residual_evaluator {
  static constexpr int kDim = Dim;
  using Bbox = geometry::bboxNd<kDim>;
  using DirectEvaluator = rbf_direct_evaluator<kDim>;
  using Evaluator = rbf_evaluator<kDim>;
  using Model = model<kDim>;
  using Points = geometry::pointsNd<kDim>;

  static constexpr index_t kInitialChunkSize = 1024;

 public:
  rbf_residual_evaluator(const Model& model, const Points& points, const Points& grad_points)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        points_(points),
        grad_points_(grad_points),
        direct_evaluator_(model, points, grad_points),
        evaluator_(model, points, grad_points) {}

  rbf_residual_evaluator(const Model& model, const Bbox& bbox)
      : model_(model),
        l_(model.poly_basis_size()),
        direct_evaluator_(model),
        evaluator_(model, bbox) {}

  template <class Derived, class Derived2>
  std::tuple<bool, double, double> converged(const Eigen::MatrixBase<Derived>& values,
                                             const Eigen::MatrixBase<Derived2>& weights,
                                             double absolute_tolerance,
                                             double grad_absolute_tolerance) const {
    POLATORY_ASSERT(values.rows() == mu_ + kDim * sigma_);
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    auto residual = 0.0;
    auto grad_residual = 0.0;

    auto nugget = model_.nugget();

    // Direct evaluation is faster for small number of target points.
    // Moreover, if fast evaluation does not have enough accuracy,
    // neither does the solution, thus it is likely to be trapped here.
    {
      direct_evaluator_.set_weights(weights);

      auto trg_mu = std::min(mu_, kInitialChunkSize);
      auto trg_sigma = std::min(sigma_, kInitialChunkSize);
      Points points = points_.topRows(trg_mu);
      Points grad_points = grad_points_.topRows(trg_sigma);

      auto fit = direct_evaluator_.evaluate(points, grad_points);
      fit.head(trg_mu) += weights.head(trg_mu) * nugget;
      residual = numeric::absolute_error<Eigen::Infinity>(fit.head(trg_mu), values.head(trg_mu));
      grad_residual = numeric::absolute_error<Eigen::Infinity>(
          fit.tail(kDim * trg_sigma), values.segment(mu_, kDim * trg_sigma));

      if (residual > absolute_tolerance || grad_residual > grad_absolute_tolerance) {
        return {false, 0.0, 0.0};
      }
    }

    evaluator_.set_weights(weights);

    auto chunk_size = 2 * kInitialChunkSize;
    for (index_t begin = kInitialChunkSize;;) {
      auto end = std::min(mu_, begin + chunk_size);
      if (begin >= end) {
        break;
      }

      auto points = points_.middleRows(begin, end - begin);
      evaluator_.set_target_points(points);
      vectord fit = evaluator_.evaluate() + weights.segment(begin, end - begin) * nugget;

      residual = std::max(residual, numeric::absolute_error<Eigen::Infinity>(
                                        fit, values.segment(begin, end - begin)));
      if (residual > absolute_tolerance) {
        return {false, 0.0, 0.0};
      }

      begin = end;
      chunk_size *= 2;
    }

    chunk_size = 2 * kInitialChunkSize;
    for (index_t begin = kInitialChunkSize;;) {
      auto end = std::min(sigma_, begin + chunk_size);
      if (begin >= end) {
        break;
      }

      auto grad_points = grad_points_.middleRows(begin, end - begin);
      evaluator_.set_target_points(Points(0, kDim), grad_points);
      auto fit = evaluator_.evaluate();

      grad_residual = std::max(grad_residual,
                               numeric::absolute_error<Eigen::Infinity>(
                                   fit, values.segment(mu_ + kDim * begin, kDim * (end - begin))));
      if (grad_residual > grad_absolute_tolerance) {
        return {false, 0.0, 0.0};
      }

      begin = end;
      chunk_size *= 2;
    }

    return {true, residual, grad_residual};
  }

  void set_points(const Points& points, const Points& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();
    points_ = points;
    grad_points_ = grad_points;

    direct_evaluator_.set_source_points(points, grad_points);
    evaluator_.set_source_points(points, grad_points);
  }

 private:
  const Model& model_;
  const index_t l_;

  index_t mu_{};
  index_t sigma_{};
  Points points_;
  Points grad_points_;
  mutable DirectEvaluator direct_evaluator_;
  mutable Evaluator evaluator_;
};

}  // namespace polatory::interpolation
