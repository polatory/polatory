#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/precision.hpp>
#include <polatory/types.hpp>
#include <tuple>

namespace polatory::interpolation {

template <int Dim>
class rbf_residual_evaluator {
  static constexpr int kDim = Dim;
  using Bbox = geometry::bboxNd<kDim>;
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
        evaluator_(model, points, grad_points, precision::kPrecise) {}

  rbf_residual_evaluator(const Model& model, const Bbox& bbox)
      : model_(model), l_(model.poly_basis_size()), evaluator_(model, bbox, precision::kPrecise) {}

  template <class Derived, class Derived2>
  std::tuple<bool, double, double> converged(const Eigen::MatrixBase<Derived>& values,
                                             const Eigen::MatrixBase<Derived2>& weights,
                                             double absolute_tolerance,
                                             double grad_absolute_tolerance) const {
    POLATORY_ASSERT(values.rows() == mu_ + kDim * sigma_);
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    evaluator_.set_weights(weights);

    auto chunk_size = kInitialChunkSize;
    auto max_residual = 0.0;
    auto max_grad_residual = 0.0;

    auto nugget = model_.nugget();
    for (index_t begin = 0;;) {
      auto end = std::min(mu_, begin + chunk_size);
      if (begin == end) {
        break;
      }

      auto points = points_.middleRows(begin, end - begin);
      evaluator_.set_target_points(points);
      vectord fit = evaluator_.evaluate() + weights.segment(begin, end - begin) * nugget;

      auto res = (values.segment(begin, end - begin) - fit).template lpNorm<Eigen::Infinity>();
      if (res >= absolute_tolerance) {
        return {false, 0.0, 0.0};
      }

      max_residual = std::max(max_residual, res);

      begin = end;
      chunk_size *= 2;
    }

    for (index_t begin = 0;;) {
      auto end = std::min(sigma_, begin + chunk_size);
      if (begin == end) {
        break;
      }

      auto grad_points = grad_points_.middleRows(begin, end - begin);
      evaluator_.set_target_points(Points(0, kDim), grad_points);
      auto fit = evaluator_.evaluate();

      auto res = (values.segment(mu_ + kDim * begin, kDim * (end - begin)) - fit)
                     .template lpNorm<Eigen::Infinity>();
      if (res >= grad_absolute_tolerance) {
        return {false, 0.0, 0.0};
      }

      max_grad_residual = std::max(max_grad_residual, res);

      begin = end;
      chunk_size *= 2;
    }

    return {true, max_residual, max_grad_residual};
  }

  void set_points(const Points& points, const Points& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();
    points_ = points;
    grad_points_ = grad_points;

    evaluator_.set_source_points(points, grad_points);
  }

 private:
  const Model& model_;
  const index_t l_;

  index_t mu_{};
  index_t sigma_{};
  Points points_;
  Points grad_points_;
  mutable Evaluator evaluator_;
};

}  // namespace polatory::interpolation
