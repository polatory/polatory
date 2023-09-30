#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <utility>

namespace polatory::interpolation {

template <class Model>
class rbf_residual_evaluator {
  static constexpr index_t kInitialChunkSize = 1024;

  static constexpr int kDim = Model::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Evaluator = rbf_evaluator<Model>;

 public:
  rbf_residual_evaluator(const Model& model, const Points& points, const Points& grad_points)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        points_(points),
        grad_points_(grad_points) {
    evaluator_ = std::make_unique<Evaluator>(model, points_, grad_points_, precision::kPrecise);
  }

  rbf_residual_evaluator(const Model& model, const Bbox& bbox)
      : model_(model), l_(model.poly_basis_size()) {
    evaluator_ = std::make_unique<Evaluator>(model, bbox, precision::kPrecise);
  }

  template <class Derived, class Derived2>
  std::pair<bool, double> converged(const Eigen::MatrixBase<Derived>& values,
                                    const Eigen::MatrixBase<Derived2>& weights,
                                    double absolute_tolerance,
                                    double grad_absolute_tolerance) const {
    POLATORY_ASSERT(values.rows() == mu_ + kDim * sigma_);
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    evaluator_->set_weights(weights);

    auto max_residual = 0.0;

    auto chunk_size = kInitialChunkSize;

    auto nugget = model_.nugget();
    for (auto begin = 0;;) {
      auto end = std::min(mu_, begin + chunk_size);
      if (begin == end) {
        break;
      }

      auto points = points_.middleRows(begin, end - begin);
      evaluator_->set_field_points(points);
      common::valuesd fit = evaluator_->evaluate() + weights.segment(begin, end - begin) * nugget;

      auto res = (values.segment(begin, end - begin) - fit).array().abs().maxCoeff();
      if (res >= absolute_tolerance) {
        return {false, 0.0};
      }

      max_residual = std::max(max_residual, res);

      begin = end;
      chunk_size *= 2;
    }

    for (auto begin = 0;;) {
      auto end = std::min(sigma_, begin + chunk_size);
      if (begin == end) {
        break;
      }

      auto grad_points = grad_points_.middleRows(begin, end - begin);
      evaluator_->set_field_points(Points(0, kDim), grad_points);
      auto fit = evaluator_->evaluate();

      auto res =
          (values.segment(mu_ + kDim * begin, kDim * (end - begin)) - fit).array().abs().maxCoeff();
      if (res >= grad_absolute_tolerance) {
        return {false, 0.0};
      }

      max_residual = std::max(max_residual, res);

      begin = end;
      chunk_size *= 2;
    }

    return {true, max_residual};
  }

  void set_points(const Points& points, const Points& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();
    points_ = points;
    grad_points_ = grad_points;

    evaluator_->set_source_points(points, grad_points);
  }

 private:
  const Model& model_;
  const index_t l_;

  index_t mu_{};
  index_t sigma_{};
  Points points_;
  Points grad_points_;

  std::unique_ptr<Evaluator> evaluator_;
};

}  // namespace polatory::interpolation
