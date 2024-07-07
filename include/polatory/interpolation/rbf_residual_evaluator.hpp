#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
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
  using Evaluator = rbf_symmetric_evaluator<kDim>;
  using Model = model<kDim>;
  using Points = geometry::pointsNd<kDim>;

  static constexpr index_t kDirectEvaluatorTargetSize = 1024;

 public:
  rbf_residual_evaluator(const Model& model, const Points& points, const Points& grad_points,
                         double accuracy, double grad_accuracy)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        points_(points),
        grad_points_(grad_points),
        direct_evaluator_(model, points, grad_points),
        evaluator_(model, points, grad_points, accuracy, grad_accuracy) {}

  rbf_residual_evaluator(const Model& model, const Bbox& bbox, double accuracy,
                         double grad_accuracy)
      : model_(model),
        l_(model.poly_basis_size()),
        direct_evaluator_(model),
        evaluator_(model, bbox, accuracy, grad_accuracy) {}

  template <class Derived, class Derived2>
  std::tuple<bool, double, double> converged(const Eigen::MatrixBase<Derived>& values,
                                             const Eigen::MatrixBase<Derived2>& weights,
                                             double absolute_tolerance,
                                             double grad_absolute_tolerance) const {
    POLATORY_ASSERT(values.rows() == mu_ + kDim * sigma_);
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    auto nugget = model_.nugget();

    // Use the direct evaluator for first iterations so that
    // weights passed to the fast evaluator does not change significantly;
    // otherwise, we must recompute the best order.
    {
      direct_evaluator_.set_weights(weights);

      auto trg_mu = std::min(mu_, kDirectEvaluatorTargetSize);
      auto trg_sigma = std::min(sigma_, kDirectEvaluatorTargetSize);
      Points points = points_.topRows(trg_mu);
      Points grad_points = grad_points_.topRows(trg_sigma);

      auto fit = direct_evaluator_.evaluate(points, grad_points);
      fit.head(trg_mu) += weights.head(trg_mu) * nugget;

      auto residual =
          numeric::absolute_error<Eigen::Infinity>(fit.head(trg_mu), values.head(trg_mu));
      auto grad_residual = numeric::absolute_error<Eigen::Infinity>(
          fit.tail(kDim * trg_sigma), values.segment(mu_, kDim * trg_sigma));

      if (residual > absolute_tolerance || grad_residual > grad_absolute_tolerance) {
        return {false, 0.0, 0.0};
      }

      if (trg_mu == mu_ && trg_sigma == sigma_) {
        return {true, residual, grad_residual};
      }
    }

    {
      evaluator_.set_weights(weights);

      vectord fit = evaluator_.evaluate();
      fit.head(mu_) += weights.head(mu_) * nugget;

      auto residual = numeric::absolute_error<Eigen::Infinity>(fit.head(mu_), values.head(mu_));
      auto grad_residual = numeric::absolute_error<Eigen::Infinity>(fit.tail(kDim * sigma_),
                                                                    values.tail(kDim * sigma_));

      if (residual > absolute_tolerance || grad_residual > grad_absolute_tolerance) {
        return {false, 0.0, 0.0};
      }

      return {true, residual, grad_residual};
    }
  }

  void set_points(const Points& points, const Points& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();
    points_ = points;
    grad_points_ = grad_points;

    direct_evaluator_.set_source_points(points, grad_points);
    evaluator_.set_points(points, grad_points);
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
