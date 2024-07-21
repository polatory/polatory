#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

struct convergence {
  bool converged{};
  double residual{};
  double grad_residual{};
  bool exact_residual{};
  bool exact_grad_residual{};
};

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

  template <class Derived>
  convergence converged(const Eigen::MatrixBase<Derived>& weights, double tolerance,
                        double grad_tolerance) const {
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    auto nugget = model_.nugget();

    // Use the direct evaluator for first iterations so that
    // weights passed to the fast evaluator does not change significantly;
    // otherwise, we must recompute the best order.
    {
      direct_evaluator_.set_weights(weights);

      auto fit = direct_evaluator_.evaluate(direct_points_, direct_grad_points_);
      fit.head(direct_mu_) += weights.head(direct_mu_) * nugget;

      auto residual = numeric::absolute_error<Eigen::Infinity>(fit.head(direct_mu_),
                                                               direct_values_.head(direct_mu_));
      auto grad_residual = numeric::absolute_error<Eigen::Infinity>(
          fit.tail(kDim * direct_sigma_), direct_values_.tail(kDim * direct_sigma_));

      auto exact_residual = direct_mu_ == mu_;
      auto exact_grad_residual = direct_sigma_ == sigma_;

      if (residual > tolerance || grad_residual > grad_tolerance) {
        return {false, residual, grad_residual, exact_residual, exact_grad_residual};
      }

      if (exact_residual && exact_grad_residual) {
        return {true, residual, grad_residual, exact_residual, exact_grad_residual};
      }
    }

    {
      evaluator_.set_weights(weights);

      vectord fit = evaluator_.evaluate();
      fit.head(mu_) += weights.head(mu_) * nugget;

      auto residual = numeric::absolute_error<Eigen::Infinity>(fit.head(mu_), values_.head(mu_));
      auto grad_residual = numeric::absolute_error<Eigen::Infinity>(fit.tail(kDim * sigma_),
                                                                    values_.tail(kDim * sigma_));

      if (residual > tolerance || grad_residual > grad_tolerance) {
        return {false, residual, grad_residual, true, true};
      }

      return {true, residual, grad_residual, true, true};
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

  template <class Derived>
  void set_values(const Eigen::MatrixBase<Derived>& values) {
    POLATORY_ASSERT(values.rows() == mu_ + kDim * sigma_);

    values_ = values;
    direct_mu_ = std::min(mu_, kDirectEvaluatorTargetSize);
    direct_sigma_ = std::min(sigma_, kDirectEvaluatorTargetSize);

    std::vector<index_t> indices(mu_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partition(indices.begin(), indices.end(), [&values](auto i) { return values(i) != 0.0; });
    indices.resize(direct_mu_);

    std::vector<index_t> grad_indices(sigma_);
    std::iota(grad_indices.begin(), grad_indices.end(), 0);
    std::partition(grad_indices.begin(), grad_indices.end(), [this, &values](auto i) {
      return !values.template segment<kDim>(mu_ + kDim * i).isZero();
    });
    grad_indices.resize(direct_sigma_);

    direct_points_ = points_(indices, Eigen::all);
    direct_grad_points_ = grad_points_(grad_indices, Eigen::all);
    direct_values_ = vectord::Zero(direct_mu_ + kDim * direct_sigma_);
    direct_values_ << values_.head(mu_)(indices),
        values_.tail(kDim * sigma_)
            .reshaped<Eigen::RowMajor>(sigma_, kDim)(grad_indices, Eigen::all)
            .reshaped<Eigen::RowMajor>();
  }

 private:
  const Model& model_;
  const index_t l_;

  index_t mu_{};
  index_t sigma_{};
  Points points_;
  Points grad_points_;
  vectord values_;
  index_t direct_mu_{};
  index_t direct_sigma_{};
  Points direct_points_;
  Points direct_grad_points_;
  vectord direct_values_;
  mutable DirectEvaluator direct_evaluator_;
  mutable Evaluator evaluator_;
};

}  // namespace polatory::interpolation
