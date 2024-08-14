#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/direct_evaluator.hpp>
#include <polatory/interpolation/symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/types.hpp>
#include <random>

namespace polatory::interpolation {

struct Convergence {
  bool converged{};
  double residual{};
  double grad_residual{};
  bool exact_residual{};
  bool exact_grad_residual{};
};

template <int Dim>
class ResidualEvaluator {
  static constexpr int kDim = Dim;
  using Bbox = geometry::Bbox<kDim>;
  using DirectEvaluator = DirectEvaluator<kDim>;
  using Evaluator = SymmetricEvaluator<kDim>;
  using Model = Model<kDim>;
  using Points = geometry::Points<kDim>;

  static constexpr Index kDirectEvaluatorTargetSize = 1024;

 public:
  ResidualEvaluator(const Model& model, const Points& points, const Points& grad_points,
                    double accuracy, double grad_accuracy)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        points_(points),
        grad_points_(grad_points),
        direct_evaluator_(model, points, grad_points),
        evaluator_(model, points, grad_points, accuracy, grad_accuracy) {}

  ResidualEvaluator(const Model& model, const Bbox& bbox, double accuracy, double grad_accuracy)
      : model_(model),
        l_(model.poly_basis_size()),
        direct_evaluator_(model),
        evaluator_(model, bbox, accuracy, grad_accuracy) {}

  template <class Derived>
  Convergence converged(const Eigen::MatrixBase<Derived>& weights, double tolerance,
                        double grad_tolerance) const {
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    auto nugget = model_.nugget();

    // We must use only the direct evaluator and not the fast evaluator
    // for at least the first few iterations to ensure that the weights passed to the fast evaluator
    // do not change significantly across iterations.
    // Otherwise, we will need to recompute the optimal interpolator configurations.
    // It is also important to choose non-trivial points (points with non-zero values)
    // as the target points for the direct evaluator to avoid mistakenly concluding that
    // convergence has been attained in the zeroth iteration when the weights are zero.
    {
      direct_evaluator_.set_weights(weights);

      auto fit = direct_evaluator_.evaluate(direct_points_, direct_grad_points_);
      fit.head(direct_mu_) += weights.head(mu_)(direct_indices_) * nugget;

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

      VecX fit = evaluator_.evaluate();
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

    direct_indices_.resize(mu_);
    std::iota(direct_indices_.begin(), direct_indices_.end(), 0);
    std::shuffle(direct_indices_.begin(), direct_indices_.end(), std::mt19937{});
    std::partition(direct_indices_.begin(), direct_indices_.end(),
                   [&values](auto i) { return values(i) != 0.0; });
    direct_indices_.resize(direct_mu_);

    direct_grad_indices_.resize(sigma_);
    std::iota(direct_grad_indices_.begin(), direct_grad_indices_.end(), 0);
    std::shuffle(direct_grad_indices_.begin(), direct_grad_indices_.end(), std::mt19937{});
    std::partition(direct_grad_indices_.begin(), direct_grad_indices_.end(),
                   [this, &values](auto i) {
                     return !values.template segment<kDim>(mu_ + kDim * i).isZero();
                   });
    direct_grad_indices_.resize(direct_sigma_);

    direct_points_ = points_(direct_indices_, Eigen::all);
    direct_grad_points_ = grad_points_(direct_grad_indices_, Eigen::all);
    direct_values_ = VecX::Zero(direct_mu_ + kDim * direct_sigma_);
    direct_values_ << values_.head(mu_)(direct_indices_),
        values_.tail(kDim * sigma_)
            .reshaped<Eigen::RowMajor>(sigma_, kDim)(direct_grad_indices_, Eigen::all)
            .reshaped<Eigen::RowMajor>();
  }

 private:
  const Model& model_;
  const Index l_;

  Index mu_{};
  Index sigma_{};
  Points points_;
  Points grad_points_;
  VecX values_;
  std::vector<Index> direct_indices_;
  std::vector<Index> direct_grad_indices_;
  Index direct_mu_{};
  Index direct_sigma_{};
  Points direct_points_;
  Points direct_grad_points_;
  VecX direct_values_;
  mutable DirectEvaluator direct_evaluator_;
  mutable Evaluator evaluator_;
};

}  // namespace polatory::interpolation
