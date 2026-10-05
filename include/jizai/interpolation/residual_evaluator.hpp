#pragma once

#include <Eigen/Core>
#include <jizai/common/macros.hpp>
#include <jizai/geometry/bbox3d.hpp>
#include <jizai/geometry/point3d.hpp>
#include <jizai/interpolation/symmetric_evaluator.hpp>
#include <jizai/model.hpp>
#include <jizai/numeric/error.hpp>
#include <jizai/types.hpp>

namespace jizai::interpolation {

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
  using Evaluator = SymmetricEvaluator<kDim>;
  using Model = Model<kDim>;
  using Points = geometry::Points<kDim>;

 public:
  ResidualEvaluator(const Model& model, const Points& points, const Points& grad_points,
                    double accuracy, double grad_accuracy)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        evaluator_(model, points, grad_points, accuracy, grad_accuracy) {}

  ResidualEvaluator(const Model& model, const Bbox& bbox, double accuracy, double grad_accuracy)
      : model_(model),
        l_(model.poly_basis_size()),
        evaluator_(model, bbox, accuracy, grad_accuracy) {}

  template <class Derived>
  Convergence approx_convergence(const Eigen::MatrixBase<Derived>& residual, double tolerance,
                                 double grad_tolerance) const {
    JIZAI_ASSERT(residual.rows() == mu_ + kDim * sigma_ + l_);

    auto res = residual.head(mu_).template lpNorm<Eigen::Infinity>();
    auto grad_res = residual.segment(mu_, kDim * sigma_).template lpNorm<Eigen::Infinity>();

    return {
        .converged = res <= tolerance && grad_res <= grad_tolerance,
        .residual = res,
        .grad_residual = grad_res,
        .exact_residual = false,
        .exact_grad_residual = false,
    };
  }

  template <class Derived>
  Convergence convergence(const Eigen::MatrixBase<Derived>& weights, double tolerance,
                          double grad_tolerance) const {
    JIZAI_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    evaluator_.set_weights(weights);

    VecX fit = evaluator_.evaluate();
    fit.head(mu_) += weights.head(mu_) * model_.nugget();

    auto res = numeric::absolute_error<Eigen::Infinity>(fit.head(mu_), values_.head(mu_));
    auto grad_res = numeric::absolute_error<Eigen::Infinity>(fit.tail(kDim * sigma_),
                                                             values_.tail(kDim * sigma_));

    return {
        .converged = res <= tolerance && grad_res <= grad_tolerance,
        .residual = res,
        .grad_residual = grad_res,
        .exact_residual = true,
        .exact_grad_residual = true,
    };
  }

  void set_points(const Points& points, const Points& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();

    evaluator_.set_points(points, grad_points);
  }

  template <class Derived>
  void set_values(const Eigen::MatrixBase<Derived>& values) {
    JIZAI_ASSERT(values.rows() == mu_ + kDim * sigma_);

    values_ = values;
  }

 private:
  const Model& model_;
  const Index l_;

  Index mu_{};
  Index sigma_{};
  mutable Evaluator evaluator_;
  VecX values_;
};

}  // namespace jizai::interpolation
