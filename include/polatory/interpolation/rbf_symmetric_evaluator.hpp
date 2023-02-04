#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/fmm/fmm_tree_height.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <int Order = 10>
class rbf_symmetric_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

 public:
  rbf_symmetric_evaluator(const model& model, const geometry::points3d& points)
      : n_points_(points.rows()), n_poly_basis_(model.poly_basis_size()) {
    auto bbox = geometry::bbox3d::from_points(points);
    a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Order>>(
        model, fmm::fmm_tree_height(n_points_), bbox);
    a_->set_points(points);

    if (n_poly_basis_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_dimension(), model.poly_degree());
      p_->set_field_points(points);
    }
  }

  common::valuesd evaluate() const {
    auto y = a_->evaluate();

    if (n_poly_basis_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == n_points_ + n_poly_basis_);

    a_->set_weights(weights.head(n_points_));

    if (n_poly_basis_ > 0) {
      p_->set_weights(weights.tail(n_poly_basis_));
    }
  }

 private:
  const index_t n_points_;
  const index_t n_poly_basis_;

  std::unique_ptr<fmm::fmm_symmetric_evaluator<Order>> a_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace polatory::interpolation
