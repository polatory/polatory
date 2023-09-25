#pragma once

#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

class rbf_direct_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

 public:
  rbf_direct_evaluator(const model& model, const geometry::points3d& source_points);

  rbf_direct_evaluator(const model& model, const geometry::points3d& source_points,
                       const geometry::points3d& source_grad_points);

  common::valuesd evaluate() const;

  void set_field_points(const geometry::points3d& field_points);

  void set_field_points(const geometry::points3d& field_points,
                        const geometry::points3d& field_grad_points);

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == mu_ + dim_ * sigma_ + l_);

    weights_ = weights;

    if (l_ > 0) {
      p_->set_weights(weights.tail(l_));
    }
  }

 private:
  const model& model_;
  const int dim_;
  const index_t l_;
  const index_t mu_;
  const index_t sigma_;
  const geometry::points3d src_points_;
  const geometry::points3d src_grad_points_;

  std::unique_ptr<PolynomialEvaluator> p_;

  index_t fld_mu_{};
  index_t fld_sigma_{};
  geometry::points3d fld_points_;
  geometry::points3d fld_grad_points_;
  common::valuesd weights_;
};

}  // namespace polatory::interpolation
