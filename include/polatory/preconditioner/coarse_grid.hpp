// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace preconditioner {

class coarse_grid {
public:
  coarse_grid(const rbf::rbf& rbf,
              std::shared_ptr<polynomial::lagrange_basis> lagrange_basis,
              const std::vector<size_t>& point_indices);

  coarse_grid(const rbf::rbf& rbf,
              std::shared_ptr<polynomial::lagrange_basis> lagrange_basis,
              const std::vector<size_t>& point_indices,
              const geometry::points3d& points_full);

  void clear();

  void setup(const geometry::points3d& points_full);

  template <class Derived>
  void set_solution_to(Eigen::MatrixBase<Derived>& weights_full) const {
    for (size_t i = 0; i < m_; i++) {
      weights_full(point_idcs_[i]) = lambda_c_(i);
    }

    weights_full.tail(l_) = lambda_c_.tail(l_).template cast<double>();
  }

  template <class Derived>
  void solve(const Eigen::MatrixBase<Derived>& values_full) {
    common::valuesd values(m_);
    for (size_t i = 0; i < m_; i++) {
      values(i) = values_full(point_idcs_[i]);
    }

    if (l_ > 0) {
      // Compute Q^T d.
      common::valuesd qtd = me_.transpose() * values.head(l_)
                            + values.tail(m_ - l_);

      // Solve Q^T A Q gamma = Q^T d for gamma.
      common::valuesd gamma = ldlt_of_qtaq_.solve(qtd);

      // Compute lambda = Q gamma.
      lambda_c_ = common::valuesd(m_ + l_);
      lambda_c_.head(l_) = me_ * gamma;
      lambda_c_.segment(l_, m_ - l_) = gamma;

      // Solve P c = d - A lambda for c at poly_points.
      common::valuesd a_top_lambda = a_top_ * lambda_c_.head(m_);
      lambda_c_.tail(l_) = lu_of_p_top_.solve(values.head(l_) - a_top_lambda);
    } else {
      lambda_c_ = ldlt_of_qtaq_.solve(values);
    }
  }

private:
  const rbf::rbf rbf_;
  const std::shared_ptr<polynomial::lagrange_basis> lagrange_basis_;
  const std::vector<size_t> point_idcs_;

  const size_t l_;
  const size_t m_;

  // Matrix -E.
  Eigen::MatrixXd me_;

  // Cholesky decomposition of matrix Q^T A Q.
  Eigen::LDLT<Eigen::MatrixXd> ldlt_of_qtaq_;

  // First l rows of matrix A.
  Eigen::MatrixXd a_top_;

  // LU decomposition of first l rows of matrix P.
  Eigen::FullPivLU<Eigen::MatrixXd> lu_of_p_top_;

  // Current solution.
  common::valuesd lambda_c_;
};

}  // namespace preconditioner
}  // namespace polatory
