// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace preconditioner {

class fine_grid {
public:
  fine_grid(const rbf::rbf& rbf,
            std::shared_ptr<polynomial::lagrange_basis> lagrange_basis,
            const std::vector<size_t>& point_indices,
            const std::vector<bool>& inner_point);

  fine_grid(const rbf::rbf& rbf,
            std::shared_ptr<polynomial::lagrange_basis> lagrange_basis,
            const std::vector<size_t>& point_indices,
            const std::vector<bool>& inner_point,
            const geometry::points3d& points_full);

  void clear();

  void setup(const geometry::points3d& points_full);

  template <class Derived>
  void set_solution_to(Eigen::MatrixBase<Derived>& weights_full) const {
    for (size_t i = 0; i < m_; i++) {
      if (inner_point_[i])
        weights_full(point_idcs_[i]) = lambda_(i);
    }
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
      lambda_ = common::valuesd(m_);
      lambda_.head(l_) = me_ * gamma;
      lambda_.tail(m_ - l_) = gamma;
    } else {
      lambda_ = ldlt_of_qtaq_.solve(values);
    }
  }

private:
  const rbf::rbf rbf_;
  const std::shared_ptr<polynomial::lagrange_basis> lagrange_basis_;
  const std::vector<size_t> point_idcs_;
  const std::vector<bool> inner_point_;

  const size_t l_;
  const size_t m_;

  // Matrix -E.
  Eigen::MatrixXd me_;

  // Cholesky decomposition of matrix Q^T A Q.
  Eigen::LDLT<Eigen::MatrixXd> ldlt_of_qtaq_;

  // Current solution.
  common::valuesd lambda_;
};

}  // namespace preconditioner
}  // namespace polatory
