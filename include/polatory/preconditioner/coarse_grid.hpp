// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace preconditioner {

class coarse_grid {
public:
  coarse_grid(const model& model,
              const std::unique_ptr<polynomial::lagrange_basis>& lagrange_basis,
              const std::vector<index_t>& point_indices);

  coarse_grid(const model& model,
              const std::unique_ptr<polynomial::lagrange_basis>& lagrange_basis,
              const std::vector<index_t>& point_indices,
              const geometry::points3d& points_full);

  void clear();

  void setup(const geometry::points3d& points_full);

  void set_solution_to(Eigen::Ref<common::valuesd> weights_full) const;

  void solve(const Eigen::Ref<const common::valuesd>& values_full);

private:
  const model& model_;
  const std::unique_ptr<polynomial::lagrange_basis>& lagrange_basis_;
  const std::vector<index_t> point_idcs_;

  const index_t l_;
  const index_t m_;

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
