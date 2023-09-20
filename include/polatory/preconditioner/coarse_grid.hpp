#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::preconditioner {

class coarse_grid {
 public:
  coarse_grid(const model& model, const std::vector<index_t>& point_indices);

  coarse_grid(const model& model, const std::vector<index_t>& point_indices,
              const std::vector<index_t>& grad_point_indices);

  void clear();

  void setup(const geometry::points3d& points_full, const Eigen::MatrixXd& lagrange_pt_full);

  void setup(const geometry::points3d& points_full, const geometry::points3d& grad_points_full,
             const Eigen::MatrixXd& lagrange_pt_full);

  void set_solution_to(Eigen::Ref<common::valuesd> weights_full) const;

  void solve(const Eigen::Ref<const common::valuesd>& values_full);

 private:
  const model& model_;
  const std::vector<index_t> point_idcs_;
  const std::vector<index_t> grad_point_idcs_;

  const index_t l_;
  const index_t mu_;
  const index_t sigma_;
  const index_t m_;
  index_t mu_full_;

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

}  // namespace polatory::preconditioner
