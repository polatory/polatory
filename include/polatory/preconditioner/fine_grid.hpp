#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::preconditioner {

class fine_grid {
 public:
  fine_grid(const model& model, domain&& domain);

  void clear();

  void setup(const geometry::points3d& points_full, const Eigen::MatrixXd& lagrange_pt_full);

  void setup(const geometry::points3d& points_full, const geometry::points3d& grad_points_full,
             const Eigen::MatrixXd& lagrange_pt_full);

  void set_solution_to(common::valuesd& weights_full) const;

  void solve(const common::valuesd& values_full);

 private:
  const model& model_;
  const std::vector<index_t> point_idcs_;
  const std::vector<index_t> grad_point_idcs_;
  const std::vector<bool> inner_point_;
  const std::vector<bool> inner_grad_point_;

  const index_t dim_;
  const index_t l_;
  const index_t mu_;
  const index_t sigma_;
  const index_t m_;
  index_t mu_full_;
  index_t sigma_full_;

  // Matrix -E.
  Eigen::MatrixXd me_;

  // Cholesky decomposition of matrix Q^T A Q.
  Eigen::LDLT<Eigen::MatrixXd> ldlt_of_qtaq_;

  // Current solution.
  common::valuesd lambda_;
};

}  // namespace polatory::preconditioner
