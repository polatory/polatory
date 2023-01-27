#pragma once

#include <Eigen/Core>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory {
namespace krylov {

class gmres_base {
 public:
  static bool print_progress;

  virtual ~gmres_base() = default;

  gmres_base(const gmres_base&) = delete;
  gmres_base(gmres_base&&) = delete;
  gmres_base& operator=(const gmres_base&) = delete;
  gmres_base& operator=(gmres_base&&) = delete;

  double absolute_residual() const;

  bool converged() const;

  virtual void iterate_process() = 0;

  index_t iteration_count() const;

  index_t max_iterations() const;

  double relative_residual() const;

  virtual void set_left_preconditioner(const linear_operator& left_preconditioner);

  void set_initial_solution(const common::valuesd& x0);

  virtual void set_right_preconditioner(const linear_operator& right_preconditioner);

  virtual void setup();

  virtual common::valuesd solution_vector() const;

  // tolerance: Tolerance of the relative residual (stopping criterion).
  void solve(double tolerance);

 protected:
  gmres_base(const linear_operator& op, const common::valuesd& rhs, index_t max_iter);

  virtual void add_preconditioned_krylov_basis(const common::valuesd& /*z*/) {}

  common::valuesd left_preconditioned(const common::valuesd& x) const;

  common::valuesd right_preconditioned(const common::valuesd& x) const;

  const linear_operator& op_;

  // Dimension.
  const index_t m_;

  // Maximum # of iteration.
  const index_t max_iter_;

  // Initial solution.
  common::valuesd x0_;

  // Left preconditioner.
  const linear_operator* left_pc_;

  // Right preconditioner.
  const linear_operator* right_pc_;

  // Current # of iteration.
  index_t iter_;

  // Constant (right-hand side) vector.
  const common::valuesd rhs_;

  // L2 norm of rhs.
  double rhs_norm_;

  // Orthonormal basis vectors for the Krylov subspace.
  std::vector<common::valuesd> vs_;

  // Upper triangular matrix of QR decomposition.
  Eigen::MatrixXd r_;

  // Cosines for the Givens rotations.
  common::valuesd c_;

  // Sines for the Givens rotations.
  common::valuesd s_;

  // Sequence of residuals.
  common::valuesd g_;

  bool converged_;
};

}  // namespace krylov
}  // namespace polatory
