// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <Eigen/Core>

#include <polatory/krylov/linear_operator.hpp>

namespace polatory {
namespace krylov {

class gmres_base {
public:
  static bool print_progress;

  double absolute_residual() const;

  bool converged() const;

  virtual void iterate_process() = 0;

  int iteration_count() const;

  int max_iterations() const;

  double relative_residual() const;

  virtual void set_left_preconditioner(const linear_operator& left_preconditioner);

  void set_initial_solution(const Eigen::VectorXd& x0);

  virtual void set_right_preconditioner(const linear_operator& right_preconditioner);

  virtual void setup();

  virtual Eigen::VectorXd solution_vector() const;

  // tolerance: Tolerance of the relative residual (stopping criterion).
  void solve(double tolerance);

protected:
  gmres_base(const linear_operator& op, const Eigen::VectorXd& rhs, int max_iter);

  virtual ~gmres_base() {}

  virtual void add_preconditioned_krylov_basis(const Eigen::VectorXd& z) {}

  Eigen::VectorXd left_preconditioned(const Eigen::VectorXd x) const;

  Eigen::VectorXd right_preconditioned(const Eigen::VectorXd x) const;

  const linear_operator& op_;

  // Dimension.
  const size_t m_;

  // Maximum # of iteration.
  const int max_iter_;

  // Initial solution.
  Eigen::VectorXd x0_;

  // Left preconditioner.
  const linear_operator *left_pc_;

  // Right preconditioner.
  const linear_operator *right_pc_;

  // Current # of iteration.
  int iter_;

  // Constant (right-hand side) vector.
  const Eigen::VectorXd rhs_;

  // L2 norm of rhs.
  double rhs_norm_;

  // Orthonormal basis vectors for the Krylov subspace.
  std::vector<Eigen::VectorXd> vs_;

  // Upper triangular matrix of QR decomposition.
  Eigen::MatrixXd r_;

  // Cosines for the Givens rotations.
  Eigen::VectorXd c_;

  // Sines for the Givens rotations.
  Eigen::VectorXd s_;

  // Sequence of residuals.
  Eigen::VectorXd g_;

  bool converged_;
};

} // namespace krylov
} // namespace polatory
