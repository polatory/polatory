#pragma once

#include <polatory/krylov/linear_operator.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::krylov {

class GmresBase {
 public:
  virtual ~GmresBase() = default;

  GmresBase(const GmresBase&) = delete;
  GmresBase(GmresBase&&) = delete;
  GmresBase& operator=(const GmresBase&) = delete;
  GmresBase& operator=(GmresBase&&) = delete;

  double absolute_residual() const;

  bool converged() const;

  virtual void iterate_process() = 0;

  Index iteration_count() const;

  Index max_iterations() const;

  double relative_residual() const;

  virtual void set_left_preconditioner(const LinearOperator& left_preconditioner);

  void set_initial_solution(const VecX& x0);

  virtual void set_right_preconditioner(const LinearOperator& right_preconditioner);

  virtual void setup();

  virtual VecX solution_vector() const;

 protected:
  GmresBase(const LinearOperator& op, const VecX& rhs, Index max_iter);

  virtual void add_preconditioned_krylov_basis(const VecX& /*z*/) {}

  VecX left_preconditioned(const VecX& x) const;

  VecX right_preconditioned(const VecX& x) const;

  const LinearOperator& op_;

  // Dimension.
  const Index m_;

  // Maximum # of iteration.
  const Index max_iter_;

  // Initial solution.
  VecX x0_;

  // Left preconditioner.
  const LinearOperator* left_pc_{};

  // Right preconditioner.
  const LinearOperator* right_pc_{};

  // Current # of iteration.
  Index iter_{};

  // Constant (right-hand side) vector.
  const VecX rhs_;

  // L2 norm of rhs.
  double rhs_norm_;

  // Orthonormal basis vectors for the Krylov subspace.
  std::vector<VecX> vs_;

  // Upper triangular matrix of QR decomposition.
  MatX r_;

  // Cosines for the Givens rotations.
  VecX c_;

  // Sines for the Givens rotations.
  VecX s_;

  // Sequence of residuals.
  VecX g_;

  bool converged_{};
};

}  // namespace polatory::krylov
