#pragma once

#include <polatory/krylov/gmres.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::krylov {

class Fgmres : public Gmres {
 public:
  Fgmres(const LinearOperator& op, const VecX& rhs, Index max_iter);

  void set_left_preconditioner(const LinearOperator& /*left_preconditioner*/) override {
    throw std::runtime_error("set_left_preconditioner is not supported");
  }

  VecX solution_vector() const override;

 private:
  void add_preconditioned_krylov_basis(const VecX& z) override;

  // zs[i] := right_preconditioned(vs[i - 1]).
  std::vector<VecX> zs_;
};

}  // namespace polatory::krylov
