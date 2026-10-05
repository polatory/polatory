#pragma once

#include <jizai/krylov/gmres.hpp>
#include <jizai/krylov/linear_operator.hpp>
#include <jizai/types.hpp>
#include <stdexcept>
#include <vector>

namespace jizai::krylov {

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

}  // namespace jizai::krylov
