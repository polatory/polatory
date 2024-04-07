#pragma once

#include <polatory/krylov/gmres.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::krylov {

class fgmres : public gmres {
 public:
  fgmres(const linear_operator& op, const vectord& rhs, index_t max_iter);

  void set_left_preconditioner(const linear_operator& /*left_preconditioner*/) override {
    throw std::runtime_error("set_left_preconditioner is not supported.");
  }

  vectord solution_vector() const override;

 private:
  void add_preconditioned_krylov_basis(const vectord& z) override;

  // zs[i] := right_preconditioned(vs[i - 1]).
  std::vector<vectord> zs_;
};

}  // namespace polatory::krylov
