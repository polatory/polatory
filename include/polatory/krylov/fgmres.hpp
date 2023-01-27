#pragma once

#include <polatory/krylov/gmres.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <vector>

namespace polatory {
namespace krylov {

class fgmres : public gmres {
 public:
  fgmres(const linear_operator& op, const common::valuesd& rhs, index_t max_iter);

  void set_left_preconditioner(const linear_operator& /*left_preconditioner*/) override {
    throw std::runtime_error("set_left_preconditioner is not supported.");
  }

  common::valuesd solution_vector() const override;

 private:
  void add_preconditioned_krylov_basis(const common::valuesd& z) override;

  // zs[i] := right_preconditioned(vs[i - 1]).
  std::vector<common::valuesd> zs_;
};

}  // namespace krylov
}  // namespace polatory
