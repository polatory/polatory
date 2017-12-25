// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <polatory/common/exception.hpp>
#include <polatory/common/types.hpp>
#include <polatory/krylov/gmres.hpp>

namespace polatory {
namespace krylov {

class fgmres : public gmres {
public:
  fgmres(const linear_operator& op, const common::valuesd& rhs, int max_iter);

  void set_left_preconditioner(const linear_operator& left_preconditioner) override {
    throw common::not_supported("set_left_preconditioner");
  }

  common::valuesd solution_vector() const override;

private:
  void add_preconditioned_krylov_basis(const common::valuesd& z) override;

  // zs[i] := right_preconditioned(vs[i - 1]).
  std::vector<common::valuesd> zs_;
};

}  // namespace krylov
}  // namespace polatory
