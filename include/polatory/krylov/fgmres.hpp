// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <Eigen/Core>

#include "gmres.hpp"
#include "polatory/common/exception.hpp"

namespace polatory {
namespace krylov {

class fgmres : public gmres {
public:
  fgmres(const linear_operator& op, const Eigen::VectorXd& rhs, int max_iter);

  void set_left_preconditioner(const linear_operator& left_preconditioner) override {
    throw common::unsupported_method("set_left_preconditioner");
  }

  Eigen::VectorXd solution_vector() const override;

private:
  void add_preconditioned_krylov_basis(const Eigen::VectorXd& z) override;

  // zs[i] := right_preconditioned(vs[i - 1]).
  std::vector<Eigen::VectorXd> zs_;
};

} // namespace krylov
} // namespace polatory
