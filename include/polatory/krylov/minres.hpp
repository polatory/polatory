// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/common/types.hpp>
#include <polatory/krylov/gmres_base.hpp>

namespace polatory {
namespace krylov {

class minres : public gmres_base {
public:
  minres(const linear_operator& op, const common::valuesd& rhs, int max_iter);

  void iterate_process() override;

private:
  double beta_;
};

} // namespace krylov
} // namespace polatory
