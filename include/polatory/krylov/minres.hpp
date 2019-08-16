// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/krylov/gmres_base.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace krylov {

class minres : public gmres_base {
public:
  minres(const linear_operator& op, const common::valuesd& rhs, index_t max_iter);

  void iterate_process() override;

private:
  double beta_;
};

}  // namespace krylov
}  // namespace polatory
