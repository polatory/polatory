#pragma once

#include <polatory/krylov/gmres_base.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/types.hpp>

namespace polatory::krylov {

class Gmres : public GmresBase {
 public:
  Gmres(const LinearOperator& op, const VecX& rhs, Index max_iter);

  void iterate_process() override;
};

}  // namespace polatory::krylov
