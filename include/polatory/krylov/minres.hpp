#pragma once

#include <polatory/krylov/gmres_base.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/types.hpp>

namespace polatory::krylov {

class Minres : public GmresBase {
 public:
  Minres(const LinearOperator& op, const VecX& rhs, Index max_iter);

  void iterate_process() override;

 private:
  double beta_{};
};

}  // namespace polatory::krylov
