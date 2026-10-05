#pragma once

#include <jizai/krylov/gmres_base.hpp>
#include <jizai/krylov/linear_operator.hpp>
#include <jizai/types.hpp>

namespace jizai::krylov {

class Minres : public GmresBase {
 public:
  Minres(const LinearOperator& op, const VecX& rhs, Index max_iter);

  void iterate_process() override;

 private:
  double beta_{};
};

}  // namespace jizai::krylov
