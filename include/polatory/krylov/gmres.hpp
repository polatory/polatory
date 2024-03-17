#pragma once

#include <polatory/common/types.hpp>
#include <polatory/krylov/gmres_base.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/types.hpp>

namespace polatory::krylov {

class gmres : public gmres_base {
 public:
  gmres(const linear_operator& op, const common::valuesd& rhs, index_t max_iter);

  void iterate_process() override;
};

}  // namespace polatory::krylov
