#pragma once

#include <polatory/krylov/gmres_base.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace krylov {

class gmres : public gmres_base {
 public:
  gmres(const linear_operator& op, const common::valuesd& rhs, index_t max_iter);

  void iterate_process() override;
};

}  // namespace krylov
}  // namespace polatory
