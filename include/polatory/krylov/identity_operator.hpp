#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/types.hpp>

namespace polatory::krylov {

class IdentityOperator : public LinearOperator {
 public:
  explicit IdentityOperator(Index n) : n_(n) {}

  VecX operator()(const VecX& v) const override {
    POLATORY_ASSERT(v.rows() == n_);
    return v;
  }

  Index size() const override { return n_; }

 private:
  const Index n_;
};

}  // namespace polatory::krylov
