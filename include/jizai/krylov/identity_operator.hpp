#pragma once

#include <jizai/common/macros.hpp>
#include <jizai/krylov/linear_operator.hpp>
#include <jizai/types.hpp>

namespace jizai::krylov {

class IdentityOperator : public LinearOperator {
 public:
  explicit IdentityOperator(Index n) : n_(n) {}

  VecX operator()(const VecX& v) const override {
    JIZAI_ASSERT(v.rows() == n_);
    return v;
  }

  Index size() const override { return n_; }

 private:
  const Index n_;
};

}  // namespace jizai::krylov
