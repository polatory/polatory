#pragma once

#include <jizai/types.hpp>

namespace jizai::krylov {

class LinearOperator {
 public:
  virtual ~LinearOperator() = default;

  LinearOperator(const LinearOperator&) = delete;
  LinearOperator(LinearOperator&&) = delete;
  LinearOperator& operator=(const LinearOperator&) = delete;
  LinearOperator& operator=(LinearOperator&&) = delete;

  virtual VecX operator()(const VecX& v) const = 0;

  virtual Index size() const = 0;

 protected:
  LinearOperator() = default;
};

}  // namespace jizai::krylov
