#pragma once

#include <polatory/types.hpp>

namespace polatory::krylov {

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

}  // namespace polatory::krylov
