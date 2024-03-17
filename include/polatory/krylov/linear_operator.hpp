#pragma once

#include <polatory/common/types.hpp>
#include <polatory/types.hpp>

namespace polatory::krylov {

class linear_operator {
 public:
  virtual ~linear_operator() = default;

  linear_operator(const linear_operator&) = delete;
  linear_operator(linear_operator&&) = delete;
  linear_operator& operator=(const linear_operator&) = delete;
  linear_operator& operator=(linear_operator&&) = delete;

  virtual common::valuesd operator()(const common::valuesd& v) const = 0;

  virtual index_t size() const = 0;

 protected:
  linear_operator() = default;
};

}  // namespace polatory::krylov
