// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/types.hpp>

namespace polatory {
namespace krylov {

class linear_operator {
public:
  virtual ~linear_operator() = default;

  virtual common::valuesd operator()(const common::valuesd& v) const = 0;

  virtual index_t size() const = 0;
};

}  // namespace krylov
}  // namespace polatory
