// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace krylov {

class identity_operator : public linear_operator {
public:
  explicit identity_operator(index_t n)
    : n_(n) {
  }

  common::valuesd operator()(const common::valuesd& v) const override {
    POLATORY_ASSERT(static_cast<int>(v.rows()) == n_);
    return v;
  }

  index_t size() const override {
    return n_;
  }

private:
  const index_t n_;
};

}  // namespace krylov
}  // namespace polatory
