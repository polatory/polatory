// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>

#include <Eigen/Core>

#include <polatory/krylov/linear_operator.hpp>

namespace polatory {
namespace krylov {

class identity_operator : public linear_operator {
public:
  identity_operator(size_t n)
    : n_(n) {
  }

  Eigen::VectorXd operator()(const Eigen::VectorXd& v) const override {
    assert(v.rows() == n_);
    return v;
  }

  size_t size() const override {
    return n_;
  }

private:
  const size_t n_;
};

} // namespace krylov
} // namespace polatory
