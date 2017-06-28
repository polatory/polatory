// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>

#include <Eigen/Core>

#include "linear_operator.hpp"

namespace polatory {
namespace krylov {

struct identity_operator : linear_operator {
   const size_t n;

   identity_operator(size_t n)
      : n(n)
   {
   }

   Eigen::VectorXd operator()(const Eigen::VectorXd& v) const override
   {
      assert(v.size() == n);
      return v;
   }

   size_t size() const override
   {
      return n;
   }
};

} // namespace krylov
} // namespace polatory
