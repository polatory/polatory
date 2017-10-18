// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

namespace polatory {
namespace krylov {

class linear_operator {
public:
  virtual ~linear_operator() {}

  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& v) const = 0;

  virtual size_t size() const = 0;
};

} // namespace krylov
} // namespace polatory
