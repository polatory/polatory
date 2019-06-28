// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <limits>
#include <vector>

#include <polatory/rbf/rbf_base.hpp>

namespace polatory {
namespace rbf {

class covariance_function_base : public rbf_base {
public:
  using rbf_base::rbf_base;

  int cpd_order() const override {
    return 0;
  }

  size_t num_parameters() const override {
    return 2;
  }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{ 0.0, 0.0, 0.0 };
    return lower_bounds;
  }

  const std::vector<double>& parameter_upper_bounds() const override  {
    static const std::vector<double> upper_bounds{ std::numeric_limits<double>::infinity(),
                                                   std::numeric_limits<double>::infinity(),
                                                   std::numeric_limits<double>::infinity() };
    return upper_bounds;
  }

  double partial_sill() const {
    return parameters()[0];
  }

  double range() const {
    return parameters()[1];
  }
};

}  // namespace rbf
}  // namespace polatory
