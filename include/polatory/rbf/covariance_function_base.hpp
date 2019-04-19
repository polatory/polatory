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

  double nugget() const override {
    return parameters()[2];
  }

  virtual size_t num_parameters() const {
    return 3;
  }

  virtual const std::vector<double>& parameter_lower_bounds() const {
    static const std::vector<double> lower_bounds{ 0.0, 0.0, 0.0 };
    return lower_bounds;
  }

  virtual const std::vector<double>& parameter_upper_bounds() const {
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

  double sill() const {
    return partial_sill() + nugget();
  }
};

}  // namespace rbf
}  // namespace polatory
