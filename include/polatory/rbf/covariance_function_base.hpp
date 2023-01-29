#pragma once

#include <limits>
#include <polatory/rbf/rbf_base.hpp>
#include <vector>

namespace polatory::rbf {

class covariance_function_base : public rbf_base {
 public:
  using rbf_base::rbf_base;

  int cpd_order() const override { return 0; }

  int num_parameters() const override { return 2; }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{0.0, 0.0};
    return lower_bounds;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{std::numeric_limits<double>::infinity(),
                                                  std::numeric_limits<double>::infinity()};
    return upper_bounds;
  }
};

}  // namespace polatory::rbf
