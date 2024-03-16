#pragma once

#include <limits>
#include <polatory/rbf/rbf_base.hpp>
#include <vector>

namespace polatory::rbf::internal {

template <int Dim>
class covariance_function_base : public rbf_base<Dim> {
  using Base = rbf_base<Dim>;

 protected:
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

 public:
  using Base::Base;

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

}  // namespace polatory::rbf::internal
