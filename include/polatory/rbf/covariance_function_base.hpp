#pragma once

#include <limits>
#include <polatory/rbf/rbf_base.hpp>
#include <string>
#include <vector>

namespace polatory::rbf::internal {

template <int Dim>
class CovarianceFunctionBase : public RbfBase<Dim> {
  using Base = RbfBase<Dim>;

 public:
  using Base::Base;

  int cpd_order() const override { return 0; }

  bool is_covariance_function() const override { return true; }

  Index num_parameters() const override { return 2; }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{0.0, 0.0};
    return lower_bounds;
  }

  const std::vector<std::string>& parameter_names() const override {
    static const std::vector<std::string> names{"psill", "range"};
    return names;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{std::numeric_limits<double>::infinity(),
                                                  std::numeric_limits<double>::infinity()};
    return upper_bounds;
  }
};

enum class SpheroidalKind { kFull, kDirectPart, kFastPart };

}  // namespace polatory::rbf::internal
