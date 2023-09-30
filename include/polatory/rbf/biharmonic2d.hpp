#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <polatory/rbf/rbf_base.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::rbf {

class biharmonic2d final : public rbf_base {
 public:
  using rbf_base::rbf_base;

  explicit biharmonic2d(const std::vector<double>& params) { set_parameters(params); }

  int cpd_order() const override { return 2; }

  double evaluate_isotropic(const vector3d& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return 0.0;
    }

    return slope * r * r * std::log(r);
  }

  vector3d evaluate_gradient_isotropic(const vector3d& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return vector3d::Zero();
    }

    auto coeff = slope * (1.0 + 2.0 * std::log(r));
    return coeff * diff;
  }

  matrix3d evaluate_hessian_isotropic(const vector3d& /*diff*/) const override {
    throw std::runtime_error("biharmonic2d::evaluate_hessian_isotropic is not implemented");
  }

  int num_parameters() const override { return 1; }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{0.0};
    return lower_bounds;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{std::numeric_limits<double>::infinity()};
    return upper_bounds;
  }
};

}  // namespace polatory::rbf
