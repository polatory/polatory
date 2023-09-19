#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <polatory/rbf/rbf_base.hpp>
#include <vector>

namespace polatory::rbf {

class multiquadric1 final : public rbf_base {
 public:
  using rbf_base::rbf_base;

  explicit multiquadric1(const std::vector<double>& params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<multiquadric1>(*this);
  }

  int cpd_order() const override { return 1; }

  double evaluate_isotropic(const vector3d& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto r = diff.norm();

    return -slope * std::hypot(r, c);
  }

  vector3d evaluate_gradient_isotropic(const vector3d& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto r = diff.norm();

    auto coeff = -slope / std::hypot(r, c);
    return coeff * diff;
  }

  matrix3d evaluate_hessian_isotropic(const vector3d& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto r = diff.norm();

    auto coeff = -slope / std::hypot(r, c);
    return coeff * (matrix3d::Identity() - diff.transpose() * diff / (r * r + c * c));
  }

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
