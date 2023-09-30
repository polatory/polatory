#pragma once

#include <limits>
#include <memory>
#include <polatory/rbf/rbf_base.hpp>
#include <vector>

namespace polatory {
namespace rbf {
namespace reference {

template <int Dim>
class triharmonic3d final : public rbf_base<Dim> {
  using Base = rbf_base<Dim>;
  using matrix3d = Base::matrix3d;
  using vector3d = Base::vector3d;

 public:
  using Base::Base;

  explicit triharmonic3d(const std::vector<double>& params) { Base::set_parameters(params); }

  int cpd_order() const override { return 2; }

  double evaluate_isotropic(const vector3d& diff) const override {
    auto slope = Base::parameters().at(0);
    auto r = diff.norm();

    return slope * r * r * r;
  }

  vector3d evaluate_gradient_isotropic(const vector3d& diff) const override {
    auto slope = Base::parameters().at(0);
    auto r = diff.norm();

    auto coeff = 3.0 * slope * r;
    return coeff * diff;
  }

  matrix3d evaluate_hessian_isotropic(const vector3d& diff) const override {
    auto slope = Base::parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return matrix3d::Zero();
    }

    auto coeff = 3.0 * slope * r;
    return coeff * (matrix3d::Identity() + diff.transpose() * diff / (r * r));
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

}  // namespace reference
}  // namespace rbf
}  // namespace polatory
