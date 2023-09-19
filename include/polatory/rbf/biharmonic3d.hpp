#pragma once

#include <limits>
#include <memory>
#include <polatory/rbf/rbf_base.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::rbf {

class biharmonic3d final : public rbf_base {
 public:
  using rbf_base::rbf_base;

  explicit biharmonic3d(const std::vector<double>& params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override { return std::make_unique<biharmonic3d>(*this); }

  int cpd_order() const override { return 1; }

  double evaluate_isotropic(const vector3d& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    return -slope * r;
  }

  vector3d evaluate_gradient_isotropic(const vector3d& /*diff*/) const override {
    throw std::runtime_error("biharmonic3d::evaluate_gradient_isotropic() is not implemented");
  }

  matrix3d evaluate_hessian_isotropic(const vector3d& /*diff*/) const override {
    throw std::runtime_error("biharmonic3d::evaluate_hessian_isotropic() is not implemented");
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
