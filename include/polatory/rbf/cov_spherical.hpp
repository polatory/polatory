#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf_proxy.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim>
class cov_spherical final : public covariance_function_base<Dim> {
 public:
  static constexpr int kDim = Dim;

 private:
  using Base = covariance_function_base<Dim>;
  using Matrix = Base::Matrix;
  using RbfPtr = Base::RbfPtr;
  using Vector = Base::Vector;

 public:
  using Base::Base;

  explicit cov_spherical(const std::vector<double>& params) { Base::set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<cov_spherical>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return r < range ? psill * (1.0 + rho * (-1.5 + 0.5 * rho * rho)) : 0.0;
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = r < range ? psill * (-1.5 / rho + 1.5 * rho) / (range * range) : 0.0;
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& /*diff*/) const override {
    throw std::runtime_error("cov_spherical::evaluate_hessian_isotropic is not implemented");
  }

  double support_radius_isotropic() const override { return Base::parameters().at(1); }
};

}  // namespace internal

DEFINE_RBF(cov_spherical);

}  // namespace polatory::rbf
