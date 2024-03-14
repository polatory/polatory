#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::rbf {

template <int Dim>
class cov_cubic final : public covariance_function_base<Dim> {
  using Base = covariance_function_base<Dim>;
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

 public:
  using Base::Base;

  explicit cov_cubic(const std::vector<double>& params) { Base::set_parameters(params); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;
    auto rho2 = rho * rho;

    return r < range ? psill * (1.0 + rho2 * (-7.0 + rho * (8.75 + rho2 * (-3.5 + 0.75 * rho2))))
                     : 0.0;
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;
    auto rho2 = rho * rho;

    auto coeff =
        r < range ? psill * (-14.0 + rho * (26.25 + rho2 * (-17.5 + 5.25 * rho2))) / (range * range)
                  : 0.0;
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& /*diff*/) const override {
    throw std::runtime_error("cov_cubic::evaluate_hessian_isotropic is not implemented");
  }

  double support_radius_isotropic() const override { return Base::parameters().at(1); }
};

}  // namespace polatory::rbf
