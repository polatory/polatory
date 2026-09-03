#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim>
class CovSpherical final : public CovarianceFunctionBase<Dim> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "sph";

 private:
  using Base = CovarianceFunctionBase<Dim>;
  using Mat = Base::Mat;
  using RbfPtr = Base::RbfPtr;
  using Vector = Base::Vector;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit CovSpherical(const std::vector<double>& params) { set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<CovSpherical>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return r < range ? psill * (1.0 + rho * (-1.5 + 0.5 * rho * rho)) : 0.0;
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = r < range ? psill * (-1.5 / rho + 1.5 * rho) / (range * range) : 0.0;
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& /*diff*/) const override {
    throw std::runtime_error("cov_spherical::evaluate_hessian_isotropic is not implemented");
  }

  std::string short_name() const override { return kShortName; }

  double support_radius_isotropic() const override { return parameters().at(1); }
};

}  // namespace internal

POLATORY_DEFINE_RBF(CovSpherical);

}  // namespace polatory::rbf
