#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim>
class CovCubic final : public CovarianceFunctionBase<Dim> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "cub";

 private:
  using Base = CovarianceFunctionBase<Dim>;
  using Mat = typename Base::Mat;
  using RbfPtr = typename Base::RbfPtr;
  using Vector = typename Base::Vector;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit CovCubic(const std::vector<double>& params) { set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<CovCubic>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;
    auto rho2 = rho * rho;

    return r < range ? psill * (1.0 + rho2 * (-7.0 + rho * (8.75 + rho2 * (-3.5 + 0.75 * rho2))))
                     : 0.0;
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;
    auto rho2 = rho * rho;

    auto coeff =
        r < range ? psill * (-14.0 + rho * (26.25 + rho2 * (-17.5 + 5.25 * rho2))) / (range * range)
                  : 0.0;
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& /*diff*/) const override {
    throw std::runtime_error("cov_cubic::evaluate_hessian_isotropic is not implemented");
  }

  std::string short_name() const override { return kShortName; }

  double support_radius_isotropic() const override { return parameters().at(1); }
};

}  // namespace internal

POLATORY_DEFINE_RBF(CovCubic);

}  // namespace polatory::rbf
