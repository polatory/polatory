#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim>
class CovSpheroidal7 final : public CovarianceFunctionBase<Dim> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "sp7";

 private:
  using Base = CovarianceFunctionBase<Dim>;
  using Mat = Base::Mat;
  using RbfPtr = Base::RbfPtr;
  using Vector = Base::Vector;

  static constexpr double kRho0 = 0.2944149476843637;
  static constexpr double kA = 1.4859979204216045;
  static constexpr double kB = 0.8494862533016855;
  static constexpr double kC = 1.44208314742683;
  static constexpr double kD = 8.57520866899984;
  static constexpr double kE = 0.6934412913598931;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit CovSpheroidal7(const std::vector<double>& params) { set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<CovSpheroidal7>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < kRho0 ? psill * (1.0 - kA * rho)
                       : psill * kB * std::pow(1.0 + kC * (rho * rho), -3.5);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -4.5)) /
        (range * range);
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -4.5)) /
        (range * range);
    return coeff *
           (Mat::Identity() - (rho < kRho0 ? 1.0 / (r * r) : 9.0 / (r * r + kE * range * range)) *
                                  diff.transpose() * diff);
  }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(CovSpheroidal7);

}  // namespace polatory::rbf
