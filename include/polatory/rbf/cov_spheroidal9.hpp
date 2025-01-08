#pragma once

#include <cmath>
#include <limits>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim, SpheroidalKind kind>
class CovSpheroidal9Generic final : public CovarianceFunctionBase<Dim> {
 public:
  using DirectPart = CovSpheroidal9Generic<Dim, SpheroidalKind::kDirectPart>;
  using FastPart = CovSpheroidal9Generic<Dim, SpheroidalKind::kFastPart>;
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "sp9";

 private:
  using Base = CovarianceFunctionBase<Dim>;
  using Mat = typename Base::Mat;
  using RbfPtr = typename Base::RbfPtr;
  using Vector = typename Base::Vector;

  static constexpr double kRho0 = 0.31622776601683794;
  static constexpr double kA = 1.4230249470757708;
  static constexpr double kB = 0.8445585690332554;
  static constexpr double kD = 7.601027121299299;

 public:
  using Base::anisotropy;
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit CovSpheroidal9Generic(const std::vector<double>& params) { set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<CovSpheroidal9Generic>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto lin = psill * (1.0 - kA * rho);
    auto imq = psill * kB * std::pow(1.0 + rho * rho, -4.5);

    if constexpr (kind == SpheroidalKind::kDirectPart) {
      return rho < kRho0 ? lin - imq : 0.0;
    }
    if constexpr (kind == SpheroidalKind::kFastPart) {
      return imq;
    }
    return rho < kRho0 ? lin : imq;
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + rho * rho, -5.5)) /
                 (range * range);
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + rho * rho, -5.5)) /
                 (range * range);
    return coeff *
           (Mat::Identity() - (rho < kRho0 ? 1.0 / (r * r) : 11.0 / (r * r + range * range)) *
                                  diff.transpose() * diff);
  }

  std::string short_name() const override { return kShortName; }

  double support_radius_isotropic() const override {
    return kind == SpheroidalKind::kDirectPart ? kRho0 * parameters().at(1)
                                               : std::numeric_limits<double>::infinity();
  }

  DirectPart direct_part() const {
    DirectPart rbf{parameters()};
    rbf.set_anisotropy(anisotropy());
    return rbf;
  }

  FastPart fast_part() const {
    FastPart rbf{parameters()};
    rbf.set_anisotropy(anisotropy());
    return rbf;
  }
};

template <int Dim>
using CovSpheroidal9 = CovSpheroidal9Generic<Dim, SpheroidalKind::kFull>;

template <int Dim>
using CovSpheroidal9DirectPart = CovSpheroidal9Generic<Dim, SpheroidalKind::kDirectPart>;

template <int Dim>
using CovSpheroidal9FastPart = CovSpheroidal9Generic<Dim, SpheroidalKind::kFastPart>;

}  // namespace internal

POLATORY_DEFINE_RBF(CovSpheroidal9);

}  // namespace polatory::rbf
