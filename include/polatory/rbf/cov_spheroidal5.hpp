#pragma once

#include <cmath>
#include <limits>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim, SpheroidalKind Kind>
class CovSpheroidal5Generic final : public CovarianceFunctionBase<Dim> {
 public:
  using DirectPart = CovSpheroidal5Generic<Dim, SpheroidalKind::kDirectPart>;
  using FastPart = CovSpheroidal5Generic<Dim, SpheroidalKind::kFastPart>;
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "sp5";

 private:
  using Base = CovarianceFunctionBase<Dim>;
  using Mat = typename Base::Mat;
  using RbfPtr = typename Base::RbfPtr;
  using Vector = typename Base::Vector;

  static constexpr double kRho0 = 0.2580127411803573;
  static constexpr double kA = 1.6149073288415876;
  static constexpr double kB = 0.8575980168032007;
  static constexpr double kC = 2.5036086535164204;
  static constexpr double kD = 10.735449080535068;
  static constexpr double kE = 0.39942344766841226;

 public:
  using Base::anisotropy;
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit CovSpheroidal5Generic(const std::vector<double>& params) { set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<CovSpheroidal5Generic>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto lin = [=]() -> double { return psill * (1.0 - kA * rho); };
    auto imq = [=]() -> double { return psill * kB / sqrt_pow<5>(1.0 + kC * rho * rho); };

    if constexpr (Kind == SpheroidalKind::kDirectPart) {
      return rho < kRho0 ? lin() - imq() : 0.0;
    }
    if constexpr (Kind == SpheroidalKind::kFastPart) {
      return imq();
    }
    return rho < kRho0 ? lin() : imq();
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto lin = [=, &diff]() -> Vector {
      auto coeff = -psill * kA / (r * range);
      return coeff * diff;
    };
    auto imq = [=, &diff]() -> Vector {
      auto coeff = -psill * kD / (sqrt_pow<7>(1.0 + kC * rho * rho) * range * range);
      return coeff * diff;
    };

    if constexpr (Kind == SpheroidalKind::kDirectPart) {
      return rho < kRho0 ? Vector{lin() - imq()} : Vector::Zero();
    }
    if constexpr (Kind == SpheroidalKind::kFastPart) {
      return imq();
    }
    return rho < kRho0 ? lin() : imq();
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto lin = [=, &diff]() -> Mat {
      auto coeff = -psill * kA / (r * range);
      return coeff * (Mat::Identity() - 1.0 / (r * r) * diff.transpose() * diff);
    };
    auto imq = [=, &diff]() -> Mat {
      auto coeff = -psill * kD / (sqrt_pow<7>(1.0 + kC * rho * rho) * range * range);
      return coeff *
             (Mat::Identity() - 7.0 / (r * r + kE * range * range) * diff.transpose() * diff);
    };

    if constexpr (Kind == SpheroidalKind::kDirectPart) {
      return rho < kRho0 ? Mat{lin() - imq()} : Mat::Zero();
    }
    if constexpr (Kind == SpheroidalKind::kFastPart) {
      return imq();
    }
    return rho < kRho0 ? lin() : imq();
  }

  std::string short_name() const override { return kShortName; }

  double support_radius_isotropic() const override {
    return Kind == SpheroidalKind::kDirectPart ? kRho0 * parameters().at(1)
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
using CovSpheroidal5 = CovSpheroidal5Generic<Dim, SpheroidalKind::kFull>;

template <int Dim>
using CovSpheroidal5DirectPart = CovSpheroidal5Generic<Dim, SpheroidalKind::kDirectPart>;

template <int Dim>
using CovSpheroidal5FastPart = CovSpheroidal5Generic<Dim, SpheroidalKind::kFastPart>;

}  // namespace internal

POLATORY_DEFINE_RBF(CovSpheroidal5);

}  // namespace polatory::rbf
