#pragma once

#include <cmath>
#include <limits>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/rbf/rbf_proxy.hpp>
#include <string>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim, int k>
class polyharmonic_even : public rbf_base<Dim> {
  using Base = rbf_base<Dim>;
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

  static_assert(k > 0 && k % 2 == 0, "k must be a positive even integer.");

  static constexpr double kSign = (k / 2 + 1) % 2 == 0 ? 1.0 : -1.0;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit polyharmonic_even(const std::vector<double>& params) { set_parameters(params); }

  int cpd_order() const override { return k / 2 + 1; }

  double evaluate_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return 0.0;
    }

    return kSign * slope * std::pow(r, k) * std::log(r);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return Vector::Zero();
    }

    auto coeff = kSign * slope * std::pow(r, k - 2) * (1.0 + k * std::log(r));
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return Matrix::Zero();
    }

    auto coeff = kSign * slope * std::pow(r, k - 2) * (1.0 + k * std::log(r));
    return coeff * (Matrix::Identity() +
                    (k - 2.0 + k / (1.0 + k * std::log(r))) / (r * r) * diff.transpose() * diff);
  }

  index_t num_parameters() const override { return 1; }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{0.0};
    return lower_bounds;
  }

  const std::vector<std::string>& parameter_names() const override {
    static const std::vector<std::string> names{"scale"};
    return names;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{std::numeric_limits<double>::infinity()};
    return upper_bounds;
  }
};

template <int Dim>
class biharmonic2d final : public polyharmonic_even<Dim, 2> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "bh2";

 private:
  using Base = polyharmonic_even<Dim, 2>;
  using RbfPtr = Base::RbfPtr;

 public:
  using Base::Base;

  RbfPtr clone() const override { return std::make_unique<biharmonic2d>(*this); }

  std::string short_name() const override { return kShortName; }
};

template <int Dim>
class triharmonic2d final : public polyharmonic_even<Dim, 4> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "th2";

 private:
  using Base = polyharmonic_even<Dim, 4>;
  using RbfPtr = Base::RbfPtr;

 public:
  using Base::Base;

  RbfPtr clone() const override { return std::make_unique<triharmonic2d>(*this); }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(biharmonic2d);
POLATORY_DEFINE_RBF(triharmonic2d);

}  // namespace polatory::rbf
