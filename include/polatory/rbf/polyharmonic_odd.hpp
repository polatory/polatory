#pragma once

#include <limits>
#include <polatory/rbf/rbf.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <string>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim, int k>
class PolyharmonicOdd : public RbfBase<Dim> {
  using Base = RbfBase<Dim>;
  using Mat = Base::Mat;
  using Vector = Base::Vector;

  static_assert(k > 0 && k % 2 == 1, "k must be a positive odd integer");

  static constexpr double kSign = ((k + 1) / 2) % 2 == 0 ? 1.0 : -1.0;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit PolyharmonicOdd(const std::vector<double>& params) { set_parameters(params); }

  int cpd_order() const override { return (k + 1) / 2; }

  double evaluate_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    return kSign * slope * std::pow(r, k);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return Vector::Zero();
    }

    auto coeff = kSign * k * slope * std::pow(r, k - 2);
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return Mat::Zero();
    }

    auto coeff = kSign * k * slope * std::pow(r, k - 2);
    return coeff * (Mat::Identity() + (k - 2) / (r * r) * diff.transpose() * diff);
  }

  Index num_parameters() const override { return 1; }

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
class Biharmonic3D final : public PolyharmonicOdd<Dim, 1> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "bh3";

 private:
  using Base = PolyharmonicOdd<Dim, 1>;
  using RbfPtr = Base::RbfPtr;

 public:
  using Base::Base;

  RbfPtr clone() const override { return std::make_unique<Biharmonic3D>(*this); }

  std::string short_name() const override { return kShortName; }
};

template <int Dim>
class Triharmonic3D final : public PolyharmonicOdd<Dim, 3> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "th3";

 private:
  using Base = PolyharmonicOdd<Dim, 3>;
  using RbfPtr = Base::RbfPtr;

 public:
  using Base::Base;

  RbfPtr clone() const override { return std::make_unique<Triharmonic3D>(*this); }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(Biharmonic3D);
POLATORY_DEFINE_RBF(Triharmonic3D);

}  // namespace polatory::rbf
