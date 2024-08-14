#pragma once

#include <cmath>
#include <limits>
#include <polatory/rbf/rbf.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <string>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim, int k>
class InverseMultiquadric : public RbfBase<Dim> {
  using Base = RbfBase<Dim>;
  using Mat = Base::Mat;
  using Vector = Base::Vector;

  static_assert(k > 0 && k % 2 == 1, "k must be a positive odd integer");

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit InverseMultiquadric(const std::vector<double>& params) { set_parameters(params); }

  int cpd_order() const override { return 0; }

  double evaluate_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto r = diff.norm();

    return slope / std::pow(std::hypot(r, c), k);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto r = diff.norm();

    auto coeff = -k * slope / std::pow(std::hypot(r, c), k + 2);
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto r = diff.norm();

    auto coeff = -k * slope / std::pow(std::hypot(r, c), k + 2);
    return coeff * (Mat::Identity() - (k + 2) / (r * r + c * c) * diff.transpose() * diff);
  }

  Index num_parameters() const override { return 2; }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{0.0, 0.0};
    return lower_bounds;
  }

  const std::vector<std::string>& parameter_names() const override {
    static const std::vector<std::string> names{"scale", "c"};
    return names;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{std::numeric_limits<double>::infinity(),
                                                  std::numeric_limits<double>::infinity()};
    return upper_bounds;
  }
};

template <int Dim>
class InverseMultiquadric1 final : public InverseMultiquadric<Dim, 1> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "imq1";

 private:
  using Base = InverseMultiquadric<Dim, 1>;
  using RbfPtr = Base::RbfPtr;

 public:
  using Base::Base;

  RbfPtr clone() const override { return std::make_unique<InverseMultiquadric1>(*this); }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(InverseMultiquadric1);

}  // namespace polatory::rbf
