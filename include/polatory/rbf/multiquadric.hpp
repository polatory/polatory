#pragma once

#include <cmath>
#include <limits>
#include <polatory/rbf/rbf.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <string>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim, int K>
class Multiquadric : public RbfBase<Dim> {
  using Base = RbfBase<Dim>;
  using Mat = typename Base::Mat;
  using Vector = typename Base::Vector;

  static_assert(K > 0 && K % 2 == 1, "k must be a positive odd integer");

  static constexpr double kSign = ((K + 1) / 2) % 2 == 0 ? 1.0 : -1.0;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit Multiquadric(const std::vector<double>& params) { set_parameters(params); }

  int cpd_order() const override { return (K + 1) / 2; }

  double evaluate_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto r = diff.norm();

    return kSign * slope * pow<K>(std::hypot(r, c));
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto r = diff.norm();

    auto coeff = kSign * K * slope * pow<K - 2>(std::hypot(r, c));
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto r = diff.norm();

    auto coeff = kSign * K * slope * pow<K - 2>(std::hypot(r, c));
    return coeff * (Mat::Identity() + (K - 2) / (r * r + c * c) * diff.transpose() * diff);
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
class Multiquadric1 final : public Multiquadric<Dim, 1> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "mq1";

 private:
  using Base = Multiquadric<Dim, 1>;
  using RbfPtr = typename Base::RbfPtr;

 public:
  using Base::Base;

  RbfPtr clone() const override { return std::make_unique<Multiquadric1>(*this); }

  std::string short_name() const override { return kShortName; }
};

template <int Dim>
class Multiquadric3 final : public Multiquadric<Dim, 3> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "mq3";

 private:
  using Base = Multiquadric<Dim, 3>;
  using RbfPtr = typename Base::RbfPtr;

 public:
  using Base::Base;

  RbfPtr clone() const override { return std::make_unique<Multiquadric3>(*this); }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(Multiquadric1);
POLATORY_DEFINE_RBF(Multiquadric3);

}  // namespace polatory::rbf
