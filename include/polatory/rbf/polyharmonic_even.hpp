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
class PolyharmonicEven : public RbfBase<Dim> {
  using Base = RbfBase<Dim>;
  using Mat = typename Base::Mat;
  using Vector = typename Base::Vector;

  static_assert(K > 0 && K % 2 == 0, "k must be a positive even integer");

  static constexpr double kSign = (K / 2 + 1) % 2 == 0 ? 1.0 : -1.0;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit PolyharmonicEven(const std::vector<double>& params) { set_parameters(params); }

  int cpd_order() const override { return K / 2 + 1; }

  double evaluate_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return 0.0;
    }

    return kSign * slope * pow<K>(r) * std::log(r);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return Vector::Zero();
    }

    auto coeff = kSign * slope * pow<K - 2>(r) * (1.0 + K * std::log(r));
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return Mat::Zero();
    }

    auto coeff = kSign * slope * pow<K - 2>(r) * (1.0 + K * std::log(r));
    return coeff * (Mat::Identity() +
                    (K - 2.0 + K / (1.0 + K * std::log(r))) / (r * r) * diff.transpose() * diff);
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
class Biharmonic2D final : public PolyharmonicEven<Dim, 2> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "bh2";

 private:
  using Base = PolyharmonicEven<Dim, 2>;
  using RbfPtr = typename Base::RbfPtr;

 public:
  using Base::Base;

  RbfPtr clone() const override { return std::make_unique<Biharmonic2D>(*this); }

  std::string short_name() const override { return kShortName; }
};

template <int Dim>
class Triharmonic2D final : public PolyharmonicEven<Dim, 4> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "th2";

 private:
  using Base = PolyharmonicEven<Dim, 4>;
  using RbfPtr = typename Base::RbfPtr;

 public:
  using Base::Base;

  RbfPtr clone() const override { return std::make_unique<Triharmonic2D>(*this); }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(Biharmonic2D);
POLATORY_DEFINE_RBF(Triharmonic2D);

}  // namespace polatory::rbf
