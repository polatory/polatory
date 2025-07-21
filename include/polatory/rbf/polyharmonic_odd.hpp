#pragma once

#include <limits>
#include <polatory/rbf/rbf.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <string>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim, int K>
class PolyharmonicOdd : public RbfBase<Dim> {
  using Base = RbfBase<Dim>;
  using Mat = typename Base::Mat;
  using Vector = typename Base::Vector;

  static_assert(K > 0 && K % 2 == 1, "k must be a positive odd integer");

  static constexpr double kSign = ((K + 1) / 2) % 2 == 0 ? 1.0 : -1.0;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit PolyharmonicOdd(const std::vector<double>& params) { set_parameters(params); }

  int cpd_order() const override { return (K + 1) / 2; }

  double evaluate_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto rho2 = diff.squaredNorm() + c * c;
    auto rho = std::sqrt(rho2);

    return kSign * slope * pow<K>(rho);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto rho2 = diff.squaredNorm() + c * c;
    auto rho = std::sqrt(rho2);

    if (rho == 0.0) {
      return Vector::Zero();
    }

    auto coeff = kSign * K * slope * pow<K - 2>(rho);
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto c = parameters().at(1);
    auto rho2 = diff.squaredNorm() + c * c;
    auto rho = std::sqrt(rho2);

    if (rho == 0.0) {
      return Mat::Zero();
    }

    auto coeff = kSign * K * slope * pow<K - 2>(rho);
    return coeff * (Mat::Identity() + (K - 2) / rho2 * diff.transpose() * diff);
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

  void set_parameters(const std::vector<double>& params) override {
    switch (params.size()) {
      case 0:
        Base::set_parameters({1.0, 0.0});
        break;
      case 1:
        Base::set_parameters({params.at(0), 0.0});
        break;
      default:
        Base::set_parameters(params);
        break;
    }
  }
};

template <int Dim>
class Biharmonic3D final : public PolyharmonicOdd<Dim, 1> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "bh3";

 private:
  using Base = PolyharmonicOdd<Dim, 1>;
  using RbfPtr = typename Base::RbfPtr;

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
  using RbfPtr = typename Base::RbfPtr;

 public:
  using Base::Base;

  RbfPtr clone() const override { return std::make_unique<Triharmonic3D>(*this); }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(Biharmonic3D);
POLATORY_DEFINE_RBF(Triharmonic3D);

}  // namespace polatory::rbf
