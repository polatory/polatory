#pragma once

#include <Eigen/LU>
#include <format>
#include <limits>
#include <memory>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace polatory::rbf::internal {

template <int Dim>
class RbfBase {
 public:
  static constexpr int kDim = Dim;

 protected:
  using Mat = Mat<Dim>;
  using RbfPtr = std::unique_ptr<RbfBase<Dim>>;
  using Vector = geometry::Vector<Dim>;

 public:
  virtual ~RbfBase() = default;

  RbfBase(const RbfBase&) = default;
  RbfBase(RbfBase&&) = default;
  RbfBase& operator=(const RbfBase&) = default;
  RbfBase& operator=(RbfBase&&) = default;

  const Mat& anisotropy() const { return aniso_; }

  virtual RbfPtr clone() const = 0;

  // The order of conditional positive definiteness.
  virtual int cpd_order() const = 0;

  double evaluate(const Vector& diff) const {
    auto a_diff = geometry::transform_vector<Dim>(aniso_, diff);
    return evaluate_isotropic(a_diff);
  }

  Vector evaluate_gradient(const Vector& diff) const {
    auto a_diff = geometry::transform_vector<Dim>(aniso_, diff);
    return evaluate_gradient_isotropic(a_diff) * aniso_;
  }

  Mat evaluate_hessian(const Vector& diff) const {
    auto a_diff = geometry::transform_vector<Dim>(aniso_, diff);
    return aniso_.transpose() * evaluate_hessian_isotropic(a_diff) * aniso_;
  }

  virtual double evaluate_isotropic(const Vector& diff) const = 0;

  virtual Vector evaluate_gradient_isotropic(const Vector& diff) const = 0;

  virtual Mat evaluate_hessian_isotropic(const Vector& diff) const = 0;

  virtual bool is_covariance_function() const { return false; }

  virtual Index num_parameters() const = 0;

  virtual const std::vector<double>& parameter_lower_bounds() const = 0;

  virtual const std::vector<std::string>& parameter_names() const = 0;

  virtual const std::vector<double>& parameter_upper_bounds() const = 0;

  const std::vector<double>& parameters() const { return params_; }

  void set_anisotropy(const Mat& aniso) {
    if (!(aniso.determinant() > 0.0)) {
      throw std::invalid_argument("aniso must have a positive determinant");
    }

    aniso_ = aniso;
  }

  void set_parameters(const std::vector<double>& params) {
    if (static_cast<Index>(params.size()) != num_parameters()) {
      throw std::invalid_argument(std::format("params.size() must be {}", num_parameters()));
    }

    params_ = params;
  }

  virtual std::string short_name() const = 0;

  virtual double support_radius_isotropic() const {
    return std::numeric_limits<double>::infinity();
  }

 protected:
  RbfBase() = default;

 private:
  std::vector<double> params_;
  Mat aniso_{Mat::Identity()};
};

template <int N>
static double pow(double x) {
  if constexpr (N == -1) {
    return 1.0 / x;
  }
  if constexpr (N == 0) {
    return 1.0;
  }
  if constexpr (N == 1) {
    return x;
  }
  if constexpr (N == 2) {
    return x * x;
  }
  if constexpr (N == 3) {
    return x * x * x;
  }
  if constexpr (N == 4) {
    auto x2 = x * x;
    return x2 * x2;
  }
  return std::pow(x, N);
}

template <int N>
static double sqrt_pow(double x) {
  if constexpr (N == 3) {
    return x * std::sqrt(x);
  }
  if constexpr (N == 5) {
    return x * x * std::sqrt(x);
  }
  if constexpr (N == 7) {
    return x * x * x * std::sqrt(x);
  }
  if constexpr (N == 9) {
    auto x2 = x * x;
    return x2 * x2 * std::sqrt(x);
  }
  if constexpr (N == 11) {
    auto x2 = x * x;
    return x2 * x2 * x * std::sqrt(x);
  }
  return std::pow(x, N / 2.0);
}

}  // namespace polatory::rbf::internal
