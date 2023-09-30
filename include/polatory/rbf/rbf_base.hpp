#pragma once

#include <memory>
#include <polatory/geometry/point3d.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace polatory::rbf {

template <int Dim>
class rbf_base {
 protected:
  using matrix3d = geometry::matrixNd<Dim>;
  using vector3d = geometry::vectorNd<Dim>;

 public:
  static constexpr int dimension = Dim;

  virtual ~rbf_base() = default;

  rbf_base(const rbf_base&) = default;
  rbf_base(rbf_base&&) = default;
  rbf_base& operator=(const rbf_base&) = default;
  rbf_base& operator=(rbf_base&&) = default;

  const geometry::linear_transformation3d& anisotropy() const { return aniso_; }

  // The order of conditional positive definiteness.
  virtual int cpd_order() const = 0;

  double evaluate(const vector3d& diff) const {
    auto a_diff = geometry::transform_vector(aniso_, diff);
    return evaluate_isotropic(a_diff);
  }

  vector3d evaluate_gradient(const vector3d& diff) const {
    auto a_diff = geometry::transform_vector(aniso_, diff);
    return evaluate_gradient_isotropic(a_diff) * aniso_;
  }

  matrix3d evaluate_hessian(const vector3d& diff) const {
    auto a_diff = geometry::transform_vector(aniso_, diff);
    return aniso_.transpose() * evaluate_hessian_isotropic(a_diff) * aniso_;
  }

  virtual double evaluate_isotropic(const vector3d& diff) const = 0;

  virtual vector3d evaluate_gradient_isotropic(const vector3d& diff) const = 0;

  virtual matrix3d evaluate_hessian_isotropic(const vector3d& diff) const = 0;

  virtual int num_parameters() const = 0;

  virtual const std::vector<double>& parameter_lower_bounds() const = 0;

  virtual const std::vector<double>& parameter_upper_bounds() const = 0;

  const std::vector<double>& parameters() const { return params_; }

  void set_anisotropy(const geometry::linear_transformation3d& aniso) {
    if (aniso.determinant() <= 0.0) {
      throw std::invalid_argument("aniso must have a positive determinant.");
    }

    aniso_ = aniso;
  }

  void set_parameters(const std::vector<double>& params) {
    if (static_cast<int>(params.size()) != num_parameters()) {
      throw std::invalid_argument("params.size() must be " + std::to_string(num_parameters()) +
                                  ".");
    }

    params_ = params;
  }

 protected:
  rbf_base() : aniso_(geometry::linear_transformation3d::Identity()) {}

 private:
  std::vector<double> params_;
  geometry::linear_transformation3d aniso_;
};

}  // namespace polatory::rbf
