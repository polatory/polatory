#pragma once

#include <Eigen/LU>
#include <format>
#include <limits>
#include <memory>
#include <polatory/fmm/interpolator_configuration.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace polatory::rbf::internal {

template <int Dim>
class rbf_base {
 public:
  static constexpr int kDim = Dim;

 protected:
  using Matrix = geometry::matrixNd<Dim>;
  using RbfPtr = std::unique_ptr<rbf_base<Dim>>;
  using Vector = geometry::vectorNd<Dim>;

 public:
  virtual ~rbf_base() = default;

  rbf_base(const rbf_base&) = default;
  rbf_base(rbf_base&&) = default;
  rbf_base& operator=(const rbf_base&) = default;
  rbf_base& operator=(rbf_base&&) = default;

  const Matrix& anisotropy() const { return aniso_; }

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

  Matrix evaluate_hessian(const Vector& diff) const {
    auto a_diff = geometry::transform_vector<Dim>(aniso_, diff);
    return aniso_.transpose() * evaluate_hessian_isotropic(a_diff) * aniso_;
  }

  virtual double evaluate_isotropic(const Vector& diff) const = 0;

  virtual Vector evaluate_gradient_isotropic(const Vector& diff) const = 0;

  virtual Matrix evaluate_hessian_isotropic(const Vector& diff) const = 0;

  const fmm::interpolator_configuration& interpolator_configuration() const { return config_; }

  virtual bool is_covariance_function() const { return false; }

  virtual index_t num_parameters() const = 0;

  virtual const std::vector<double>& parameter_lower_bounds() const = 0;

  virtual const std::vector<std::string>& parameter_names() const = 0;

  virtual const std::vector<double>& parameter_upper_bounds() const = 0;

  const std::vector<double>& parameters() const { return params_; }

  void set_anisotropy(const Matrix& aniso) {
    if (!(aniso.determinant() > 0.0)) {
      throw std::invalid_argument("aniso must have a positive determinant");
    }

    aniso_ = aniso;
  }

  void set_interpolator_configuration(const fmm::interpolator_configuration& config) {
    config_ = config;
  }

  void set_parameters(const std::vector<double>& params) {
    if (static_cast<index_t>(params.size()) != num_parameters()) {
      throw std::invalid_argument(std::format("params.size() must be {}", num_parameters()));
    }

    params_ = params;
  }

  virtual std::string short_name() const = 0;

  virtual double support_radius_isotropic() const {
    return std::numeric_limits<double>::infinity();
  }

 protected:
  rbf_base() = default;

 private:
  std::vector<double> params_;
  Matrix aniso_{Matrix::Identity()};
  fmm::interpolator_configuration config_{8, fmm::interpolator_configuration::kClassic};
};

}  // namespace polatory::rbf::internal
