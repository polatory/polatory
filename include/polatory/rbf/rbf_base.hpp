#pragma once

#include <memory>
#include <polatory/geometry/point3d.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace polatory {
namespace rbf {

class rbf_base {
 public:
  virtual ~rbf_base() = default;

  rbf_base(rbf_base&&) = delete;
  rbf_base& operator=(const rbf_base&) = delete;
  rbf_base& operator=(rbf_base&&) = delete;

  const geometry::linear_transformation3d& anisotropy() const { return aniso_; }

  virtual std::unique_ptr<rbf_base> clone() const = 0;

  // The order of conditional positive definiteness.
  virtual int cpd_order() const = 0;

  double evaluate(const geometry::vectors3d& v) const {
    return evaluate_untransformed(geometry::transform_vector(aniso_, v).norm());
  }

  virtual void evaluate_gradient_untransformed(double* gradx, double* grady, double* gradz,
                                               double x, double y, double z, double r) const = 0;

  virtual double evaluate_untransformed(double r) const = 0;

  virtual int num_parameters() const = 0;

  virtual const std::vector<double>& parameter_lower_bounds() const = 0;

  virtual const std::vector<double>& parameter_upper_bounds() const = 0;

  const std::vector<double>& parameters() const { return params_; }

  void set_anisotropy(const geometry::linear_transformation3d& aniso) {
    if (aniso.determinant() <= 0.0)
      throw std::invalid_argument("aniso must have a positive determinant.");

    aniso_ = aniso;
  }

  void set_parameters(const std::vector<double>& params) {
    if (static_cast<int>(params.size()) != num_parameters())
      throw std::invalid_argument("params.size() must be " + std::to_string(num_parameters()) +
                                  ".");

    params_ = params;
  }

 protected:
  rbf_base() : aniso_(geometry::linear_transformation3d::Identity()) {}

  rbf_base(const rbf_base&) = default;

 private:
  std::vector<double> params_;
  geometry::linear_transformation3d aniso_;
};

}  // namespace rbf
}  // namespace polatory
