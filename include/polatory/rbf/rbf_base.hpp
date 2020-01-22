// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace rbf {

class rbf_base {
public:
  virtual ~rbf_base() = default;

  rbf_base(rbf_base&&) = delete;
  rbf_base& operator=(const rbf_base&) = delete;
  rbf_base& operator=(rbf_base&&) = delete;

  const Eigen::Matrix3d& anisotropy() const {
    return aniso_;
  }

  virtual std::unique_ptr<rbf_base> clone() const = 0;

  // The order of conditional positive definiteness.
  virtual int cpd_order() const = 0;

  double evaluate(const geometry::vectors3d& v) const {
    return evaluate_untransformed((aniso_ * v.transpose()).norm());
  }

  virtual void evaluate_gradient_untransformed(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const = 0;

  virtual double evaluate_untransformed(double r) const = 0;

  const Eigen::Matrix3d& inverse_anisotropy() const {
    return inv_aniso_;
  }

  virtual int num_parameters() const = 0;

  virtual const std::vector<double>& parameter_lower_bounds() const = 0;

  virtual const std::vector<double>& parameter_upper_bounds() const = 0;

  const std::vector<double>& parameters() const {
    return params_;
  }

  void set_anisotropy(const Eigen::Matrix3d& aniso) {
    aniso_ = aniso;
    inv_aniso_ = aniso.inverse();
  }

  void set_parameters(const std::vector<double>& params) {
    if (static_cast<int>(params.size()) != num_parameters())
      throw std::invalid_argument("params.size() must be " + std::to_string(num_parameters()) + ".");

    params_ = params;
  }

protected:
  rbf_base()
    : aniso_(Eigen::Matrix3d::Identity())
    , inv_aniso_(Eigen::Matrix3d::Identity()) {
  }

  rbf_base(const rbf_base&) = default;

private:
  std::vector<double> params_;
  Eigen::Matrix3d aniso_;
  Eigen::Matrix3d inv_aniso_;
};

}  // namespace rbf
}  // namespace polatory
