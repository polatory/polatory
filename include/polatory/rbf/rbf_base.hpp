// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <polatory/common/exception.hpp>
#include <polatory/geometry/affine_transformation3d.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace rbf {

class rbf_base {
public:
  virtual ~rbf_base() = default;

  rbf_base(const rbf_base&) = delete;
  rbf_base(rbf_base&&) = delete;
  rbf_base& operator=(const rbf_base&) = delete;
  rbf_base& operator=(rbf_base&&) = delete;

  const geometry::affine_transformation3d& affine_transformation() const {
    return t_;
  }

  virtual std::shared_ptr<rbf_base> clone() const = 0;

  // The order of conditional positive definiteness.
  virtual int cpd_order() const = 0;

  double evaluate(const geometry::vectors3d& v) const {
    auto t_v = t_.transform_vector(v);
    return evaluate_transformed(t_v.norm());
  }

  virtual void evaluate_gradient_transformed(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const = 0;

  virtual double evaluate_transformed(double r) const = 0;

  // The effect of nugget parameter is also known as spline smoothing.
  virtual double nugget() const {
    return 0.0;
  }

  virtual size_t num_parameters() const = 0;

  const std::vector<double>& parameters() const {
    return params_;
  }

  void set_affine_transformation(const geometry::affine_transformation3d& t) {
    t_ = t;
  }

  void set_parameters(const std::vector<double>& params) {
    if (params.size() != num_parameters())
      throw common::invalid_argument("params.size() == " + std::to_string(num_parameters()));

    params_ = params;
  }

protected:
  rbf_base() = default;

private:
  std::vector<double> params_;
  geometry::affine_transformation3d t_;
};

}  // namespace rbf
}  // namespace polatory
