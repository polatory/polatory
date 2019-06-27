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

  virtual std::shared_ptr<rbf_base> clone() const = 0;

  // The order of conditional positive definiteness.
  virtual int cpd_order() const = 0;

  double evaluate(const geometry::vectors3d& v) const {
    auto ti_v = ti_.transform_vector(v);
    return evaluate_untransformed(ti_v.norm());
  }

  virtual void evaluate_gradient_untransformed(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const = 0;

  virtual double evaluate_untransformed(double r) const = 0;

  const geometry::affine_transformation3d& inverse_transformation() const {
    return ti_;
  }

  // The effect of nugget parameter is also known as spline smoothing.
  virtual double nugget() const {
    return 0.0;
  }

  virtual size_t num_parameters() const = 0;

  const std::vector<double>& parameters() const {
    return params_;
  }

  void set_parameters(const std::vector<double>& params) {
    if (params.size() != num_parameters())
      throw common::invalid_argument("params.size() == " + std::to_string(num_parameters()));

    params_ = params;
  }

  void set_transformation(const geometry::affine_transformation3d& t) {
    t_ = t;
    ti_ = t.inverse();
  }

  const geometry::affine_transformation3d& transformation() const {
    return t_;
  }

protected:
  rbf_base() = default;

private:
  std::vector<double> params_;
  geometry::affine_transformation3d t_;
  geometry::affine_transformation3d ti_;
};

}  // namespace rbf
}  // namespace polatory
