// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_solver.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace interpolation {

class rbf_fitter : rbf_solver {
public:
  using rbf_solver::set_points;

  rbf_fitter(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
             const geometry::points3d& points)
    : rbf_solver(rbf, poly_dimension, poly_degree, points) {
  }

  template <class Derived>
  common::valuesd fit(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance) const {
    return solve(values, absolute_tolerance);
  }

  template <class Derived, class Derived2>
  common::valuesd fit(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance,
                      const Eigen::MatrixBase<Derived2>& initial_solution) const {
    return solve(values, absolute_tolerance, initial_solution);
  }
};

} // namespace interpolation
} // namespace polatory
