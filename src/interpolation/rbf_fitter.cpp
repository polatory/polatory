// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/interpolation/rbf_fitter.hpp>

#include <polatory/interpolation/rbf_solver.hpp>

namespace polatory {
namespace interpolation {

rbf_fitter::rbf_fitter(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
                       const geometry::points3d& points)
  : rbf_(rbf)
  , poly_dimension_(poly_dimension)
  , poly_degree_(poly_degree)
  , points_(points) {
}

common::valuesd rbf_fitter::fit(const common::valuesd& values, double absolute_tolerance) const {
  rbf_solver solver(rbf_, poly_dimension_, poly_degree_, points_);
  return solver.solve(values, absolute_tolerance);
}

}  // namespace interpolation
}  // namespace polatory
