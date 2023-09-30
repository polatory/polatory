#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_solver.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <class Model>
class rbf_fitter {
  using RbfSolver = rbf_solver<Model>;

 public:
  rbf_fitter(const Model& model, const geometry::points3d& points)
      : rbf_fitter(model, points, geometry::points3d(0, 3)) {}

  rbf_fitter(const Model& model, const geometry::points3d& points,
             const geometry::points3d& grad_points)
      : solver_(model, points, grad_points) {}

  common::valuesd fit(const common::valuesd& values, double absolute_tolerance,
                      int max_iter) const {
    return fit(values, absolute_tolerance, absolute_tolerance, max_iter);
  }

  common::valuesd fit(const common::valuesd& values, double absolute_tolerance,
                      double grad_absolute_tolerance, int max_iter) const {
    return solver_.solve(values, absolute_tolerance, grad_absolute_tolerance, max_iter);
  }

 private:
  RbfSolver solver_;
};

}  // namespace polatory::interpolation
