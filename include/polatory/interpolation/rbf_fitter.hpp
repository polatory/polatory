#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_solver.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

class rbf_fitter {
 public:
  rbf_fitter(const model& model, const geometry::points3d& points);

  rbf_fitter(const model& model, const geometry::points3d& points,
             const geometry::points3d& grad_points);

  common::valuesd fit(const common::valuesd& values, double absolute_tolerance, int max_iter) const;

 private:
  rbf_solver solver_;
};

}  // namespace polatory::interpolation
