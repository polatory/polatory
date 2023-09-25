#include <polatory/interpolation/rbf_fitter.hpp>

namespace polatory::interpolation {

rbf_fitter::rbf_fitter(const model& model, const geometry::points3d& points)
    : rbf_fitter(model, points, geometry::points3d(0, 3)) {}

rbf_fitter::rbf_fitter(const model& model, const geometry::points3d& points,
                       const geometry::points3d& grad_points)
    : solver_(model, points, grad_points) {}

common::valuesd rbf_fitter::fit(const common::valuesd& values, double absolute_tolerance,
                                int max_iter) const {
  return solver_.solve(values, absolute_tolerance, max_iter);
}

}  // namespace polatory::interpolation
