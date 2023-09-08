#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_solver.hpp>

namespace polatory::interpolation {

rbf_fitter::rbf_fitter(const model& model, const geometry::points3d& points)
    : model_(model), points_(points) {}

common::valuesd rbf_fitter::fit(const common::valuesd& values, double absolute_tolerance,
                                int max_iter) const {
  rbf_solver solver(model_, points_);
  return solver.solve(values, absolute_tolerance, max_iter);
}

}  // namespace polatory::interpolation
