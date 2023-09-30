#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_solver.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <class Model>
class rbf_fitter {
  static constexpr int kDim = Model::kDim;
  using Points = geometry::pointsNd<kDim>;
  using RbfSolver = rbf_solver<Model>;

 public:
  rbf_fitter(const Model& model, const Points& points)
      : rbf_fitter(model, points, Points(0, kDim)) {}

  rbf_fitter(const Model& model, const Points& points, const Points& grad_points)
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
