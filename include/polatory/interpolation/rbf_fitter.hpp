#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_solver.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <int Dim>
class rbf_fitter {
  static constexpr int kDim = Dim;
  using Model = model<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Solver = rbf_solver<kDim>;

 public:
  rbf_fitter(const Model& model, const Points& points, const Points& grad_points)
      : model_(model), points_(points), grad_points_(grad_points) {}

  vectord fit(const vectord& values, double tolerance, double grad_tolerance, int max_iter,
              double accuracy, double grad_accuracy,
              const vectord* initial_weights = nullptr) const {
    Solver solver(model_, points_, grad_points_, accuracy, grad_accuracy);

    return solver.solve(values, tolerance, grad_tolerance, max_iter, initial_weights);
  }

 private:
  const Model& model_;
  const Points& points_;
  const Points& grad_points_;
};

}  // namespace polatory::interpolation
