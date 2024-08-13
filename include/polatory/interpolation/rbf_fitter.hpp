#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_solver.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <int Dim>
class Fitter {
  static constexpr int kDim = Dim;
  using Model = Model<kDim>;
  using Points = geometry::Points<kDim>;
  using Solver = Solver<kDim>;

 public:
  Fitter(const Model& model, const Points& points, const Points& grad_points)
      : model_(model), points_(points), grad_points_(grad_points) {}

  VecX fit(const VecX& values, double tolerance, double grad_tolerance, int max_iter,
           double accuracy, double grad_accuracy, const VecX* initial_weights = nullptr) const {
    Solver solver(model_, points_, grad_points_, accuracy, grad_accuracy);

    return solver.solve(values, tolerance, grad_tolerance, max_iter, initial_weights);
  }

 private:
  const Model& model_;
  const Points& points_;
  const Points& grad_points_;
};

}  // namespace polatory::interpolation
