#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

template <class Model>
class rbf_field_function_25d : public field_function {
  static_assert(Model::kDim == 2, "Model must be two-dimensional.");

  using Interpolant = interpolant<Model>;

 public:
  explicit rbf_field_function_25d(Interpolant& interpolant) : interpolant_(interpolant) {}

  common::valuesd operator()(const geometry::points3d& points) const override {
    geometry::points2d points_2d(points.leftCols(2));

    return points.col(2) - interpolant_.evaluate_impl(points_2d);
  }

  void set_evaluation_bbox(const geometry::bbox3d& bbox) override {
    geometry::bbox2d bbox_2d{bbox.min().head<2>(), bbox.max().head<2>()};

    interpolant_.set_evaluation_bbox_impl(bbox_2d);
  }

 private:
  Interpolant& interpolant_;
};

}  // namespace polatory::isosurface
