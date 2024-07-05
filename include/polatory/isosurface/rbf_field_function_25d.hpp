#pragma once

#include <limits>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

class rbf_field_function_25d : public field_function {
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();
  using Interpolant = interpolant<2>;

 public:
  explicit rbf_field_function_25d(Interpolant& interpolant, double accuracy = kInfinity,
                                  double grad_accuracy = kInfinity)
      : interpolant_(interpolant), accuracy_(accuracy), grad_accuracy_(grad_accuracy) {}

  vectord operator()(const geometry::points3d& points) const override {
    geometry::points2d points_2d(points.leftCols(2));

    return points.col(2) - interpolant_.evaluate_impl(points_2d);
  }

  void set_evaluation_bbox(const geometry::bbox3d& bbox) override {
    geometry::bbox2d bbox_2d{bbox.min().head<2>(), bbox.max().head<2>()};

    interpolant_.set_evaluation_bbox_impl(bbox_2d, accuracy_, grad_accuracy_);
  }

 private:
  Interpolant& interpolant_;
  double accuracy_;
  double grad_accuracy_;
};

}  // namespace polatory::isosurface
