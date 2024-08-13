#pragma once

#include <limits>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

class RbfFieldFunction25D : public FieldFunction {
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();
  using Interpolant = Interpolant<2>;

 public:
  explicit RbfFieldFunction25D(Interpolant& interpolant, double accuracy = kInfinity,
                               double grad_accuracy = kInfinity)
      : interpolant_(interpolant), accuracy_(accuracy), grad_accuracy_(grad_accuracy) {}

  VecX operator()(const geometry::Points3& points) const override {
    geometry::Points2 points_2d(points.leftCols(2));

    return points.col(2) - interpolant_.evaluate_impl(points_2d);
  }

  void set_evaluation_bbox(const geometry::Bbox3& bbox) override {
    geometry::Bbox2 bbox_2d{bbox.min().head<2>(), bbox.max().head<2>()};

    interpolant_.set_evaluation_bbox_impl(bbox_2d, accuracy_, grad_accuracy_);
  }

 private:
  Interpolant& interpolant_;
  double accuracy_;
  double grad_accuracy_;
};

}  // namespace polatory::isosurface
