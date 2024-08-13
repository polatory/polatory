#pragma once

#include <limits>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

class RbfFieldFunction : public FieldFunction {
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();
  using Interpolant = Interpolant<3>;

 public:
  explicit RbfFieldFunction(Interpolant& interpolant, double accuracy = kInfinity,
                            double grad_accuracy = kInfinity)
      : interpolant_(interpolant), accuracy_(accuracy), grad_accuracy_(grad_accuracy) {}

  VecX operator()(const geometry::Points3& points) const override {
    return interpolant_.evaluate_impl(points);
  }

  void set_evaluation_bbox(const geometry::Bbox3& bbox) override {
    interpolant_.set_evaluation_bbox_impl(bbox, accuracy_, grad_accuracy_);
  }

 private:
  Interpolant& interpolant_;
  double accuracy_;
  double grad_accuracy_;
};

}  // namespace polatory::isosurface
