#pragma once

#include <limits>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/structural/interpolant.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

class StructuralRbfFieldFunction : public FieldFunction {
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();

 public:
  explicit StructuralRbfFieldFunction(structural::StructuralInterpolant3& interpolant,
                                      double accuracy = kInfinity)
      : interpolant_(interpolant), accuracy_(accuracy) {}

  VecX operator()(const geometry::Points3& points) const override {
    return interpolant_.evaluate_impl(points);
  }

  void set_evaluation_bbox(const geometry::Bbox3& bbox) override {
    interpolant_.set_evaluation_bbox_impl(bbox, accuracy_);
  }

 private:
  structural::StructuralInterpolant3& interpolant_;
  double accuracy_;
};

}  // namespace polatory::isosurface
