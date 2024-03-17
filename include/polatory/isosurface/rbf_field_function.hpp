#pragma once

#include <polatory/common/types.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

class rbf_field_function : public field_function {
  using Interpolant = interpolant<3>;

 public:
  explicit rbf_field_function(Interpolant& interpolant) : interpolant_(interpolant) {}

  common::valuesd operator()(const geometry::points3d& points) const override {
    return interpolant_.evaluate_impl(points);
  }

  void set_evaluation_bbox(const geometry::bbox3d& bbox) override {
    interpolant_.set_evaluation_bbox_impl(bbox);
  }

 private:
  Interpolant& interpolant_;
};

}  // namespace polatory::isosurface
