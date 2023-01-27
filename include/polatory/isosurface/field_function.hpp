#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace isosurface {

class field_function {
 public:
  virtual ~field_function() = default;

  field_function(const field_function&) = delete;
  field_function(field_function&&) = delete;
  field_function& operator=(const field_function&) = delete;
  field_function& operator=(field_function&&) = delete;

  virtual common::valuesd operator()(const geometry::points3d& points) const = 0;

  virtual void set_evaluation_bbox(const geometry::bbox3d& /*bbox*/) {}

 protected:
  field_function() = default;
};

}  // namespace isosurface
}  // namespace polatory
