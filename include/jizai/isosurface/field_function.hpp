#pragma once

#include <jizai/geometry/bbox3d.hpp>
#include <jizai/geometry/point3d.hpp>
#include <jizai/types.hpp>

namespace jizai::isosurface {

class FieldFunction {
 public:
  virtual ~FieldFunction() = default;

  FieldFunction(const FieldFunction&) = delete;
  FieldFunction(FieldFunction&&) = delete;
  FieldFunction& operator=(const FieldFunction&) = delete;
  FieldFunction& operator=(FieldFunction&&) = delete;

  virtual VecX operator()(const geometry::Points3& points) const = 0;

  virtual void set_evaluation_bbox(const geometry::Bbox3& /*bbox*/) {}

 protected:
  FieldFunction() = default;
};

}  // namespace jizai::isosurface
