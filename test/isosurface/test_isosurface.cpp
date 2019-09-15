// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>

#include <gtest/gtest.h>

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/isosurface.hpp>
#include <polatory/types.hpp>

using polatory::common::valuesd;
using polatory::geometry::bbox3d;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::isosurface::field_function;
using polatory::isosurface::isosurface;

namespace {

class test_field_function : public field_function {
public:
  valuesd operator()(const points3d& points) const override {
    valuesd values(points.rows());

    for (auto i = 0; i < points.rows(); i++) {
      values(i) = std::sqrt(points.row(i).dot(points.row(i)));
    }

    return values;
  }
};

}  // namespace

TEST(isosurface, generate) {
  const bbox3d bbox(
    point3d(-1.0, -1.0, -1.0),
    point3d(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  isosurface isosurf(bbox, resolution);
  test_field_function field_fn;

  auto surface = isosurf.generate(field_fn, 1.0);

  // TODO(mizuno): Check topological properties instead.
  ASSERT_EQ(1466u, surface.vertices().size());
  ASSERT_EQ(2928u, surface.faces().size());
}

TEST(isosurface, generate_from_seed_points) {
  const bbox3d bbox(
    point3d(-1.0, -1.0, -1.0),
    point3d(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  points3d seed_points(1, 3);
  seed_points.row(0) = point3d(1.0, 0.0, 0.0);

  isosurface isosurf(bbox, resolution);
  test_field_function field_fn;

  auto surface = isosurf.generate_from_seed_points(seed_points, field_fn, 1.0);

  ASSERT_EQ(1466u, surface.vertices().size());
  ASSERT_EQ(2928u, surface.faces().size());
}
