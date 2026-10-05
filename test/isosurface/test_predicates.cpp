#include <gtest/gtest.h>

#include <jizai/geometry/point3d.hpp>
#include <jizai/isosurface/predicates.hpp>

using jizai::geometry::Point2;
using jizai::isosurface::incircle;
using jizai::isosurface::orient2d;

TEST(predicates, orient2d_sign_and_magnitude) {
  Point2 a(0.0, 0.0);
  Point2 b(1.0, 0.0);
  Point2 c(0.0, 1.0);

  EXPECT_DOUBLE_EQ(orient2d(a, b, c), 1.0);
  EXPECT_DOUBLE_EQ(orient2d(a, c, b), -1.0);
  EXPECT_DOUBLE_EQ(orient2d(a, b, Point2(2.0, 0.0)), 0.0);
}

TEST(predicates, incircle_inside_outside_on) {
  Point2 a(0.0, 0.0);
  Point2 b(1.0, 0.0);
  Point2 c(0.0, 1.0);

  EXPECT_GT(incircle(a, b, c, Point2(0.5, 0.5)), 0.0);
  EXPECT_LT(incircle(a, b, c, Point2(2.0, 2.0)), 0.0);
  EXPECT_NEAR(incircle(a, b, c, Point2(1.0, 1.0)), 0.0, 1e-15);

  EXPECT_LT(incircle(a, c, b, Point2(0.5, 0.5)), 0.0);
}
