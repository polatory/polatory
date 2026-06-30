#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/predicates.hpp>

using polatory::geometry::Point2;
using polatory::isosurface::incircle;
using polatory::isosurface::orient2d;

TEST(predicates, orient2d_sign_and_magnitude) {
  Point2 a(0.0, 0.0);
  Point2 b(1.0, 0.0);
  Point2 c(0.0, 1.0);

  // Counterclockwise is positive, clockwise negative, collinear zero. The magnitude is
  // twice the signed area (here area 1/2).
  EXPECT_DOUBLE_EQ(orient2d(a, b, c), 1.0);
  EXPECT_DOUBLE_EQ(orient2d(a, c, b), -1.0);
  EXPECT_DOUBLE_EQ(orient2d(a, b, Point2(2.0, 0.0)), 0.0);
}

TEST(predicates, incircle_inside_outside_on) {
  // Circumcircle of this right triangle is centered at (1/2, 1/2) with radius sqrt(1/2).
  Point2 a(0.0, 0.0);
  Point2 b(1.0, 0.0);
  Point2 c(0.0, 1.0);

  EXPECT_GT(incircle(a, b, c, Point2(0.5, 0.5)), 0.0);          // the center, inside
  EXPECT_LT(incircle(a, b, c, Point2(2.0, 2.0)), 0.0);          // far outside
  EXPECT_NEAR(incircle(a, b, c, Point2(1.0, 1.0)), 0.0, 1e-15);  // on the circle

  // The sign flips with the winding.
  EXPECT_LT(incircle(a, c, b, Point2(0.5, 0.5)), 0.0);
}
