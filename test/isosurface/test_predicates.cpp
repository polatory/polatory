#include <gtest/gtest.h>

#include <array>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/predicates.hpp>

using polatory::geometry::Point2;
using polatory::isosurface::incircle;
using polatory::isosurface::orient2d;
using polatory::isosurface::triangles_overlap_2d;

namespace {

std::array<Point2, 3> triangle(double ox, double oy) {
  return {Point2(ox, oy), Point2(ox + 1.0, oy), Point2(ox, oy + 1.0)};
}

}  // namespace

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

  EXPECT_GT(incircle(a, b, c, Point2(0.5, 0.5)), 0.0);  // the center, inside
  EXPECT_LT(incircle(a, b, c, Point2(2.0, 2.0)), 0.0);  // far outside
  EXPECT_NEAR(incircle(a, b, c, Point2(1.0, 1.0)), 0.0, 1e-15);  // on the circle

  // The sign flips with the winding.
  EXPECT_LT(incircle(a, c, b, Point2(0.5, 0.5)), 0.0);
}

TEST(predicates, triangles_overlap_2d_basic) {
  auto a = triangle(0.0, 0.0);
  constexpr double kSlack = 1e-9;

  EXPECT_TRUE(triangles_overlap_2d(a, triangle(0.2, 0.2), kSlack));  // interpenetrating
  EXPECT_FALSE(triangles_overlap_2d(a, triangle(5.0, 5.0), kSlack));  // disjoint

  // One triangle strictly inside the other.
  std::array<Point2, 3> inner{Point2(0.1, 0.1), Point2(0.4, 0.1), Point2(0.1, 0.4)};
  EXPECT_TRUE(triangles_overlap_2d(a, inner, kSlack));

  // Sharing only a vertex, otherwise apart: the touch reads as separated.
  std::array<Point2, 3> at_vertex{Point2(1.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 1.0)};
  EXPECT_FALSE(triangles_overlap_2d(a, at_vertex, kSlack));

  // Sharing an edge but on opposite sides: the two tile, they do not overlap.
  std::array<Point2, 3> across_edge{Point2(1.0, 0.0), Point2(0.0, 1.0), Point2(1.0, 1.0)};
  EXPECT_FALSE(triangles_overlap_2d(a, across_edge, kSlack));
}

TEST(predicates, triangles_overlap_2d_slack) {
  // triangle(d, d) overlaps triangle(0, 0) along the diagonal by a margin of (1 - 2d) * sqrt(1/2);
  // an overlap thinner than slack must read as separated, a thicker one as overlapping.
  auto a = triangle(0.0, 0.0);
  constexpr double kSlack = 1e-3;

  auto thin = triangle(0.5 - 1e-6, 0.5 - 1e-6);  // overlap ~1.4e-6 < slack
  EXPECT_FALSE(triangles_overlap_2d(a, thin, kSlack));

  auto thick = triangle(0.4, 0.4);  // overlap ~0.14 > slack
  EXPECT_TRUE(triangles_overlap_2d(a, thick, kSlack));
}
