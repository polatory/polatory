#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <set>
#include <utility>
#include <vector>

#include "../../src/isosurface/snapper/triangulation.hpp"

using polatory::Index;
using polatory::geometry::Point2;
using polatory::isosurface::Faces;
using polatory::isosurface::snapper::Triangulation;

namespace {

double tri_area(const Point2& a, const Point2& b, const Point2& c) {
  return 0.5 * ((b(0) - a(0)) * (c(1) - a(1)) - (b(1) - a(1)) * (c(0) - a(0)));
}

double polygon_area(const std::vector<Point2>& poly) {
  double a = 0.0;
  auto n = static_cast<Index>(poly.size());
  for (Index i = 0; i < n; i++) {
    const auto& p = poly.at(i);
    const auto& q = poly.at((i + 1) % n);
    a += p(0) * q(1) - q(0) * p(1);
  }
  return 0.5 * std::abs(a);
}

void check_triangulation(const std::vector<Point2>& boundary, const std::vector<Point2>& interior) {
  std::vector<Point2> p = boundary;
  p.insert(p.end(), interior.begin(), interior.end());

  Triangulation triangulation(boundary, interior);
  const auto& tris = triangulation.faces();
  ASSERT_GT(tris.rows(), 0);

  double sum = 0.0;
  std::set<std::pair<Index, Index>> edges;
  for (auto t : tris.rowwise()) {
    auto a = tri_area(p.at(t(0)), p.at(t(1)), p.at(t(2)));
    EXPECT_GT(a, 0.0) << "triangle is not CCW / has non-positive area";
    sum += a;
    for (auto k = 0; k < 3; k++) {
      auto u = t(k);
      auto w = t((k + 1) % 3);
      edges.insert(u < w ? std::pair{u, w} : std::pair{w, u});
    }
  }

  auto expected = polygon_area(boundary);
  EXPECT_NEAR(sum, expected, 1e-9 * expected) << "triangles do not tile the polygon exactly";

  auto nb = static_cast<Index>(boundary.size());
  for (Index i = 0; i < nb; i++) {
    auto u = i;
    auto w = (i + 1) % nb;
    auto key = u < w ? std::pair{u, w} : std::pair{w, u};
    EXPECT_TRUE(edges.contains(key)) << "boundary edge " << u << "-" << w << " is missing";
  }
}

Point2 pt(double x, double y) {
  Point2 p;
  p << x, y;
  return p;
}

constexpr Index kLastVertexOnEdge0 = 4;

std::vector<Point2> along_edge_boundary() {
  return {pt(0, 0), pt(0.25, -0.1), pt(0.5, -0.1), pt(0.75, -0.2), pt(1, 0), pt(0.5, 0.8)};
}

int count_along_edge_chords(const Faces& tris) {
  int chords = 0;
  for (auto t : tris.rowwise()) {
    for (auto k = 0; k < 3; k++) {
      auto u = t(k);
      auto w = t((k + 1) % 3);
      if (u <= kLastVertexOnEdge0 && w <= kLastVertexOnEdge0 && std::abs(u - w) > 1) {
        chords++;
      }
    }
  }
  return chords;
}

}  // namespace

TEST(cdt, square_no_interior) { check_triangulation({pt(0, 0), pt(1, 0), pt(1, 1), pt(0, 1)}, {}); }

TEST(cdt, square_clockwise_orientation_is_detected) {
  check_triangulation({pt(0, 0), pt(0, 1), pt(1, 1), pt(1, 0)}, {});
}

TEST(cdt, square_with_center_point) {
  check_triangulation({pt(0, 0), pt(1, 0), pt(1, 1), pt(0, 1)}, {pt(0.5, 0.5)});
}

TEST(cdt, square_with_many_interior_points) {
  std::vector<Point2> interior;
  for (auto i = 1; i < 8; i++) {
    for (auto j = 1; j < 8; j++) {
      interior.push_back(pt(i / 8.0, j / 8.0));
    }
  }
  check_triangulation({pt(0, 0), pt(1, 0), pt(1, 1), pt(0, 1)}, interior);
}

TEST(cdt, nonconvex_polygon) {
  check_triangulation({pt(0, 0), pt(2, 0), pt(2, 1), pt(1, 1), pt(1, 2), pt(0, 2)}, {});
}

TEST(cdt, nonconvex_polygon_with_interior) {
  check_triangulation({pt(0, 0), pt(2, 0), pt(2, 1), pt(1, 1), pt(1, 2), pt(0, 2)},
                      {pt(0.5, 0.5), pt(1.5, 0.5), pt(0.5, 1.5)});
}

TEST(cdt, subdivided_edges_like_a_snapper_patch) {
  std::vector<Point2> boundary{
      pt(0, 0), pt(0.33, 0.02), pt(0.66, -0.02), pt(1, 0), pt(0.52, 0.5), pt(0, 1), pt(-0.02, 0.5),
  };
  check_triangulation(boundary, {pt(0.3, 0.3), pt(0.2, 0.5)});
}

TEST(cdt, no_diagonal_runs_along_a_subdivided_edge) {
  std::vector<std::array<int, 2>> boundary_edges{
      {0, 2}, {0, -1}, {0, -1}, {0, -1}, {0, 1}, {1, 2},
  };

  Triangulation triangulation(along_edge_boundary(), {pt(0.3, -0.05)}, boundary_edges);
  const auto& tris = triangulation.faces();
  EXPECT_TRUE(triangulation.simple());
  ASSERT_GT(tris.rows(), 0);
  EXPECT_EQ(count_along_edge_chords(tris), 0) << "a diagonal runs along the subdivided edge";
}

TEST(cdt, along_edge_chord_appears_without_edge_labels) {
  Triangulation triangulation(along_edge_boundary(), {pt(0.3, -0.05)});
  EXPECT_GT(count_along_edge_chords(triangulation.faces()), 0);
}
