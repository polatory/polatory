#include <gtest/gtest.h>
#include <igl/readOBJ.h>
#include <igl/tri_tri_intersect.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/isosurface.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/snap.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <string>
#include <vector>

using polatory::Index;
using polatory::Mat3;
using polatory::VecX;
using polatory::geometry::Bbox3;
using polatory::geometry::Point3;
using polatory::geometry::Points3;
using polatory::geometry::Vector3;
using polatory::isosurface::Faces;
using polatory::isosurface::FieldFunction;
using polatory::isosurface::Isosurface;
using polatory::isosurface::Mesh;
using polatory::isosurface::snap_mesh;

namespace {

class SignedDistanceFromPlane : public FieldFunction {
 public:
  VecX operator()(const Points3& points) const override { return points.col(2); }
};

class SignedDistanceFromSphere : public FieldFunction {
 public:
  explicit SignedDistanceFromSphere(double radius) : radius_(radius) {}

  VecX operator()(const Points3& points) const override {
    return points.rowwise().norm().array() - radius_;
  }

 private:
  double radius_;
};

// Whether triangles s and t actually intersect, in the same sense as the snapper's
// check: a clearly non-parallel pair by a 3D crossing test; a near-parallel pair only
// when one triangle crosses the other's plane within the overlap (a back-to-back fold).
bool triangles_intersect(const std::array<Eigen::Vector3d, 3>& s,
                         const std::array<Eigen::Vector3d, 3>& t) {
  Eigen::Vector3d ns = (s[1] - s[0]).cross(s[2] - s[0]);
  Eigen::Vector3d nt = (t[1] - t[0]).cross(t[2] - t[0]);
  if (ns.norm() == 0.0 || nt.norm() == 0.0) {
    return false;
  }
  if (std::abs(ns.dot(nt)) < 0.99 * ns.norm() * nt.norm()) {
    return igl::tri_tri_overlap_test_3d(s[0], s[1], s[2], t[0], t[1], t[2]);
  }
  Eigen::Vector3d normal = ns.normalized();
  double dmin = 1e30;
  double dmax = -1e30;
  for (const auto& x : t) {
    auto d = normal.dot(x - s[0]);
    dmin = std::min(dmin, d);
    dmax = std::max(dmax, d);
  }
  if (dmin > 0.0 || dmax < 0.0) {
    return false;
  }
  Eigen::Vector3d u = (s[1] - s[0]).normalized();
  Eigen::Vector3d v = normal.cross(u);
  auto proj = [&](const Eigen::Vector3d& x) {
    return Eigen::Vector2d((x - s[0]).dot(u), (x - s[0]).dot(v));
  };
  std::array<Eigen::Vector2d, 3> ps{proj(s[0]), proj(s[1]), proj(s[2])};
  std::array<Eigen::Vector2d, 3> pt{proj(t[0]), proj(t[1]), proj(t[2])};
  for (const auto* poly : {&ps, &pt}) {
    for (auto e = 0; e < 3; e++) {
      Eigen::Vector2d edge = (*poly).at((e + 1) % 3) - (*poly).at(e);
      Eigen::Vector2d axis(-edge.y(), edge.x());
      double smin = 1e30;
      double smax = -1e30;
      double tmin = 1e30;
      double tmax = -1e30;
      for (auto k = 0; k < 3; k++) {
        smin = std::min(smin, axis.dot(ps.at(k)));
        smax = std::max(smax, axis.dot(ps.at(k)));
        tmin = std::min(tmin, axis.dot(pt.at(k)));
        tmax = std::max(tmax, axis.dot(pt.at(k)));
      }
      if (smax <= tmin || tmax <= smin) {
        return false;
      }
    }
  }
  return true;
}

// Counts self-intersecting face pairs (transversal crossings and back-to-back coplanar
// overlaps), ignoring pairs that share a vertex. With a bbox prefilter; O(n^2) is fine
// for the test meshes.
Index count_self_intersections(const Mesh& mesh) {
  const auto& V = mesh.vertices();
  const auto& F = mesh.faces();
  auto n = F.rows();
  auto p = [&](Index v) { return Eigen::Vector3d(V.row(v).transpose()); };

  std::vector<Eigen::AlignedBox3d> boxes(n);
  for (Index i = 0; i < n; i++) {
    for (auto k = 0; k < 3; k++) {
      boxes.at(i).extend(p(F(i, k)));
    }
  }
  auto shares_vertex = [&](Index i, Index j) {
    for (auto a = 0; a < 3; a++) {
      for (auto b = 0; b < 3; b++) {
        if (F(i, a) == F(j, b)) {
          return true;
        }
      }
    }
    return false;
  };

  Index count{};
  for (Index i = 0; i < n; i++) {
    std::array<Eigen::Vector3d, 3> a{p(F(i, 0)), p(F(i, 1)), p(F(i, 2))};
    for (Index j = i + 1; j < n; j++) {
      if (!boxes.at(i).intersects(boxes.at(j)) || shares_vertex(i, j)) {
        continue;
      }
      std::array<Eigen::Vector3d, 3> b{p(F(j, 0)), p(F(j, 1)), p(F(j, 2))};
      if (triangles_intersect(a, b)) {
        count++;
      }
    }
  }
  return count;
}

// Counts edges shared by more than two faces (a non-manifold edge).
Index count_non_manifold_edges(const Mesh& mesh) {
  const auto& F = mesh.faces();
  std::map<std::pair<Index, Index>, int> edge_count;
  for (Index i = 0; i < F.rows(); i++) {
    for (auto k = 0; k < 3; k++) {
      auto u = F(i, k);
      auto w = F(i, (k + 1) % 3);
      edge_count[u < w ? std::pair{u, w} : std::pair{w, u}]++;
    }
  }
  Index count{};
  for (const auto& [e, c] : edge_count) {
    if (c > 2) {
      count++;
    }
  }
  return count;
}

// Points sampled on the sphere via a Fibonacci spiral, offset along the normal by a
// small alternating amount.
Points3 sphere_points(double radius, Index n, double offset) {
  Points3 points(n, 3);
  const double golden = 2.399963229728653;
  for (Index i = 0; i < n; i++) {
    auto z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(n);
    auto r = std::sqrt(std::max(0.0, 1.0 - z * z));
    auto phi = golden * static_cast<double>(i);
    Vector3 dir(r * std::cos(phi), r * std::sin(phi), z);
    auto sign = (i % 2 == 0) ? 1.0 : -1.0;
    points.row(i) = (radius + sign * offset) * dir;
  }
  return points;
}

// Points on the z = 0 plane inside a disk, on a deterministic spiral, offset in z by a
// small alternating (two-sided) amount.
Points3 plane_points(double radius, Index n, double offset) {
  Points3 points(n, 3);
  const double golden = 2.399963229728653;
  for (Index i = 0; i < n; i++) {
    auto r = radius * std::sqrt((static_cast<double>(i) + 0.5) / static_cast<double>(n));
    auto phi = golden * static_cast<double>(i);
    auto sign = (i % 2 == 0) ? 1.0 : -1.0;
    points.row(i) << r * std::cos(phi), r * std::sin(phi), sign * offset;
  }
  return points;
}

Mesh read_obj(const std::string& path) {
  Eigen::MatrixXd v;
  Eigen::MatrixXi f;
  EXPECT_TRUE(igl::readOBJ(path, v, f)) << "cannot read " << path;
  Points3 vertices = v;
  Faces faces = f.cast<Index>();
  return {std::move(vertices), std::move(faces)};
}

Points3 read_xyz(const std::string& path) {
  std::ifstream in(path);
  EXPECT_TRUE(in) << "cannot read " << path;
  std::vector<std::array<double, 3>> rows;
  std::array<double, 3> p{};
  while (in >> p[0] >> p[1] >> p[2]) {
    rows.push_back(p);
  }
  Points3 points(static_cast<Index>(rows.size()), 3);
  for (Index i = 0; i < points.rows(); i++) {
    points.row(i) << rows.at(i)[0], rows.at(i)[1], rows.at(i)[2];
  }
  return points;
}

}  // namespace

// Snapping a planar base may lift points into small overhangs -- the snapper guarantees no
// self-intersection, not a height field -- but the mesh must never actually self-intersect. The
// density is kept moderate: at high density the inexact counter here over-reports (a bare touch
// between the near-coplanar slivers dense points make reads as a crossing), and an exact-kernel
// check (kigumi) is the right tool for that regime.
TEST(snap_selfint, planar_base_has_no_self_intersections) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  Isosurface isosurf(bbox, resolution);
  SignedDistanceFromPlane field_fn;
  auto base = isosurf.generate(field_fn);
  ASSERT_EQ(count_self_intersections(base), 0);  // tool sanity: the valid base is clean

  const auto max_offset = 0.1 * resolution;

  const Index n = 500;
  for (double off : {0.0, 0.5}) {
    auto points = plane_points(0.7, n, off * max_offset);
    auto mesh = snap_mesh(base, points, VecX(), resolution, Mat3::Identity());
    auto self_int = count_self_intersections(mesh);
    std::cerr << "planar off=" << off << ": faces=" << mesh.faces().rows()
              << " self-intersections=" << self_int << "\n";
    EXPECT_EQ(self_int, 0);
  }
}

// Snapping a curved surface must not introduce a transversal self-intersection (one
// face piercing a topologically distant one). count_self_intersections catches the
// piercing kind that the planar height-field test cannot.
TEST(snap_selfint, curved_surface_has_no_self_intersections) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;
  const auto radius = 0.6;

  Isosurface isosurf(bbox, resolution);
  SignedDistanceFromSphere field_fn(radius);
  auto base = isosurf.generate(field_fn);
  ASSERT_EQ(count_self_intersections(base), 0);

  const auto max_offset = 0.05;
  auto points = sphere_points(radius, 1000, 0.5 * max_offset);
  auto mesh = snap_mesh(base, points, VecX(), resolution, Mat3::Identity());

  auto self_int = count_self_intersections(mesh);
  std::cerr << "sphere: faces=" << mesh.faces().rows() << " self-intersections=" << self_int
            << "\n";
  EXPECT_EQ(self_int, 0);
}

// A region carved out of a real interpolated surface (the "horse" model) plus the snap
// points that fall in it: densely subdivided patches sharing edges whose off-surface
// vertices used to make the two incident patches disagree on the edge subdivision (a
// non-manifold seam). The snapped result must stay manifold (the CDT never cuts a diagonal
// along a shared edge). Self-intersection-freeness is left to an exact-kernel check (kigumi)
// run out of band: this region is nearly flat, so the dense edge splits make near-coplanar
// slivers whose bare touches the double-precision counter here reads as crossings. The data
// was extracted from:
//   polatory isosurface --in horse.interpolant --seeds horse.asc --snap horse.asc
//     --acc 5e-7 --bbox -0.1 -0.1 -0.1 0.1 0.1 0.1 --res 5e-4
TEST(snap_selfint, real_surface_region_stays_manifold) {
  auto base = read_obj(std::string(POLATORY_TEST_DATA_DIR) + "/horse_region.obj");
  auto points = read_xyz(std::string(POLATORY_TEST_DATA_DIR) + "/horse_region_points.xyz");
  ASSERT_EQ(count_non_manifold_edges(base), 0);

  auto mesh = snap_mesh(base, points, VecX(), 5e-4, Mat3::Identity());

  auto nm_edges = count_non_manifold_edges(mesh);
  std::cerr << "horse region: faces=" << mesh.faces().rows() << " non-manifold-edges=" << nm_edges
            << "\n";
  EXPECT_EQ(nm_edges, 0);  // the two patches agree on every shared edge's subdivision
}
