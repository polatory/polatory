#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <boost/container_hash/hash.hpp>
#include <cmath>
#include <cstddef>
#include <optional>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/isosurface.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/snap.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

using Halfedge = std::pair<Index, Index>;

struct HalfedgeHash {
  std::size_t operator()(const Halfedge& e) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, e.first);
    boost::hash_combine(seed, e.second);
    return seed;
  }
};

// Builds a regular grid of cells x cells unit squares in the z = 0 plane, each split
// into two triangles. Its boundary is the perimeter of [0, cells] x [0, cells].
Mesh planar_grid(int cells) {
  auto n = cells + 1;
  Points3 vertices(n * n, 3);
  for (auto j = 0; j < n; j++) {
    for (auto i = 0; i < n; i++) {
      vertices.row(j * n + i) << static_cast<double>(i), static_cast<double>(j), 0.0;
    }
  }
  Faces faces(2 * cells * cells, 3);
  Index f = 0;
  for (auto j = 0; j < cells; j++) {
    for (auto i = 0; i < cells; i++) {
      auto a = j * n + i;
      auto b = a + 1;
      auto c = a + n + 1;
      auto d = a + n;
      faces.row(f++) << a, b, c;
      faces.row(f++) << a, c, d;
    }
  }
  return {std::move(vertices), std::move(faces)};
}

std::optional<Index> find_vertex(const Mesh& mesh, const Point3& p, double tol) {
  for (Index i = 0; i < mesh.vertices().rows(); i++) {
    if ((mesh.vertices().row(i) - p).norm() <= tol) {
      return i;
    }
  }
  return std::nullopt;
}

// Checks that the mesh is a valid oriented manifold: every directed edge appears
// at most once, and every undirected edge is shared by at most two faces.
bool is_oriented_manifold(const Mesh& mesh) {
  std::unordered_map<Halfedge, int, HalfedgeHash> directed;
  std::unordered_map<Halfedge, int, HalfedgeHash> undirected;
  for (auto f : mesh.faces().rowwise()) {
    if (f(0) == f(1) || f(1) == f(2) || f(2) == f(0)) {
      return false;
    }
    for (auto i = 0; i < 3; i++) {
      auto u = f(i);
      auto w = f((i + 1) % 3);
      directed[{u, w}]++;
      undirected[{std::min(u, w), std::max(u, w)}]++;
    }
  }
  for (const auto& [e, n] : directed) {
    if (n > 1) {
      return false;
    }
  }
  for (const auto& [e, n] : undirected) {
    if (n > 2) {
      return false;
    }
  }
  return true;
}

// V - E + F.
Index euler_characteristic(const Mesh& mesh) {
  std::unordered_set<Halfedge, HalfedgeHash> edges;
  for (auto f : mesh.faces().rowwise()) {
    for (auto i = 0; i < 3; i++) {
      auto u = f(i);
      auto w = f((i + 1) % 3);
      edges.insert({std::min(u, w), std::max(u, w)});
    }
  }
  return mesh.vertices().rows() - static_cast<Index>(edges.size()) + mesh.faces().rows();
}

class SignedDistanceFromPlane : public FieldFunction {
 public:
  SignedDistanceFromPlane(const Point3& point, const Vector3& direction)
      : normal_(direction.normalized()), d_(-normal_.dot(point)) {}

  VecX operator()(const Points3& points) const override {
    return (points * normal_.transpose()).array() + d_;
  }

 private:
  Vector3 normal_;
  double d_;
};

// Exact signed distance to a capped cylinder of the given radius and half-height, axis along z.
class SignedDistanceFromCappedCylinder : public FieldFunction {
 public:
  SignedDistanceFromCappedCylinder(double radius, double half_height)
      : radius_(radius), half_height_(half_height) {}

  VecX operator()(const Points3& points) const override {
    VecX d(points.rows());
    for (Index i = 0; i < points.rows(); i++) {
      double dr = std::hypot(points(i, 0), points(i, 1)) - radius_;
      double dz = std::abs(points(i, 2)) - half_height_;
      double outside = std::hypot(std::max(dr, 0.0), std::max(dz, 0.0));
      double inside = std::min(std::max(dr, dz), 0.0);
      d(i) = inside + outside;
    }
    return d;
  }

 private:
  double radius_;
  double half_height_;
};

// Approximate signed distance to a solid acute cone: apex at (0, 0, height), circular base of the
// given radius at z = 0. The intersection of the lateral cone and the base half-space; the lateral
// implicit is divided by its gradient magnitude to read as a distance near the surface.
class SignedDistanceFromCone : public FieldFunction {
 public:
  SignedDistanceFromCone(double radius, double height) : radius_(radius), height_(height) {}

  VecX operator()(const Points3& points) const override {
    double slope = std::hypot(1.0, radius_ / height_);
    VecX d(points.rows());
    for (Index i = 0; i < points.rows(); i++) {
      double rho = std::hypot(points(i, 0), points(i, 1));
      double lateral = (rho - radius_ * (height_ - points(i, 2)) / height_) / slope;
      double base = -points(i, 2);
      d(i) = std::max(lateral, base);
    }
    return d;
  }

 private:
  double radius_;
  double height_;
};

// Exact signed distance to a sphere of the given radius centered at the origin.
class SignedDistanceFromSphere : public FieldFunction {
 public:
  explicit SignedDistanceFromSphere(double radius) : radius_(radius) {}

  VecX operator()(const Points3& points) const override {
    return points.rowwise().norm().array() - radius_;
  }

 private:
  double radius_;
};

// Exact signed distance to an axis-aligned cube centered at the origin with the given half-extent.
class SignedDistanceFromCube : public FieldFunction {
 public:
  explicit SignedDistanceFromCube(double half_extent) : half_(half_extent) {}

  VecX operator()(const Points3& points) const override {
    VecX d(points.rows());
    for (Index i = 0; i < points.rows(); i++) {
      double qx = std::abs(points(i, 0)) - half_;
      double qy = std::abs(points(i, 1)) - half_;
      double qz = std::abs(points(i, 2)) - half_;
      double outside = std::hypot(std::max(qx, 0.0), std::max(qy, 0.0), std::max(qz, 0.0));
      double inside = std::min(std::max({qx, qy, qz}), 0.0);
      d(i) = inside + outside;
    }
    return d;
  }

 private:
  double half_;
};

// Points exercising the three Voronoi regions of the center cell (whose diagonal runs from (1,1) to
// (2,2)): they project nearest to, respectively, the interior vertex (1,1), the diagonal midpoint,
// and a face centroid.
Points3 center_points() {
  Points3 points(3, 3);
  points << 1.1, 1.05, 0.03,        // snaps to vertex (1,1)
      1.5, 1.5, 0.04,               // snaps to the diagonal edge
      5.0 / 3.0, 4.0 / 3.0, -0.02;  // snaps to a face interior (centroid of (1,1),(2,1),(2,2))
  return points;
}

}  // namespace

TEST(snap, points_become_vertices) {
  auto points = center_points();
  auto mesh = snap_mesh(planar_grid(3), points, VecX(), 1.0, Mat3::Identity());

  // Each point passes exactly through the mesh.
  for (Index i = 0; i < points.rows(); i++) {
    Point3 p = points.row(i);
    auto vi = find_vertex(mesh, p, 1e-12);
    ASSERT_TRUE(vi.has_value());
    ASSERT_EQ(Point3(mesh.vertices().row(*vi)), p);
  }
}

TEST(snap, vertex_contention_honors_one) {
  // Two points fall in the Voronoi cell of the same vertex (1,1). Only one can take it (a
  // vertex cannot be split); the snapper prefers a clean vertex move over a sliver-pinning
  // insert, so the loser is left for a later pass rather than cascaded in this one. The
  // winner passes through the mesh exactly and the result stays a manifold.
  Points3 points(2, 3);
  points << 1.06, 1.04, 0.03,  //
      1.10, 1.08, 0.06;
  auto mesh = snap_mesh(planar_grid(3), points, VecX(), 1.0, Mat3::Identity());

  Index honored = 0;
  for (Index i = 0; i < points.rows(); i++) {
    if (find_vertex(mesh, Point3(points.row(i)), 1e-12).has_value()) {
      honored++;
    }
  }
  ASSERT_EQ(honored, 1);
  ASSERT_TRUE(is_oriented_manifold(mesh));
}

TEST(snap, manifold_and_disk_topology) {
  auto mesh = snap_mesh(planar_grid(3), center_points(), VecX(), 1.0, Mat3::Identity());

  ASSERT_TRUE(is_oriented_manifold(mesh));
  ASSERT_EQ(1, euler_characteristic(mesh));
}

TEST(snap, empty_points_is_noop) {
  auto grid = planar_grid(3);
  Points3 points(0, 3);
  auto mesh = snap_mesh(grid, points, VecX(), 1.0, Mat3::Identity());

  ASSERT_EQ(grid.vertices().rows(), mesh.vertices().rows());
  ASSERT_EQ(grid.faces().rows(), mesh.faces().rows());
}

TEST(snap, rejects_points_beyond_the_resolution) {
  // Both points project to the edge midpoint (1.5, 1.5); with the resolution 1.0, the one
  // 0.3 away is within the snapping distance and accepted, the one 1.5 away is rejected.
  Points3 near_point(1, 3);
  near_point << 1.5, 1.5, 0.3;
  Points3 far_point(1, 3);
  far_point << 1.5, 1.5, 1.5;

  auto accepted = snap_mesh(planar_grid(3), near_point, VecX(), 1.0, Mat3::Identity());
  ASSERT_TRUE(find_vertex(accepted, Point3(near_point.row(0)), 1e-12).has_value());

  auto rejected = snap_mesh(planar_grid(3), far_point, VecX(), 1.0, Mat3::Identity());
  ASSERT_FALSE(find_vertex(rejected, Point3(far_point.row(0)), 1e-12).has_value());
}

TEST(snap, rejects_invalid_tolerance_ratios) {
  Isosurface isosurf(Bbox3(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0)), 0.1);
  Points3 points(1, 3);
  points << 0.0, 0.0, 0.0;

  VecX wrong_size(2);
  wrong_size << 0.1, 0.2;
  VecX negative(1);
  negative << -0.1;
  VecX above_max(1);
  above_max << 1.5;  // exceeds the maximum ratio 1.0
  VecX ok(1);
  ok << 0.1;

  ASSERT_THROW(isosurf.set_snap_points(points, wrong_size), std::invalid_argument);
  ASSERT_THROW(isosurf.set_snap_points(points, negative), std::invalid_argument);
  ASSERT_THROW(isosurf.set_snap_points(points, above_max), std::invalid_argument);
  ASSERT_NO_THROW(isosurf.set_snap_points(points, ok));
  ASSERT_NO_THROW(isosurf.set_snap_points(points));  // empty tolerances disable the skip
}

TEST(snap, per_point_tolerance_skips_satisfied_point) {
  auto points = center_points();

  // The face-interior point (row 2) lies 0.02 off the mesh. A per-point tolerance of 0.1
  // above its distance marks it already satisfied, so it is skipped; the others, with
  // tolerance 0, still snap and become vertices.
  VecX tolerances(points.rows());
  tolerances << 0.0, 0.0, 0.1;
  auto mesh = snap_mesh(planar_grid(3), points, tolerances, 1.0, Mat3::Identity());

  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(0)), 1e-12).has_value());
  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(1)), 1e-12).has_value());
  ASSERT_FALSE(find_vertex(mesh, Point3(points.row(2)), 1e-12).has_value());
}

TEST(snap, vertex_within_tolerance_ball_is_moved_not_inserted) {
  auto base = planar_grid(3);
  auto n_vertices = base.vertices().rows();

  // The point lies ~0.06 from vertex (1, 1), within its 0.1 tolerance ball. The snapper moves that
  // vertex exactly onto the point rather than inserting a new one, so the point becomes a vertex,
  // vertex (1, 1) leaves its place, and the vertex count is unchanged.
  Points3 points(1, 3);
  points << 1.05, 1.03, 0.02;
  VecX tolerances(1);
  tolerances << 0.1;
  auto mesh = snap_mesh(base, points, tolerances, 1.0, Mat3::Identity());

  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(0)), 1e-12).has_value());
  ASSERT_FALSE(find_vertex(mesh, Point3(1.0, 1.0, 0.0), 1e-12).has_value());
  ASSERT_EQ(mesh.vertices().rows(), n_vertices);
}

TEST(snap, isosurface_integration) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  Isosurface isosurf(bbox, resolution);
  SignedDistanceFromPlane field_fn(Point3::Zero(), Vector3::UnitZ());

  // Points near the z = 0 plane (within the triangle size), plus one far from the mesh that
  // max_distance must reject.
  Points3 points(3, 3);
  points << 0.3, 0.2, 0.02,  //
      -0.5, 0.4, -0.03,      //
      2.0, 0.0, 0.0;         // far from the surface
  isosurf.set_snap_points(points);

  auto mesh = isosurf.generate(field_fn);

  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(0)), 1e-10).has_value());
  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(1)), 1e-10).has_value());
  ASSERT_FALSE(find_vertex(mesh, Point3(points.row(2)), 1e-10).has_value());

  ASSERT_TRUE(is_oriented_manifold(mesh));
}

// TEMPORARY (no assertions): sweeps capped cylinders of varying radius/half-height/resolution,
// snapping each to its two rim circles plus four vertical laterals (at 45/135/225/315, crossing
// both rims). Writes /tmp/cyl_<name>.obj per config.
TEST(snap, TEMP_cylinder_feature_lines) {
  struct Config {
    const char* name;
    double radius, half_height, resolution;
  };
  const Config configs[] = {
      {"R1_H1_res0.2", 1.0, 1.0, 0.2},     {"R1_H1_res0.1", 1.0, 1.0, 0.1},
      {"R1_H2_res0.2", 1.0, 2.0, 0.2},     {"R2_H1_res0.2", 2.0, 1.0, 0.2},
      {"R0.5_H1.5_res0.1", 0.5, 1.5, 0.1},
  };
  const double two_pi = 2.0 * std::acos(-1.0);
  for (const auto& cfg : configs) {
    const double radius = cfg.radius, half_height = cfg.half_height, resolution = cfg.resolution;
    const double pad = 3.0 * resolution;
    const Bbox3 bbox(Point3(-radius - pad, -radius - pad, -half_height - pad),
                     Point3(radius + pad, radius + pad, half_height + pad));
    SignedDistanceFromCappedCylinder field_fn(radius, half_height);

    // Rims + laterals sampled at ~1/4 res; n a multiple of 8 so a rim sample lands on each lateral.
    const Index n =
        8 * std::max<Index>(
                1, static_cast<Index>(std::llround(two_pi * radius / (0.25 * resolution) / 8.0)));
    const Index n_lat = 4;
    const Index m = static_cast<Index>(2.0 * half_height / (0.25 * resolution));
    Points3 snap(2 * n + n_lat * (m - 1), 3);
    Index row = 0;
    for (Index i = 0; i < n; i++) {
      double a = two_pi * static_cast<double>(i) / static_cast<double>(n);
      snap.row(row++) << radius * std::cos(a), radius * std::sin(a), half_height;
      snap.row(row++) << radius * std::cos(a), radius * std::sin(a), -half_height;
    }
    for (Index g = 0; g < n_lat; g++) {
      double a = two_pi * (static_cast<double>(g) + 0.5) / static_cast<double>(n_lat);
      double cx = radius * std::cos(a), cy = radius * std::sin(a);
      for (Index j = 1; j < m; j++) {
        double z =
            -half_height + static_cast<double>(j) * (2.0 * half_height / static_cast<double>(m));
        snap.row(row++) << cx, cy, z;
      }
    }
    Isosurface snap_iso(bbox, resolution);
    snap_iso.set_snap_points(snap, VecX::Constant(snap.rows(), 0.05));
    snap_iso.generate(field_fn, 0.0).export_obj(std::string("/tmp/cyl_") + cfg.name + ".obj");
  }
}

// TEMPORARY (no assertions): sweeps solid cones of varying radius/height/resolution (acute to
// obtuse), snapping each to its base rim, four apex-to-rim generators, and those generators
// continued across the base cap to its centre -- where the four lines cross. Writes
// /tmp/cone_<name>.obj per config.
TEST(snap, TEMP_cone_feature_lines) {
  struct Config {
    const char* name;
    double radius, height, resolution;
  };
  const Config configs[] = {
      {"R1_H4_res0.2", 1.0, 4.0, 0.2},     {"R1_H2_res0.2", 1.0, 2.0, 0.2},
      {"R1_H1_res0.2", 1.0, 1.0, 0.2},     {"R1_H1_res0.1", 1.0, 1.0, 0.1},
      {"R1_H0.5_res0.2", 1.0, 0.5, 0.2},    // obtuse: rim sits beyond max_distance = res
      {"R1_H0.25_res0.2", 1.0, 0.25, 0.2},  // harsher: rim ~2 res beyond max_distance
  };
  const double two_pi = 2.0 * std::acos(-1.0);
  for (const auto& cfg : configs) {
    const double radius = cfg.radius, height = cfg.height, resolution = cfg.resolution;
    const double pad = 3.0 * resolution;
    const Bbox3 bbox(Point3(-radius - pad, -radius - pad, -pad),
                     Point3(radius + pad, radius + pad, height + pad));
    SignedDistanceFromCone field_fn(radius, height);
    const Point3 apex(0.0, 0.0, height);

    const Index n_rim = static_cast<Index>(two_pi * radius / (0.25 * resolution));
    const Index n_gen = 8;
    const double slant = std::hypot(radius, height);
    const Index m = static_cast<Index>(slant / (0.25 * resolution));
    const Index m_cap = static_cast<Index>(radius / (0.25 * resolution));
    Points3 snap(n_rim + 1 + n_gen * (m - 1) + n_gen * (m_cap - 1) + 1, 3);
    Index row = 0;
    for (Index i = 0; i < n_rim; i++) {
      double a = two_pi * static_cast<double>(i) / static_cast<double>(n_rim);
      snap.row(row++) << radius * std::cos(a), radius * std::sin(a), 0.0;
    }
    snap.row(row++) = apex;
    for (Index g = 0; g < n_gen; g++) {
      double a = two_pi * (static_cast<double>(g) + 0.5) / static_cast<double>(n_gen);
      Point3 rim(radius * std::cos(a), radius * std::sin(a), 0.0);
      for (Index j = 1; j < m; j++) {
        double t = static_cast<double>(j) / static_cast<double>(m);
        snap.row(row++) = apex + t * (rim - apex);
      }
      // Continue the generator across the base cap from the rim toward the centre; the opposite
      // generator does the same, so the four lines cross at the centre (added once below).
      for (Index j = 1; j < m_cap; j++) {
        double t = static_cast<double>(j) / static_cast<double>(m_cap);
        snap.row(row++) = (1.0 - t) * rim;
      }
    }
    snap.row(row++) = Point3(0.0, 0.0, 0.0);
    Isosurface snap_iso(bbox, resolution);
    snap_iso.set_snap_points(snap, VecX::Constant(snap.rows(), 0.05));
    snap_iso.generate(field_fn, 0.0).export_obj(std::string("/tmp/cone_") + cfg.name + ".obj");
  }
}

// TEMPORARY (no assertions): sweeps axis-aligned cubes of varying size/resolution, snapping each to
// its 8 corners plus 12 edges (each sampled at ~1/4 res). Writes /tmp/cube_<name>.obj per config.
TEST(snap, TEMP_cube_feature_lines) {
  struct Config {
    const char* name;
    double half, resolution;
  };
  const Config configs[] = {
      {"S1_res0.2", 0.5, 0.2},
      {"S1_res0.1", 0.5, 0.1},
      {"S2_res0.2", 1.0, 0.2},
      {"S2_res0.1", 1.0, 0.1},
  };
  for (const auto& cfg : configs) {
    const double half = cfg.half, resolution = cfg.resolution;
    const double pad = 3.0 * resolution;
    const Bbox3 bbox(Point3(-half - pad, -half - pad, -half - pad),
                     Point3(half + pad, half + pad, half + pad));
    SignedDistanceFromCube field_fn(half);

    const Index m = static_cast<Index>(2.0 * half / (0.25 * resolution));
    Points3 snap(8 + 12 * (m - 1), 3);
    Index row = 0;
    for (int sx = -1; sx <= 1; sx += 2) {
      for (int sy = -1; sy <= 1; sy += 2) {
        for (int sz = -1; sz <= 1; sz += 2) {
          snap.row(row++) << sx * half, sy * half, sz * half;
        }
      }
    }
    for (int axis = 0; axis < 3; axis++) {
      int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
      for (int s1 = -1; s1 <= 1; s1 += 2) {
        for (int s2 = -1; s2 <= 1; s2 += 2) {
          for (Index j = 1; j < m; j++) {
            double t = -half + static_cast<double>(j) * (2.0 * half / static_cast<double>(m));
            Point3 p;
            p(axis) = t;
            p(a1) = s1 * half;
            p(a2) = s2 * half;
            snap.row(row++) = p;
          }
        }
      }
    }
    Isosurface snap_iso(bbox, resolution);
    snap_iso.set_snap_points(snap, VecX::Constant(snap.rows(), 0.05));
    snap_iso.generate(field_fn, 0.0).export_obj(std::string("/tmp/cube_") + cfg.name + ".obj");
  }
}

TEST(smooth, flat_plane_random_orientation_no_flips) {
  using polatory::isosurface::smooth_snapped_mesh;

  const int n = 25;
  const double h = 20.0;  // grid spacing == resolution
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::uniform_real_distribution<double> jit(-0.3 * h, 0.3 * h);

  // n x n grid on the z = 0 plane, interior vertices jittered IN-PLANE so quads are generic but the
  // surface stays exactly flat (every flip is bend-neutral; only FP noise could trigger one).
  Points3 v(n * n, 3);
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      double x = i * h;
      double y = j * h;
      if (i > 0 && i < n - 1) x += jit(rng);
      if (j > 0 && j < n - 1) y += jit(rng);
      v.row(j * n + i) = Point3(x, y, 0.0);
    }
  }
  auto id = [&](int i, int j) { return static_cast<Index>(j * n + i); };
  std::vector<std::array<Index, 3>> tris;
  for (int j = 0; j < n - 1; j++) {
    for (int i = 0; i < n - 1; i++) {
      tris.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1)});
      tris.push_back({id(i, j), id(i + 1, j + 1), id(i, j + 1)});
    }
  }
  Faces f(static_cast<Index>(tris.size()), 3);
  for (std::size_t k = 0; k < tris.size(); k++) {
    f.row(static_cast<Index>(k)) << tris[k][0], tris[k][1], tris[k][2];
  }

  auto edges = [](const Faces& faces) {
    std::set<std::pair<Index, Index>> s;
    for (Index r = 0; r < faces.rows(); r++) {
      for (int k = 0; k < 3; k++) {
        Index a = faces(r, k);
        Index b = faces(r, (k + 1) % 3);
        s.insert({std::min(a, b), std::max(a, b)});
      }
    }
    return s;
  };
  auto before = edges(f);

  // Many generic orientations: a flat surface gives every flip zero bend change, so any flip would
  // be pure FP noise crossing the 1e-6 threshold.
  for (int trial = 0; trial < 50; trial++) {
    Eigen::Vector3d axis(u(rng), u(rng), u(rng));
    axis.normalize();
    Mat3 r = Eigen::AngleAxisd(3.141592653589793 * u(rng), axis).toRotationMatrix();
    Points3 vr = v * r.transpose();

    auto out = smooth_snapped_mesh(Mesh(vr, f), Points3(0, 3), VecX(), h, Mat3::Identity());
    auto after = edges(out.faces());
    int flipped = 0;
    for (const auto& e : before) {
      if (!after.contains(e)) flipped++;
    }
    EXPECT_EQ(flipped, 0) << "trial " << trial << ": " << flipped
                          << " edges flipped on a flat plane";
  }
}
