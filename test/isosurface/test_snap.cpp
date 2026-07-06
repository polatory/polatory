#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <boost/container_hash/hash.hpp>
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
