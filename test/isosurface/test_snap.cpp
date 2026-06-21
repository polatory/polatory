#include <gtest/gtest.h>

#include <Eigen/Core>
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
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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

// Collects the undirected boundary edges (incident to exactly one face).
std::unordered_set<Halfedge, HalfedgeHash> boundary_edges(const Mesh& mesh) {
  std::unordered_map<Halfedge, int, HalfedgeHash> undirected;
  for (auto f : mesh.faces().rowwise()) {
    for (auto i = 0; i < 3; i++) {
      auto u = f(i);
      auto w = f((i + 1) % 3);
      undirected[{std::min(u, w), std::max(u, w)}]++;
    }
  }
  std::unordered_set<Halfedge, HalfedgeHash> result;
  for (const auto& [e, n] : undirected) {
    if (n == 1) {
      result.insert(e);
    }
  }
  return result;
}

std::unordered_set<Index> boundary_vertices(const Mesh& mesh) {
  std::unordered_set<Index> result;
  for (const auto& e : boundary_edges(mesh)) {
    result.insert(e.first);
    result.insert(e.second);
  }
  return result;
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

// A bbox around the center cell of planar_grid(3), strictly inside the grid so that
// the grid boundary (the perimeter of [0, 3]^2) lies outside it.
const Bbox3 kCenterBbox(Point3(0.9, 0.9, -1.0), Point3(2.1, 2.1, 1.0));

// A generous absolute snapping distance: the grid cells have unit size and the test
// points sit at most 0.06 off the z = 0 plane.
constexpr double kMaxDistance = 0.5;

// Points exercising the three Voronoi regions of the center cell (whose diagonal
// runs from (1,1) to (2,2)), plus one near the grid corner whose projection is
// outside kCenterBbox. The first three project nearest to, respectively, the
// interior vertex (1,1), the diagonal midpoint, and a face centroid.
Points3 center_points() {
  Points3 points(4, 3);
  points << 1.1, 1.05, 0.03,  // snaps to vertex (1,1)
      1.5, 1.5, 0.04,         // snaps to the diagonal edge
      5.0 / 3.0, 4.0 / 3.0, -0.02,  // snaps to a face interior (centroid of (1,1),(2,1),(2,2))
      0.3, 0.3, 0.03;               // projection outside kCenterBbox
  return points;
}

}  // namespace

TEST(snap, points_inside_bbox_become_vertices) {
  auto points = center_points();
  auto mesh = snap_mesh(planar_grid(3), points, VecX(), kCenterBbox, kMaxDistance, Mat3::Identity());

  // The three points whose projections lie inside the bbox pass exactly through the
  // mesh.
  for (Index i = 0; i < 3; i++) {
    Point3 p = points.row(i);
    auto vi = find_vertex(mesh, p, 1e-12);
    ASSERT_TRUE(vi.has_value());
    ASSERT_EQ(Point3(mesh.vertices().row(*vi)), p);
  }

  // The point whose projection is outside the bbox is ignored.
  ASSERT_FALSE(find_vertex(mesh, Point3(points.row(3)), 1e-12).has_value());
}

TEST(snap, vertex_contention_cascades) {
  // Two points fall in the Voronoi cell of the same vertex (1,1). The one nearer the
  // mesh takes the vertex; the other cannot (a vertex cannot be split), so it
  // cascades to its next-nearest simplex (an edge) and is still snapped. Both pass
  // through the mesh exactly.
  Points3 points(2, 3);
  points << 1.06, 1.04, 0.03,  // nearer the mesh: takes vertex (1,1)
      1.10, 1.08, 0.06;        // farther: cascades to an edge
  auto mesh = snap_mesh(planar_grid(3), points, VecX(), kCenterBbox, kMaxDistance, Mat3::Identity());

  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(0)), 1e-12).has_value());
  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(1)), 1e-12).has_value());
  ASSERT_TRUE(is_oriented_manifold(mesh));
}

TEST(snap, manifold_and_disk_topology) {
  auto mesh = snap_mesh(planar_grid(3), center_points(), VecX(), kCenterBbox, kMaxDistance, Mat3::Identity());

  ASSERT_TRUE(is_oriented_manifold(mesh));
  ASSERT_EQ(1, euler_characteristic(mesh));
}

TEST(snap, leaves_boundary_untouched) {
  auto grid = planar_grid(3);

  auto mesh = snap_mesh(grid, center_points(), VecX(), kCenterBbox, kMaxDistance, Mat3::Identity());

  // The boundary is unchanged: snapping never reaches it, even for the point near
  // the grid corner, because its projection is outside the bbox. (Interior vertices
  // may move, e.g. (1,1) snaps to the first point, but boundary vertices do not.)
  ASSERT_EQ(boundary_edges(grid), boundary_edges(mesh));

  for (auto vi : boundary_vertices(grid)) {
    ASSERT_EQ(Point3(mesh.vertices().row(vi)), Point3(grid.vertices().row(vi)));
  }
}

TEST(snap, empty_points_is_noop) {
  auto grid = planar_grid(3);
  Points3 points(0, 3);
  auto mesh = snap_mesh(grid, points, VecX(), kCenterBbox, kMaxDistance, Mat3::Identity());

  ASSERT_EQ(grid.vertices().rows(), mesh.vertices().rows());
  ASSERT_EQ(grid.faces().rows(), mesh.faces().rows());
}

TEST(snap, rejects_points_beyond_max_distance) {
  // The diagonal point projects to the edge midpoint (1.5, 1.5) at distance 0.3; a
  // max distance of 0.2 rejects it, while 0.4 accepts it.
  Points3 points(1, 3);
  points << 1.5, 1.5, 0.3;

  auto rejected = snap_mesh(planar_grid(3), points, VecX(), kCenterBbox, 0.2, Mat3::Identity());
  ASSERT_FALSE(find_vertex(rejected, Point3(points.row(0)), 1e-12).has_value());

  auto accepted = snap_mesh(planar_grid(3), points, VecX(), kCenterBbox, 0.4, Mat3::Identity());
  ASSERT_TRUE(find_vertex(accepted, Point3(points.row(0)), 1e-12).has_value());
}

TEST(snap, rejects_invalid_max_distance_ratio) {
  Isosurface isosurf(Bbox3(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0)), 0.1);
  Points3 points(1, 3);
  points << 0.0, 0.0, 0.0;

  ASSERT_THROW(isosurf.set_snap_points(points, 0.0), std::invalid_argument);
  ASSERT_THROW(isosurf.set_snap_points(points, 1.5), std::invalid_argument);
  ASSERT_NO_THROW(isosurf.set_snap_points(points, 1.0));
  ASSERT_NO_THROW(isosurf.set_snap_points(points, 0.5));
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
  above_max << 0.6;  // exceeds max distance ratio 0.5
  VecX ok(1);
  ok << 0.1;

  ASSERT_THROW(isosurf.set_snap_points(points, 0.5, wrong_size), std::invalid_argument);
  ASSERT_THROW(isosurf.set_snap_points(points, 0.5, negative), std::invalid_argument);
  ASSERT_THROW(isosurf.set_snap_points(points, 0.5, above_max), std::invalid_argument);
  ASSERT_NO_THROW(isosurf.set_snap_points(points, 0.5, ok));
  ASSERT_NO_THROW(isosurf.set_snap_points(points, 0.5));  // empty tolerances disable the skip
}

TEST(snap, per_point_tolerance_skips_satisfied_point) {
  auto points = center_points();

  // The face-interior point (row 2) lies 0.02 off the mesh. A per-point tolerance of 0.1
  // above its distance marks it already satisfied, so it is skipped; the others, with
  // tolerance 0, still snap and become vertices.
  VecX tolerances(points.rows());
  tolerances << 0.0, 0.0, 0.1, 0.0;
  auto mesh = snap_mesh(planar_grid(3), points, tolerances, kCenterBbox, kMaxDistance, Mat3::Identity());

  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(0)), 1e-12).has_value());
  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(1)), 1e-12).has_value());
  ASSERT_FALSE(find_vertex(mesh, Point3(points.row(2)), 1e-12).has_value());
}

TEST(snap, isosurface_integration) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  Isosurface isosurf(bbox, resolution);
  SignedDistanceFromPlane field_fn(Point3::Zero(), Vector3::UnitZ());

  // Points near the z = 0 plane (within the triangle size), well inside the bbox,
  // plus one point outside the bbox that must be ignored.
  Points3 points(3, 3);
  points << 0.3, 0.2, 0.02,  //
      -0.5, 0.4, -0.03,      //
      2.0, 0.0, 0.0;         // outside the bbox
  isosurf.set_snap_points(points);

  auto mesh = isosurf.generate(field_fn);

  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(0)), 1e-10).has_value());
  ASSERT_TRUE(find_vertex(mesh, Point3(points.row(1)), 1e-10).has_value());
  ASSERT_FALSE(find_vertex(mesh, Point3(points.row(2)), 1e-10).has_value());

  ASSERT_TRUE(is_oriented_manifold(mesh));
}
