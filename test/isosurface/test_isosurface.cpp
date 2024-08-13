#include <gtest/gtest.h>

#include <boost/container_hash/hash.hpp>
#include <numbers>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/isosurface.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/types.hpp>
#include <unordered_set>
#include <utility>

#include "../utility.hpp"

using polatory::Index;
using polatory::Mat3;
using polatory::VecX;
using polatory::geometry::Bbox3;
using polatory::geometry::Point3;
using polatory::geometry::Points3;
using polatory::geometry::Vector3;
using polatory::isosurface::FieldFunction;
using polatory::isosurface::Isosurface;
using polatory::isosurface::Mesh;
using polatory::isosurface::MeshDefectsFinder;

namespace {

class DistanceFromPoint : public FieldFunction {
 public:
  DistanceFromPoint() : point_(Point3::Zero()) {}

  explicit DistanceFromPoint(const Point3& point) : point_(point) {}

  VecX operator()(const Points3& points) const override {
    return (points.rowwise() - point_).rowwise().norm();
  }

 private:
  Point3 point_;
};

class RandomFieldFunction : public FieldFunction {
 public:
  VecX operator()(const Points3& points) const override {
    VecX values = VecX::Random(points.rows());
    // Randomly replace some values with 0.0.
    values = (VecX::Random(points.rows()).array().abs() < 0.1).select(0.0, values);
    return values;
  }
};

class SignedDistanceFromPlane : public FieldFunction {
 public:
  explicit SignedDistanceFromPlane(const Point3& point, const Vector3& direction)
      : normal_(direction.normalized()), d_(-normal_.dot(point)) {}

  VecX operator()(const Points3& points) const override {
    return (points * normal_.transpose()).array() + d_;
  }

 private:
  Vector3 normal_;
  double d_;
};

using Halfedge = std::pair<Index, Index>;

struct HalfedgeHash {
  std::size_t operator()(const Halfedge& e) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, e.first);
    boost::hash_combine(seed, e.second);
    return seed;
  }
};

bool test_boundary_coordinates(const Mesh& mesh, const Bbox3& bbox) {
  std::unordered_set<Halfedge, HalfedgeHash> boundary_hes;
  for (auto f : mesh.faces().rowwise()) {
    for (auto i = 0; i < 3; i++) {
      auto he = std::make_pair(f(i), f((i + 1) % 3));
      auto opp_he = std::make_pair(he.second, he.first);
      auto it = boundary_hes.find(opp_he);
      if (it == boundary_hes.end()) {
        boundary_hes.insert(he);
      } else {
        boundary_hes.erase(it);
      }
    }
  }

  std::unordered_set<Index> boundary_vertices;
  for (const auto& he : boundary_hes) {
    boundary_vertices.insert(he.first);
    boundary_vertices.insert(he.second);
  }

  const auto& min = bbox.min();
  const auto& max = bbox.max();
  for (auto vi : boundary_vertices) {
    auto p = mesh.vertices().row(vi);
    if (!(bbox.contains(p) && (p.array() == min.array() || p.array() == max.array()).any())) {
      return false;
    }
  }

  return true;
}

}  // namespace

TEST(isosurface, generate) {
  const Bbox3 bbox(Point3(-1.2, -1.2, -1.2), Point3(1.2, 1.2, 1.2));
  const auto resolution = 0.1;

  Isosurface isosurf(bbox, resolution);
  DistanceFromPoint field_fn;

  auto mesh = isosurf.generate(field_fn, 1.0);

  ASSERT_EQ(1082, mesh.vertices().rows());
  ASSERT_EQ(2160, mesh.faces().rows());
}

TEST(isosurface, generate_from_seed_points) {
  const Bbox3 bbox(Point3(-1.2, -1.2, -1.2), Point3(1.2, 1.2, 1.2));
  const auto resolution = 0.1;

  Isosurface isosurf(bbox, resolution);
  DistanceFromPoint field_fn;

  Points3 seed_points(1, 3);
  seed_points << Point3(1.0, 0.0, 0.0);

  auto mesh = isosurf.generate_from_seed_points(seed_points, field_fn, 1.0);

  ASSERT_EQ(1082, mesh.vertices().rows());
  ASSERT_EQ(2160, mesh.faces().rows());
}

TEST(isosurface, generate_empty) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  Isosurface isosurf(bbox, resolution);
  SignedDistanceFromPlane field_fn(Point3::Zero(), Vector3::Ones().normalized());

  auto mesh = isosurf.generate(field_fn, -1.01 * std::numbers::sqrt3);

  ASSERT_TRUE(mesh.is_empty());
  ASSERT_FALSE(mesh.is_entire());
}

TEST(isosurface, generate_empty_from_seed_points) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  Isosurface isosurf(bbox, resolution);
  SignedDistanceFromPlane field_fn(Point3::Zero(), Vector3::Ones().normalized());

  Points3 seed_points(1, 3);
  seed_points << Point3::Zero();

  auto mesh = isosurf.generate_from_seed_points(seed_points, field_fn, -1.01 * std::numbers::sqrt3);

  ASSERT_TRUE(mesh.is_empty());
  ASSERT_FALSE(mesh.is_entire());
}

TEST(isosurface, generate_entire) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  Isosurface isosurf(bbox, resolution);
  SignedDistanceFromPlane field_fn(Point3::Zero(), Vector3::Ones().normalized());

  auto mesh = isosurf.generate(field_fn, 1.01 * std::numbers::sqrt3);

  ASSERT_FALSE(mesh.is_empty());
  ASSERT_TRUE(mesh.is_entire());
}

TEST(isosurface, generate_entire_from_seed_points) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  Isosurface isosurf(bbox, resolution);
  SignedDistanceFromPlane field_fn(Point3::Zero(), Vector3::Ones().normalized());

  Points3 seed_points(1, 3);
  seed_points << Point3::Zero();

  auto mesh = isosurf.generate_from_seed_points(seed_points, field_fn, 1.01 * std::numbers::sqrt3);

  ASSERT_FALSE(mesh.is_empty());
  ASSERT_TRUE(mesh.is_entire());
}

TEST(isosurface, generate_from_seed_points_gradient_search) {
  const Bbox3 bbox(Point3(-1.2, -1.2, -1.2), Point3(1.2, 1.2, 1.2));
  const auto resolution = 0.1;

  for (auto i = 0; i < 100; i++) {
    const auto aniso = random_anisotropy<3>();

    Isosurface isosurf(bbox, resolution, aniso);
    SignedDistanceFromPlane field_fn(bbox.center(), Vector3::Random().normalized());

    Points3 seed_points(1, 3);
    seed_points << Point3::Zero();

    auto expected = isosurf.generate(field_fn, 1.0);
    isosurf.clear();
    auto actual = isosurf.generate_from_seed_points(seed_points, field_fn, 1.0);

    ASSERT_EQ(expected.faces().rows(), actual.faces().rows());
  }
}

TEST(isosurface, generate_plane) {
  const Bbox3 bbox(Point3(-1.2, -1.2, -1.2), Point3(1.2, 1.2, 1.2));
  const auto resolution = 0.1;

  Isosurface isosurf(bbox, resolution);
  SignedDistanceFromPlane field_fn(Point3::Zero(), Vector3::Ones().normalized());

  auto mesh = isosurf.generate(field_fn);

  ASSERT_EQ(820, mesh.vertices().rows());
  ASSERT_EQ(1421, mesh.faces().rows());
}

TEST(isosurface, manifold) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;
  const auto aniso = random_anisotropy<3>();

  Isosurface isosurf(bbox, resolution, aniso);
  RandomFieldFunction field_fn;

  auto mesh = isosurf.generate(field_fn, 0.0, 0);

  MeshDefectsFinder defects(mesh);

  const auto& min = bbox.min();
  const auto& max = bbox.max();
  for (auto vi : defects.singular_vertices()) {
    Point3 p = mesh.vertices().row(vi);
    ASSERT_TRUE((p.array() == min.array() || p.array() == max.array()).any());
  }

  ASSERT_TRUE(defects.intersecting_faces().empty());
}

TEST(isosurface, boundary_coordinates) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;
  const auto aniso = random_anisotropy<3>();

  Isosurface isosurf(bbox, resolution, aniso);
  RandomFieldFunction field_fn;

  auto mesh = isosurf.generate(field_fn, 0.0, 0);

  ASSERT_TRUE(test_boundary_coordinates(mesh, bbox));
}

TEST(isosurface, boundary_coordinates_seed_points) {
  const Bbox3 bbox(Point3(-1.0, -1.0, -1.0), Point3(1.0, 1.0, 1.0));
  const auto resolution = 0.1;
  const auto aniso = random_anisotropy<3>();

  Isosurface isosurf(bbox, resolution, aniso);
  RandomFieldFunction field_fn;

  Points3 seed_points(1, 3);
  seed_points << Point3::Zero();

  auto mesh = isosurf.generate_from_seed_points(seed_points, field_fn, 0.0, 0);

  ASSERT_TRUE(test_boundary_coordinates(mesh, bbox));
}
