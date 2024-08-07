#include <gtest/gtest.h>

#include <boost/container_hash/hash.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/isosurface.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/types.hpp>
#include <unordered_set>
#include <utility>

#include "../utility.hpp"

using polatory::vectord;
using polatory::geometry::bbox3d;
using polatory::geometry::matrix3d;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::geometry::vector3d;
using polatory::isosurface::field_function;
using polatory::isosurface::isosurface;
using polatory::isosurface::mesh_defects_finder;
using polatory::isosurface::surface;
using polatory::isosurface::vertex_index;

namespace {

class constant_field_function : public field_function {
 public:
  explicit constant_field_function(double value) : value_(value) {}

  vectord operator()(const points3d& points) const override {
    return vectord::Constant(points.rows(), value_);
  }

 private:
  double value_;
};

class distance_from_point : public field_function {
 public:
  distance_from_point() : point_(point3d::Zero()) {}

  explicit distance_from_point(const point3d& point) : point_(point) {}

  vectord operator()(const points3d& points) const override {
    return (points.rowwise() - point_).rowwise().norm();
  }

 private:
  point3d point_;
};

class random_field_function : public field_function {
 public:
  vectord operator()(const points3d& points) const override {
    return vectord::Random(points.rows());
  }
};

class signed_distance_from_plane : public field_function {
 public:
  explicit signed_distance_from_plane(const point3d& point, const vector3d& direction)
      : normal_(direction.normalized()), d_(-normal_.dot(point)) {}

  vectord operator()(const points3d& points) const override {
    return (points * normal_.transpose()).array() + d_;
  }

 private:
  vector3d normal_;
  double d_;
};

using halfedge = std::pair<vertex_index, vertex_index>;

struct halfedge_hash {
  std::size_t operator()(const halfedge& e) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, e.first);
    boost::hash_combine(seed, e.second);
    return seed;
  }
};

bool test_boundary_coordinates(const surface& surface, const bbox3d& bbox) {
  std::unordered_set<halfedge, halfedge_hash> boundary_hes;
  for (auto f : surface.faces().rowwise()) {
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

  std::unordered_set<vertex_index> boundary_vertices;
  for (const auto& he : boundary_hes) {
    boundary_vertices.insert(he.first);
    boundary_vertices.insert(he.second);
  }

  const auto& min = bbox.min();
  const auto& max = bbox.max();
  for (auto vi : boundary_vertices) {
    auto p = surface.vertices().row(vi);
    if (!(bbox.contains(p) && (p.array() == min.array() || p.array() == max.array()).any())) {
      return false;
    }
  }

  return true;
}

}  // namespace

TEST(isosurface, generate) {
  const bbox3d bbox(point3d(-1.2, -1.2, -1.2), point3d(1.2, 1.2, 1.2));
  const auto resolution = 0.1;

  isosurface isosurf(bbox, resolution);
  distance_from_point field_fn;

  auto surface = isosurf.generate(field_fn, 1.0);

  ASSERT_EQ(1082, surface.vertices().rows());
  ASSERT_EQ(2160, surface.faces().rows());
}

TEST(isosurface, generate_empty) {
  const bbox3d bbox(point3d(-1.0, -1.0, -1.0), point3d(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  isosurface isosurf(bbox, resolution);
  constant_field_function field_fn(1.0);

  auto surface = isosurf.generate(field_fn);

  ASSERT_TRUE(surface.is_empty());
  ASSERT_FALSE(surface.is_entire());
}

TEST(isosurface, generate_entire) {
  const bbox3d bbox(point3d(-1.0, -1.0, -1.0), point3d(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  isosurface isosurf(bbox, resolution);
  constant_field_function field_fn(-1.0);

  auto surface = isosurf.generate(field_fn);

  ASSERT_FALSE(surface.is_empty());
  ASSERT_TRUE(surface.is_entire());
}

TEST(isosurface, generate_from_seed_points) {
  const bbox3d bbox(point3d(-1.2, -1.2, -1.2), point3d(1.2, 1.2, 1.2));
  const auto resolution = 0.1;

  isosurface isosurf(bbox, resolution);
  distance_from_point field_fn;

  points3d seed_points(1, 3);
  seed_points << point3d(1.0, 0.0, 0.0);

  auto surface = isosurf.generate_from_seed_points(seed_points, field_fn, 1.0);

  ASSERT_EQ(1082, surface.vertices().rows());
  ASSERT_EQ(2160, surface.faces().rows());
}

TEST(isosurface, generate_from_seed_points_gradient_search) {
  const bbox3d bbox(point3d(-1.2, -1.2, -1.2), point3d(1.2, 1.2, 1.2));
  const auto resolution = 0.1;

  for (auto i = 0; i < 100; i++) {
    const auto aniso = random_anisotropy<3>();

    isosurface isosurf(bbox, resolution, aniso);
    signed_distance_from_plane field_fn(bbox.center(), vector3d::Random().normalized());

    points3d seed_points(1, 3);
    seed_points << vector3d::Zero();

    auto expected = isosurf.generate(field_fn, 1.0);
    isosurf.clear();
    auto actual = isosurf.generate_from_seed_points(seed_points, field_fn, 1.0);

    ASSERT_EQ(expected.faces().rows(), actual.faces().rows());
  }
}

TEST(isosurface, manifold) {
  const bbox3d bbox(point3d(-1.0, -1.0, -1.0), point3d(1.0, 1.0, 1.0));
  const auto resolution = 0.1;
  const auto aniso = random_anisotropy<3>();

  isosurface isosurf(bbox, resolution, aniso);
  random_field_function field_fn;

  // Do not use vertex refinement with random_field_function, as it may create singular vertices.
  auto surface = isosurf.generate(field_fn, 0.0, 0);

  mesh_defects_finder defects(surface.vertices(), surface.faces());

  ASSERT_TRUE(defects.singular_vertices().empty());
}

TEST(isosurface, boundary_coordinates) {
  const bbox3d bbox(point3d(-1.0, -1.0, -1.0), point3d(1.0, 1.0, 1.0));
  const auto resolution = 0.1;
  const auto aniso = random_anisotropy<3>();

  isosurface isosurf(bbox, resolution, aniso);
  random_field_function field_fn;

  auto surface = isosurf.generate(field_fn, 0.0, 0);

  ASSERT_TRUE(test_boundary_coordinates(surface, bbox));
}

TEST(isosurface, boundary_coordinates_seed_points) {
  const bbox3d bbox(point3d(-1.0, -1.0, -1.0), point3d(1.0, 1.0, 1.0));
  const auto resolution = 0.1;
  const auto aniso = random_anisotropy<3>();

  isosurface isosurf(bbox, resolution, aniso);
  random_field_function field_fn;

  points3d seed_points(1, 3);
  seed_points.row(0) = point3d(0.0, 0.0, 0.0);

  // Do not use vertex refinement with random_field_function, as it may create singular vertices.
  auto surface = isosurf.generate_from_seed_points(seed_points, field_fn, 0.0, 0);

  ASSERT_TRUE(test_boundary_coordinates(surface, bbox));
}
