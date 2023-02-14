#include <gtest/gtest.h>

#include <boost/container_hash/hash.hpp>
#include <functional>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/isosurface.hpp>
#include <polatory/types.hpp>
#include <unordered_set>
#include <utility>

using polatory::common::valuesd;
using polatory::geometry::bbox3d;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::isosurface::field_function;
using polatory::isosurface::isosurface;
using polatory::isosurface::vertex_index;

namespace {

class distance_from_origin : public field_function {
 public:
  valuesd operator()(const points3d& points) const override { return points.rowwise().norm(); }
};

class random_field_function : public field_function {
 public:
  valuesd operator()(const points3d& points) const override {
    return valuesd::Random(points.rows());
  }
};

}  // namespace

TEST(isosurface, generate) {
  const bbox3d bbox(point3d(-2.0, -2.0, -2.0), point3d(2.0, 2.0, 2.0));
  const auto resolution = 0.1;

  isosurface isosurf(bbox, resolution);
  distance_from_origin field_fn;

  auto surface = isosurf.generate(field_fn, 1.0);

  // TODO(mizuno): Check topological properties instead.
  ASSERT_EQ(1466u, surface.vertices().size());
  ASSERT_EQ(2928u, surface.faces().size());
}

TEST(isosurface, generate_from_seed_points) {
  const bbox3d bbox(point3d(-2.0, -2.0, -2.0), point3d(2.0, 2.0, 2.0));
  const auto resolution = 0.1;

  points3d seed_points(1, 3);
  seed_points.row(0) = point3d(1.0, 0.0, 0.0);

  isosurface isosurf(bbox, resolution);
  distance_from_origin field_fn;

  auto surface = isosurf.generate_from_seed_points(seed_points, field_fn, 1.0);

  ASSERT_EQ(1466u, surface.vertices().size());
  ASSERT_EQ(2928u, surface.faces().size());
}

using halfedge = std::pair<vertex_index, vertex_index>;

template <>
struct std::hash<halfedge> {
  std::size_t operator()(const halfedge& e) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, std::hash<vertex_index>()(e.first));
    boost::hash_combine(seed, std::hash<vertex_index>()(e.second));
    return seed;
  }
};

TEST(isosurface, boundary_coordinates) {
  const bbox3d bbox(point3d(-1.0, -1.0, -1.0), point3d(1.0, 1.0, 1.0));
  const auto resolution = 0.1;

  isosurface isosurf(bbox, resolution);
  random_field_function field_fn;

  auto surface = isosurf.generate(field_fn, 0.0);

  std::unordered_set<halfedge> boundary_hes;
  for (const auto& f : surface.faces()) {
    for (std::size_t i = 0; i < 3; i++) {
      auto he = std::make_pair(f.at(i), f.at((i + 1) % 3));
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

  for (auto vi : boundary_vertices) {
    const auto& p = surface.vertices().at(vi);
    ASSERT_TRUE((p.array() == bbox.min().array() || p.array() == bbox.max().array()).any());
  }
}
