#include <igl/AABB.h>
#include <igl/barycentric_coordinates.h>
#include <igl/read_triangle_mesh.h>

#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "parse_options.hpp"

using polatory::index_t;
using polatory::interpolant;
using polatory::model;
using polatory::read_table;
using polatory::tabled;
using polatory::common::valuesd;
using polatory::geometry::bbox3d;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::geometry::vector3d;
using polatory::isosurface::field_function;
using polatory::isosurface::isosurface;
using polatory::point_cloud::distance_filter;
using polatory::rbf::biharmonic3d;
using face = Eigen::Matrix<index_t, 1, 3>;
using faces = Eigen::Matrix<index_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

class mesh_distance {
  using halfedge = std::pair<index_t, index_t>;

  struct halfedge_hash {
    std::size_t operator()(const halfedge& edge) const noexcept {
      std::size_t seed{};
      boost::hash_combine(seed, std::hash<index_t>{}(edge.first));
      boost::hash_combine(seed, std::hash<index_t>{}(edge.second));
      return seed;
    }
  };

 public:
  mesh_distance(points3d&& vertices, faces&& faces)
      : vertices_(std::move(vertices)), faces_(std::move(faces)) {
    tree_.init(vertices_, faces_);

    for (auto f : faces_.rowwise()) {
      for (auto i = 0; i < 3; i++) {
        auto j = (i + 1) % 3;
        halfedge he{f(i), f(j)};
        halfedge he_opp{f(j), f(i)};
        auto it = boundary_.find(he_opp);
        if (it != boundary_.end()) {
          boundary_.erase(it);
        } else {
          boundary_.insert(he);
        }
      }
    }

    for (auto he : boundary_) {
      boundary_vertices_.insert(he.first);
      boundary_vertices_.insert(he.second);
    }
  }

  std::pair<points3d, valuesd> operator()(const points3d& points) const {
    valuesd values(points.rows());
    points3d closest_points(points.rows(), 3);

    for (index_t i = 0; i < points.rows(); i++) {
      point3d p = points.row(i);

      int fi;
      point3d closest_point;
      auto sqrd = tree_.squared_distance(vertices_, faces_, p, fi, closest_point);

      face f = faces_.row(fi);
      point3d a = vertices_.row(f(0));
      point3d b = vertices_.row(f(1));
      point3d c = vertices_.row(f(2));

      vector3d l;
      igl::barycentric_coordinates(closest_point, a, b, c, l);

      auto boundary = false;
      for (auto i = 0; i < 3; i++) {
        auto j = (i + 1) % 3;
        auto k = (i + 2) % 3;
        halfedge he{f(i), f(j)};
        if (boundary_.contains(he) && std::abs(l(k)) < 1e-10) {
          boundary = true;
          break;
        }

        if (boundary_vertices_.contains(f(i)) && std::abs(1.0 - l(i)) < 1e-10) {
          boundary = true;
          break;
        }
      }

      if (boundary) {
        vector3d n = (b - a).cross(c - a).normalized();
        values(i) = n.dot(p - a);
      } else {
        auto sign = -orient3d_inexact(a, b, c, p);
        values(i) = sign * std::sqrt(sqrd);
      }

      closest_points.row(i) = closest_point;
    }

    return {std::move(closest_points), std::move(values)};
  }

 private:
  static double orient3d_inexact(const point3d& a, const point3d& b, const point3d& c,
                                 const point3d& d) {
    Eigen::Matrix3d m;
    m << a(0) - d(0), a(1) - d(1), a(2) - d(2), b(0) - d(0), b(1) - d(1), b(2) - d(2), c(0) - d(0),
        c(1) - d(1), c(2) - d(2);
    auto det = m.determinant();
    return det < 0.0 ? -1.0 : det > 0.0 ? 1.0 : 0.0;
  }

  points3d vertices_;
  faces faces_;
  igl::AABB<points3d, 3> tree_;
  std::unordered_multiset<halfedge, halfedge_hash> boundary_;
  std::unordered_set<index_t> boundary_vertices_;
};

class offset_field_function : public field_function {
  using Interpolant = interpolant<3>;

 public:
  explicit offset_field_function(Interpolant& interpolant, const mesh_distance& dist)
      : interpolant_(interpolant), dist_(dist) {}

  valuesd operator()(const points3d& points) const override {
    auto [C, S] = dist_(points);

    return S - interpolant_.evaluate_impl(C);
  }

  void set_evaluation_bbox(const bbox3d& bbox) override {
    interpolant_.set_evaluation_bbox_impl(bbox);
  }

 private:
  Interpolant& interpolant_;
  const mesh_distance& dist_;
};

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load the points.
    tabled table = read_table(opts.in);
    points3d P = table(Eigen::all, {0, 1, 2});

    // Load the mesh.
    points3d V;
    faces F;
    if (!igl::read_triangle_mesh(opts.mesh_in, V, F)) {
      throw std::runtime_error("Failed to read mesh file.");
    }

    mesh_distance mesh_dist(std::move(V), std::move(F));
    auto [C, S] = mesh_dist(P);

    // Remove very close points.
    distance_filter filter(C, opts.min_distance);
    std::tie(C, S) = filter(C, S);

    // Define the model.
    biharmonic3d<3> rbf({1.0});
    model<3> model(std::move(rbf), 0);

    // Fit.
    interpolant<3> interpolant(model);
    if (opts.reduce) {
      interpolant.fit_incrementally(C, S, opts.absolute_tolerance, opts.max_iter);
    } else {
      interpolant.fit(C, S, opts.absolute_tolerance, opts.max_iter);
    }

    // Generate the isosurface.
    isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    offset_field_function field_fn(interpolant, mesh_dist);

    isosurf.generate_from_seed_points(P, field_fn).export_obj(opts.mesh_out);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
