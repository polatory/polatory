#include <igl/AABB.h>
#include <igl/barycentric_coordinates.h>
#include <igl/read_triangle_mesh.h>

#include <Eigen/Core>
#include <cmath>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "parse_options.hpp"

using polatory::Index;
using polatory::Interpolant;
using polatory::Mat3;
using polatory::MatX;
using polatory::Model;
using polatory::read_table;
using polatory::VecX;
using polatory::geometry::Bbox3;
using polatory::geometry::Point3;
using polatory::geometry::Points3;
using polatory::geometry::Vector3;
using polatory::isosurface::Face;
using polatory::isosurface::Faces;
using polatory::isosurface::FieldFunction;
using polatory::isosurface::Isosurface;
using polatory::rbf::Biharmonic3D;

class MeshDistance {
  using Halfedge = std::pair<Index, Index>;

  struct HalfedgeHash {
    std::size_t operator()(const Halfedge& edge) const noexcept {
      std::size_t seed{};
      boost::hash_combine(seed, edge.first);
      boost::hash_combine(seed, edge.second);
      return seed;
    }
  };

 public:
  MeshDistance(Points3&& vertices, Faces&& faces)
      : vertices_(std::move(vertices)), faces_(std::move(faces)) {
    tree_.init(vertices_, faces_);

    for (auto f : faces_.rowwise()) {
      for (auto i = 0; i < 3; i++) {
        auto j = (i + 1) % 3;
        Halfedge he{f(i), f(j)};
        Halfedge he_opp{f(j), f(i)};
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

  std::pair<Points3, VecX> operator()(const Points3& points) const {
    VecX values(points.rows());
    Points3 closest_points(points.rows(), 3);

    for (Index i = 0; i < points.rows(); i++) {
      Point3 p = points.row(i);

      int fi{};
      Point3 closest_point;
      auto sqrd = tree_.squared_distance(vertices_, faces_, p, fi, closest_point);

      Face f = faces_.row(fi);
      Point3 a = vertices_.row(f(0));
      Point3 b = vertices_.row(f(1));
      Point3 c = vertices_.row(f(2));

      Vector3 l;
      igl::barycentric_coordinates(closest_point, a, b, c, l);

      auto boundary = false;
      for (auto i = 0; i < 3; i++) {
        auto j = (i + 1) % 3;
        auto k = (i + 2) % 3;
        Halfedge he{f(i), f(j)};
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
        Vector3 n = (b - a).cross(c - a).normalized();
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
  static double orient3d_inexact(const Point3& a, const Point3& b, const Point3& c,
                                 const Point3& d) {
    Mat3 m;
    m << a(0) - d(0), a(1) - d(1), a(2) - d(2), b(0) - d(0), b(1) - d(1), b(2) - d(2), c(0) - d(0),
        c(1) - d(1), c(2) - d(2);
    auto det = m.determinant();
    return det < 0.0 ? -1.0 : det > 0.0 ? 1.0 : 0.0;
  }

  Points3 vertices_;
  Faces faces_;
  igl::AABB<Points3, 3> tree_;
  std::unordered_multiset<Halfedge, HalfedgeHash> boundary_;
  std::unordered_set<Index> boundary_vertices_;
};

class OffsetFieldFunction : public FieldFunction {
  using Interpolant = Interpolant<3>;

 public:
  explicit OffsetFieldFunction(Interpolant& interpolant, const MeshDistance& dist, double accuracy)
      : interpolant_(interpolant), dist_(dist), accuracy_(accuracy) {}

  VecX operator()(const Points3& points) const override {
    auto [C, S] = dist_(points);

    return S - interpolant_.evaluate_impl(C);
  }

  void set_evaluation_bbox(const Bbox3& bbox) override {
    interpolant_.set_evaluation_bbox_impl(bbox, accuracy_);
  }

 private:
  Interpolant& interpolant_;
  const MeshDistance& dist_;
  double accuracy_;
};

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load the points.
    MatX table = read_table(opts.in);
    Points3 P = table(Eigen::all, {0, 1, 2});

    // Load the mesh.
    Points3 V;
    Faces F;
    if (!igl::read_triangle_mesh(opts.mesh_in, V, F)) {
      throw std::runtime_error("failed to read the mesh file");
    }

    MeshDistance mesh_dist(std::move(V), std::move(F));
    auto [C, S] = mesh_dist(P);

    // Define the model.
    Biharmonic3D<3> rbf({1.0});
    Model<3> model(std::move(rbf));

    // Fit.
    Interpolant<3> interpolant(model);
    if (opts.reduce) {
      interpolant.fit_incrementally(C, S, opts.tolerance, opts.max_iter, opts.accuracy);
    } else {
      interpolant.fit(C, S, opts.tolerance, opts.max_iter, opts.accuracy);
    }

    // Generate the isosurface.
    Isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    OffsetFieldFunction field_fn(interpolant, mesh_dist, opts.accuracy);

    isosurf.generate_from_seed_points(P, field_fn).export_obj(opts.mesh_out);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "unknown error" << std::endl;
    return 1;
  }
}
