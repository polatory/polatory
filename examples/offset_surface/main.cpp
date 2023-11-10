#include <igl/read_triangle_mesh.h>
#include <igl/signed_distance.h>

#include <exception>
#include <iostream>
#include <optional>
#include <polatory/polatory.hpp>
#include <tuple>

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
using polatory::geometry::vectors3d;
using polatory::isosurface::field_function;
using polatory::isosurface::isosurface;
using polatory::point_cloud::distance_filter;
using polatory::rbf::biharmonic3d;
using faces = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;
using indices = Eigen::VectorXi;

class mesh_distance {
 public:
  mesh_distance(points3d&& vertices, faces&& faces)
      : vertices_(std::move(vertices)), faces_(std::move(faces)) {
    tree_.init(vertices_, faces_);
  }

  std::pair<points3d, valuesd> operator()(const points3d& points) const {
    valuesd values(points.rows());
    points3d closest_points(points.rows(), 3);

    for (index_t i = 0; i < points.rows(); i++) {
      point3d p = points.row(i);

      int fi;
      point3d closest_point;
      auto sqrd = tree_.squared_distance(vertices_, faces_, p, fi, closest_point);

      point3d a = vertices_.row(faces_(fi, 0));
      point3d b = vertices_.row(faces_(fi, 1));
      point3d c = vertices_.row(faces_(fi, 2));
      auto sign = -orient3d_inexact(a, b, c, p);

      values(i) = sign * std::sqrt(sqrd);
      closest_points.row(i) = closest_point;

      // TODO: Compute the barycentric coordinates of c in the triangle fi.
      // If c is on the boundary of the mesh, recompute the signed distance
      // as the distance between p and the supporting plane of the triangle..
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
};

template <class Model>
class offset_field_function : public field_function {
  static_assert(Model::kDim == 3, "Model must be three-dimensional.");

  using Interpolant = interpolant<Model>;

 public:
  explicit offset_field_function(Interpolant& interpolant, const mesh_distance& dist)
      : interpolant_(interpolant), dist_(dist) {}

  valuesd operator()(const points3d& points) const override {
    auto [closest_points, values] = dist_(points);

    return values - interpolant_.evaluate_impl(closest_points);
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

    // Load points (x,y,z).
    tabled table = read_table(opts.in);
    points3d P = table(Eigen::all, {0, 1, 2});

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
    model model(rbf, 0);

    // Fit.
    interpolant interpolant(model);
    interpolant.fit(C, S, opts.absolute_tolerance, opts.max_iter);

    // Generate the isosurface.
    isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    offset_field_function field_fn(interpolant, mesh_dist);

    isosurf.generate(field_fn).export_obj(opts.mesh_out);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
