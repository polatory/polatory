#include <igl/AABB.h>
#include <igl/barycentric_coordinates.h>
#include <igl/read_triangle_mesh.h>

#include <Eigen/Core>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "parse_options.hpp"

using polatory::Index;
using polatory::Interpolant;
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

class SignedDistanceField {
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
  SignedDistanceField(Points3&& vertices, Faces&& faces)
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
    Points3 C(points.rows(), 3);
    VecX D(points.rows());

    for (Index i = 0; i < points.rows(); i++) {
      Point3 p = points.row(i);

      int fi{};
      Point3 closest_point;
      auto d2 = tree_.squared_distance(vertices_, faces_, p, fi, closest_point);

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
        if (std::abs(l(k)) < 1e-10 && boundary_.contains(he)) {
          boundary = true;
          break;
        }

        if (std::abs(1.0 - l(i)) < 1e-10 && boundary_vertices_.contains(f(i))) {
          boundary = true;
          break;
        }
      }

      Vector3 n = (b - a).cross(c - a).normalized();
      auto dot = n.dot(p - a);

      C.row(i) = closest_point;
      D(i) = boundary ? dot : std::copysign(std::sqrt(d2), dot);
    }

    return {std::move(C), std::move(D)};
  }

 private:
  Points3 vertices_;
  Faces faces_;
  igl::AABB<Points3, 3> tree_;
  std::unordered_set<Halfedge, HalfedgeHash> boundary_;
  std::unordered_set<Index> boundary_vertices_;
};

class OffsetFieldFunction : public FieldFunction {
  using Interpolant = Interpolant<3>;

 public:
  explicit OffsetFieldFunction(Interpolant& interpolant, const SignedDistanceField& sdf,
                               double accuracy)
      : interpolant_(interpolant), sdf_(sdf), accuracy_(accuracy) {}

  VecX operator()(const Points3& points) const override {
    auto [C, D] = sdf_(points);

    return D - interpolant_.evaluate_impl(C);
  }

  void set_evaluation_bbox(const Bbox3& bbox) override {
    interpolant_.set_evaluation_bbox_impl(bbox, accuracy_);
  }

 private:
  Interpolant& interpolant_;
  const SignedDistanceField& sdf_;
  double accuracy_;
};

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load the points and their sides of the mesh.
    MatX table = read_table(opts.in);
    Points3 points = table(Eigen::all, {0, 1, 2});
    VecX sides = table.col(3);

    // Load the mesh.
    Points3 V;
    Faces F;
    if (!igl::read_triangle_mesh(opts.mesh_in, V, F)) {
      throw std::runtime_error("failed to read the mesh file");
    }

    SignedDistanceField sdf(std::move(V), std::move(F));
    auto [C, D] = sdf(points);

    auto nan = std::numeric_limits<double>::quiet_NaN();
    VecX DL(VecX::Constant(points.rows(), nan));
    VecX DU(VecX::Constant(points.rows(), nan));
    for (Index i = 0; i < points.rows(); i++) {
      if (sides(i) < 0.0) {
        DL(i) = D(i);
        D(i) = nan;
      } else if (sides(i) > 0.0) {
        DU(i) = D(i);
        D(i) = nan;
      }
    }

    // Define the model.
    Biharmonic3D<3> rbf({1.0});
    Model<3> model(std::move(rbf));

    // Fit.
    Interpolant<3> interpolant(model);
    interpolant.fit_inequality(C, D, DL, DU, opts.tolerance, opts.max_iter, opts.accuracy);

    // Generate the isosurface.
    Isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    OffsetFieldFunction field_fn(interpolant, sdf, opts.accuracy);

    isosurf.generate_from_seed_points(points, field_fn).export_obj(opts.mesh_out);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "unknown error" << std::endl;
    return 1;
  }
}
