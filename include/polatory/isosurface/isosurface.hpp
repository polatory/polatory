#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/clip.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/rmt/lattice.hpp>
#include <polatory/isosurface/sign.hpp>
#include <stdexcept>
#include <unordered_set>

namespace polatory::isosurface {

class isosurface {
  using Mesh = mesh;

 public:
  isosurface(const geometry::bbox3d& bbox, double resolution)
      : isosurface(bbox, resolution, geometry::matrix3d::Identity()) {}

  isosurface(const geometry::bbox3d& bbox, double resolution, const geometry::matrix3d& aniso)
      : lattice_(bbox, resolution, aniso) {
    if (!(aniso.determinant() > 0.0)) {
      throw std::invalid_argument("aniso must have a positive determinant");
    }
  }

  void clear() { lattice_.clear(); }

  Mesh generate(field_function& field_fn, double isovalue = 0.0, int refine = 1) {
    if (refine < 0) {
      throw std::runtime_error("refine must be non-negative");
    }

    field_fn.set_evaluation_bbox(lattice_.second_extended_bbox());

    lattice_.add_all_nodes(field_fn, isovalue);
    lattice_.refine_vertices(field_fn, isovalue, refine);

    return generate_common();
  }

  Mesh generate_from_seed_points(const geometry::points3d& seed_points, field_function& field_fn,
                                 double isovalue = 0.0, int refine = 1) {
    if (seed_points.rows() == 0) {
      throw std::runtime_error("seed points must not be empty");
    }

    if (refine < 0) {
      throw std::runtime_error("refine must be non-negative");
    }

    field_fn.set_evaluation_bbox(lattice_.second_extended_bbox());

    lattice_.add_nodes_from_seed_points(seed_points, field_fn, isovalue);
    lattice_.refine_vertices(field_fn, isovalue, refine);

    return generate_common();
  }

 private:
  Mesh generate_common() {
    lattice_.cluster_vertices();

    Mesh mesh;

    // Unclustering non-manifold vertices may require a few iterations.
    while (true) {
      mesh = lattice_.get_mesh();
      mesh_defects_finder defects(mesh);

      auto vis = defects.singular_vertices();
      auto fis = defects.intersecting_faces();

      std::unordered_set<index_t> vertices_to_uncluster;
      for (auto vi : vis) {
        vertices_to_uncluster.insert(vi);
      }
      for (auto fi : fis) {
        auto f = mesh.faces().row(fi);
        vertices_to_uncluster.insert(f(0));
        vertices_to_uncluster.insert(f(1));
        vertices_to_uncluster.insert(f(2));
      }

      auto num_unclustered = lattice_.uncluster_vertices(vertices_to_uncluster);
      if (num_unclustered == 0) {
        break;
      }
    }

    mesh = clip(mesh, lattice_.bbox());

    if (mesh.is_empty() &&
        lattice_.value_sign_at_arbitrary_point_within_bbox() == binary_sign::kNeg) {
      mesh = Mesh(entire_tag{});
    }

    lattice_.clear();

    return mesh;
  }

  rmt::lattice lattice_;
};

}  // namespace polatory::isosurface
