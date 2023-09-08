#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <tuple>

#include "parse_options.hpp"

using polatory::index_t;
using polatory::interpolant;
using polatory::model;
using polatory::read_table;
using polatory::tabled;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::distance_filter;

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load points (x,y,z) and values (value).
    tabled table = read_table(opts.in_file);
    points3d points = table(Eigen::all, {0, 1, 2});
    valuesd values = table.col(3);

    // Remove very close points.
    std::tie(points, values) = distance_filter(points, opts.min_distance)(points, values);

    // Define the model.
    auto rbf = make_rbf(opts.rbf_name, opts.rbf_params);
    rbf->set_anisotropy(opts.aniso);
    model model(*rbf, 3, opts.poly_degree);
    model.set_nugget(opts.nugget);

    // Fit.
    interpolant interpolant(model);
    if (opts.reduce) {
      interpolant.fit_incrementally(points, values, opts.absolute_tolerance, opts.max_iter);
    } else {
      interpolant.fit(points, values, opts.absolute_tolerance, opts.max_iter);
    }

    // Generate the isosurface.
    auto n_surface_points = (values.array() == 0.0).count();
    points3d surface_points(n_surface_points, 3);
    index_t si{};
    for (index_t i = 0; i < points.rows(); i++) {
      if (values(i) == 0.0) {
        surface_points.row(si++) = points.row(i);
      }
    }

    isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    rbf_field_function field_fn(interpolant);

    isosurf.generate_from_seed_points(surface_points, field_fn).export_obj(opts.mesh_file);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
