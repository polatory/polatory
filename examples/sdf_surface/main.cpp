// Copyright (c) 2016, GSI and The Polatory Authors.

#include <exception>
#include <iostream>
#include <tuple>

#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::common::take_cols;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::index_t;
using polatory::interpolant;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::model;
using polatory::point_cloud::distance_filter;
using polatory::read_table;
using polatory::tabled;

int main(int argc, const char *argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load points (x,y,z) and values (value).
    tabled table = read_table(opts.in_file);
    points3d points = take_cols(table, 0, 1, 2);
    valuesd values = table.col(3);

    // Remove very close points.
    std::tie(points, values) = distance_filter(points, opts.min_distance)
      .filtered(points, values);

    // Define the model.
    auto rbf = make_rbf(opts.rbf_name, opts.rbf_params);
    model model(*rbf, 3, opts.poly_degree);
    model.set_nugget(opts.nugget);

    // Fit.
    interpolant interpolant(model);
    interpolant.fit(points, values, opts.absolute_tolerance);

    // Generate the isosurface.
    index_t n_surface_points = static_cast<index_t>((values.array() == 0.0).count());
    points3d surface_points(n_surface_points, 3);
    index_t si = 0;
    for (index_t i = 0; i < points.rows(); i++) {
      if (values(i) == 0.0) {
        surface_points.row(si++) = points.row(i);
      }
    }

    isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    rbf_field_function field_fn(interpolant);

    isosurf.generate_from_seed_points(surface_points, field_fn)
      .export_obj(opts.mesh_file);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
