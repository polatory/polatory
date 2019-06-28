// Copyright (c) 2016, GSI and The Polatory Authors.

#include <exception>
#include <iostream>
#include <tuple>

#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::common::concatenate_cols;
using polatory::common::take_cols;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolant;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function_25d;
using polatory::model;
using polatory::point_cloud::distance_filter;
using polatory::rbf::biharmonic3d;
using polatory::read_table;

int main(int argc, const char *argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Read points and normals.
    auto table = read_table(opts.in_file);
    points3d points = concatenate_cols(take_cols(table, 0, 1), valuesd::Zero(table.rows()));
    valuesd values = table.col(2);

    // Remove very close points.
    std::tie(points, values) = distance_filter(points, opts.min_distance)
      .filtered(points, values);

    // Define model.
    model model(biharmonic3d({ 1.0 }), opts.poly_dimension, opts.poly_degree);
    model.set_nugget(opts.smooth);
    interpolant interpolant(model);

    // Fit.
    if (opts.incremental_fit) {
      interpolant.fit_incrementally(points, values, opts.absolute_tolerance);
    } else {
      interpolant.fit(points, values, opts.absolute_tolerance);
    }
    std::cout << "Number of RBF centers: " << interpolant.centers().rows() << std::endl;

    // Generate isosurface.
    polatory::isosurface::isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    rbf_field_function_25d field_fn(interpolant);

    points.col(2) = values;
    isosurf.generate_from_seed_points(points, field_fn)
      .export_obj(opts.mesh_file);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
