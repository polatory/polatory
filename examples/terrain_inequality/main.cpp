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
using polatory::isosurface::isosurface_2d;
using polatory::isosurface::rbf_field_function;
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
    valuesd values_lb = table.col(3);
    valuesd values_ub = table.col(4);

    // Remove very close points.
    std::tie(points, values, values_lb, values_ub) = distance_filter(points, opts.min_distance)
      .filtered(points, values, values_lb, values_ub);

    // Define model.
    model model(biharmonic3d({ 1.0, opts.smooth }), opts.poly_dimension, opts.poly_degree);
    interpolant interpolant(model);

    interpolant.fit_inequality(points, values, values_lb, values_ub, opts.absolute_tolerance);
    std::cout << "Number of RBF centers: " << interpolant.centers().rows() << std::endl;

    // Generate isosurface.
    isosurface_2d isosurf(opts.mesh_bbox, opts.mesh_resolution);
    rbf_field_function field_fn(interpolant);

    points.col(2) = values;
    isosurf.generate(field_fn)
      .export_obj(opts.mesh_file);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
