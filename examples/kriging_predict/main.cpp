// Copyright (c) 2016, GSI and The Polatory Authors.

#include <exception>
#include <iostream>
#include <tuple>

#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::common::take_cols;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolant;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::distance_filter;
using polatory::rbf::cov_quasi_spherical9;
using polatory::read_table;

int main(int argc, const char *argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    auto table = read_table(opts.in_file);
    points3d points = take_cols(table, 0, 1, 2);
    valuesd values = table.col(3);

    // Remove very close points.
    std::tie(points, values) = distance_filter(points, opts.filter_distance)
      .filtered(points, values);

    // Define model.
    cov_quasi_spherical9 rbf({ opts.psill, opts.range, opts.nugget });
    interpolant interpolant(rbf, opts.poly_dimension, opts.poly_degree);

    // Fit.
    interpolant.fit(points, values, opts.absolute_tolerance);

    // Generate isosurface of given values.
    isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    rbf_field_function field_f(interpolant);

    for (auto isovalue_name : opts.mesh_values_files) {
      isosurf.generate(field_f, isovalue_name.first)
        .export_obj(isovalue_name.second);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
