// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <tuple>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/types.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/isosurface/isosurface.hpp>
#include <polatory/isosurface/rbf_field_function_25d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/rbf/biharmonic.hpp>
#include <polatory/table.hpp>

#include "parse_options.hpp"

using polatory::common::concatenate_cols;
using polatory::common::take_cols;
using polatory::common::valuesd;
using polatory::interpolant;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function_25d;
using polatory::point_cloud::distance_filter;
using polatory::rbf::biharmonic;
using polatory::read_table;

int main(int argc, const char *argv[]) {
  auto opts = parse_options(argc, argv);

  // Read points and normals.
  auto table = read_table(opts.in_file);
  auto points = concatenate_cols(take_cols(table, 0, 1), valuesd::Zero(table.rows()));
  auto values = table.col(2);

  // Remove very close points.
  std::tie(points, values) = distance_filter(points, opts.filter_distance)
    .filtered(points, values);

  // Define model.
  biharmonic rbf({ 1.0, opts.rho });
  interpolant interpolant(rbf, opts.poly_dimension, opts.poly_degree);

  // Fit.
  if (opts.incremental_fit) {
    interpolant.fit_incrementally(points, values, opts.absolute_tolerance);
  } else {
    interpolant.fit(points, values, opts.absolute_tolerance);
  }
  std::cout << "Number of RBF centers: " << interpolant.centers().rows() << std::endl;

  // Generate isosurface.
  polatory::isosurface::isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
  rbf_field_function_25d field_f(interpolant);

  points.col(2) = values;
  isosurf.generate_from_seed_points(points, field_f)
    .export_obj(opts.mesh_file);

  return 0;
}
