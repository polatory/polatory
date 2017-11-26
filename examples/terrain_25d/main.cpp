// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <tuple>

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/io/read_table.hpp>
#include <polatory/isosurface/export_obj.hpp>
#include <polatory/isosurface/isosurface.hpp>
#include <polatory/isosurface/rbf_field_function_25d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/rbf/biharmonic.hpp>

#include "parse_options.hpp"

using polatory::geometry::points3d;
using polatory::interpolant;
using polatory::io::read_points_2d_and_values;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function_25d;
using polatory::point_cloud::distance_filter;
using polatory::rbf::biharmonic;

int main(int argc, const char *argv[]) {
  auto opts = parse_options(argc, argv);

  // Read points and normals.
  points3d points;
  Eigen::VectorXd values;
  std::tie(points, values) = read_points_2d_and_values(opts.in_file);

  // Remove very close points.
  distance_filter filter(points, opts.filter_distance);
  points = filter.filter_points(points);
  values = filter.filter_values(values);

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
  isosurf.generate_from_seed_points(points, field_f);

  export_obj(opts.mesh_file, isosurf);

  return 0;
}
