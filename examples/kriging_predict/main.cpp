// Copyright (c) 2016, GSI and The Polatory Authors.

#include <tuple>

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/io/read_table.hpp>
#include <polatory/isosurface/export_obj.hpp>
#include <polatory/isosurface/isosurface.hpp>
#include <polatory/isosurface/rbf_field_function.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/rbf/cov_quasi_spherical9.hpp>

#include "parse_options.hpp"

using polatory::geometry::points3d;
using polatory::interpolant;
using polatory::io::read_points_and_values;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::distance_filter;
using polatory::rbf::cov_quasi_spherical9;

int main(int argc, const char *argv[]) {
  auto opts = parse_options(argc, argv);

  points3d points;
  Eigen::VectorXd values;
  std::tie(points, values) = read_points_and_values(opts.in_file);

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
    isosurf.generate(field_f, isovalue_name.first);
    export_obj(isovalue_name.second, isosurf);
  }

  return 0;
}
