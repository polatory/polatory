// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <tuple>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/isosurface/export_obj.hpp>
#include <polatory/isosurface/isosurface.hpp>
#include <polatory/isosurface/rbf_field_function.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/point_cloud/normal_estimator.hpp>
#include <polatory/point_cloud/sdf_data_generator.hpp>
#include <polatory/rbf/biharmonic.hpp>
#include <polatory/table.hpp>

#include "parse_options.hpp"

using polatory::common::concatenate_cols;
using polatory::common::take_cols;
using polatory::interpolant;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::distance_filter;
using polatory::point_cloud::normal_estimator;
using polatory::point_cloud::sdf_data_generator;
using polatory::rbf::biharmonic;
using polatory::read_table;
using polatory::write_table;

int main(int argc, const char *argv[]) {
  auto opts = parse_options(argc, argv);

  // Read points.
  auto table = read_table(opts.in_file);
  auto terrain_points = take_cols(table, 0, 1, 2);

  // Estimate normals.
  normal_estimator norm_est(terrain_points);
  norm_est.estimate_with_knn(20, opts.min_plane_factor);
  norm_est.orient_normals_by_outward_vector({0, 0, 1});
  auto terrain_normals = norm_est.normals();

  // Generate SDF data.
  sdf_data_generator sdf_data(terrain_points, terrain_normals, opts.min_sdf_distance, opts.max_sdf_distance, 2.0);
  auto points = sdf_data.sdf_points();
  auto values = sdf_data.sdf_values();

  // Remove very close points.
  std::tie(points, values) = distance_filter(points, opts.filter_distance)
    .filtered(points, values);

  // Output SDF data (optional).
  if (!opts.sdf_data_file.empty()) {
    write_table(opts.sdf_data_file, concatenate_cols(points, values));
  }

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
  rbf_field_function field_f(interpolant);

  isosurf.generate_from_seed_points(terrain_points, field_f);

  export_obj(opts.mesh_file, isosurf);

  return 0;
}
