// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>

#include <Eigen/Core>

#include "polatory/interpolant.hpp"
#include "polatory/io/read_table.hpp"
#include "polatory/isosurface/export_obj.hpp"
#include "polatory/isosurface/isosurface.hpp"
#include "polatory/isosurface/rbf_field_function.hpp"
#include "polatory/point_cloud/distance_filter.hpp"
#include "polatory/point_cloud/normal_estimator.hpp"
#include "polatory/point_cloud/sdf_data_generator.hpp"
#include "polatory/rbf/biharmonic.hpp"

#include "parse_options.hpp"

using polatory::interpolant;
using polatory::io::read_points;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::distance_filter;
using polatory::point_cloud::normal_estimator;
using polatory::point_cloud::sdf_data_generator;
using polatory::rbf::biharmonic;

int main(int argc, const char *argv[]) {
  auto opts = parse_options(argc, argv);

  // Read points.
  auto terrain_points = read_points(opts.in_file);

  // Estimate normals.
  normal_estimator norm_est(terrain_points);
  norm_est.estimate_with_knn(20);
  norm_est.orient_normals_by_outward_vector(Eigen::Vector3d(0, 0, 1));
  auto terrain_normals = norm_est.normals();

  // Generate SDF data.
  sdf_data_generator sdf_data(terrain_points, terrain_normals, opts.min_sdf_distance, opts.max_sdf_distance, 2.0);
  auto points = sdf_data.sdf_points();
  auto values = sdf_data.sdf_values();

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
  std::cout << "Number of RBF centers: " << interpolant.centers().size() << std::endl;

  // Generate isosurface.
  polatory::isosurface::isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
  rbf_field_function field_f(interpolant);

  isosurf.generate_from_seed_points(terrain_points, field_f);

  export_obj(opts.mesh_file, isosurf);

  return 0;
}
