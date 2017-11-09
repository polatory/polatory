// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "polatory/interpolant.hpp"
#include "polatory/io/read_table.hpp"
#include "polatory/io/write_table.hpp"
#include "polatory/isosurface/export_obj.hpp"
#include "polatory/isosurface/isosurface.hpp"
#include "polatory/isosurface/rbf_field_function.hpp"
#include "polatory/point_cloud/distance_filter.hpp"
#include "polatory/point_cloud/sdf_data_generator.hpp"
#include "polatory/rbf/biharmonic.hpp"

#include "parse_options.hpp"

using polatory::interpolant;
using polatory::io::read_points_and_normals;
using polatory::io::write_points_and_values;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::distance_filter;
using polatory::point_cloud::sdf_data_generator;
using polatory::rbf::biharmonic;

int main(int argc, const char *argv[]) {
  auto opts = parse_options(argc, argv);

  // Read points and normals.
  std::vector<Eigen::Vector3d> cloud_points;
  std::vector<Eigen::Vector3d> cloud_normals;
  std::tie(cloud_points, cloud_normals) = read_points_and_normals(opts.in_file);

  // Generate SDF data.
  sdf_data_generator sdf_data(
    cloud_points, cloud_normals,
    opts.min_sdf_distance, opts.max_sdf_distance);
  auto points = sdf_data.sdf_points();
  auto values = sdf_data.sdf_values();

  // Remove very close points.
  distance_filter filter(points, opts.filter_distance);
  points = filter.filter_points(points);
  values = filter.filter_values(values);

  // Output SDF data (optional).
  if (!opts.sdf_data_file.empty()) {
    write_points_and_values(opts.sdf_data_file, points, values);
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
  std::cout << "Number of RBF centers: " << interpolant.centers().size() << std::endl;

  // Generate isosurface.
  polatory::isosurface::isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
  rbf_field_function field_f(interpolant);

  auto n_seed_points = std::max(cloud_points.size() / 10, size_t(100));
  std::vector<Eigen::Vector3d> seed_points(cloud_points.begin(),
                                           cloud_points.begin() + n_seed_points);
  isosurf.generate_from_seed_points(seed_points, field_f);

  export_obj(opts.mesh_file, isosurf);

  return 0;
}
