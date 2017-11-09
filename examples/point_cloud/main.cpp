// Copyright (c) 2016, GSI and The Polatory Authors.

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "polatory/geometry/bbox3d.hpp"
#include "polatory/interpolant.hpp"
#include "polatory/io/read_table.hpp"
#include "polatory/io/write_table.hpp"
#include "polatory/isosurface/export_obj.hpp"
#include "polatory/isosurface/isosurface.hpp"
#include "polatory/isosurface/rbf_field_function.hpp"
#include "polatory/point_cloud/distance_filter.hpp"
#include "polatory/point_cloud/sdf_data_generator.hpp"
#include "polatory/rbf/biharmonic.hpp"

using polatory::geometry::bbox3d;
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
  if (argc < 10) {
    std::cerr << "Usage: " << argv[0] << " in_file min_sdf_distance max_sdf_distance" << std::endl
              << "  filter_distance fit_incrementally(0|1) fitting_accuracy" << std::endl
              << "  mesh_resolution out_sdf_data_file out_obj_file" << std::endl;
    return 1;
  }

  auto in_file = argv[1];
  auto min_normal_distance = std::stod(argv[2]);
  auto max_normal_distance = std::stod(argv[3]);
  auto filter_distance = std::stod(argv[4]);
  auto fit_incrementally = static_cast<bool>(std::stoi(argv[5]));
  auto fitting_accuracy = std::stod(argv[6]);
  auto mesh_resolution = std::stod(argv[7]);
  auto out_sdf_data_file = argv[8];
  auto out_obj_file = argv[9];

  // Read points and normals.
  std::vector<Eigen::Vector3d> cloud_points;
  std::vector<Eigen::Vector3d> cloud_normals;
  std::tie(cloud_points, cloud_normals) = read_points_and_normals(in_file);

  // Generate SDF data.
  sdf_data_generator sdf_data(
    cloud_points, cloud_normals,
    min_normal_distance, max_normal_distance);
  auto points = sdf_data.sdf_points();
  auto values = sdf_data.sdf_values();

  // Remove very close points.
  distance_filter filter(points, filter_distance);
  points = filter.filter_points(points);
  values = filter.filter_values(values);

  // Output SDF data (optional).
  write_points_and_values(out_sdf_data_file, points, values);

  // Define model.
  biharmonic rbf({ 1.0, 0.0 });
  interpolant interpolant(rbf, 3, 0);

  // Fit.
  if (fit_incrementally) {
    interpolant.fit_incrementally(points, values, fitting_accuracy);
  } else {
    interpolant.fit(points, values, fitting_accuracy);
  }
  std::cout << "Number of RBF centers: " << interpolant.centers().size() << std::endl;

  // Generate isosurface.
  auto mesh_bbox = bbox3d::from_points(cloud_points);
  mesh_bbox = bbox3d(mesh_bbox.min() - 0.1 * mesh_bbox.size(), mesh_bbox.max() + 0.1 * mesh_bbox.size());
  polatory::isosurface::isosurface isosurf(mesh_bbox, mesh_resolution);
  rbf_field_function field_f(interpolant);

  auto n_seed_points = cloud_points.size() / 10;
  std::vector<Eigen::Vector3d> seed_points(cloud_points.begin(),
                                           cloud_points.begin() + n_seed_points);
  isosurf.generate_from_seed_points(seed_points, field_f);

  export_obj(out_obj_file, isosurf);

  return 0;
}
