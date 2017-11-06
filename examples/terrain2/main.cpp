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
#include "polatory/isosurface/export_obj.hpp"
#include "polatory/isosurface/isosurface.hpp"
#include "polatory/isosurface/rbf_field_function.hpp"
#include "polatory/point_cloud/distance_filter.hpp"
#include "polatory/point_cloud/normal_estimator.hpp"
#include "polatory/point_cloud/sdf_data_generator.hpp"
#include "polatory/rbf/biharmonic.hpp"

using polatory::geometry::bbox3d;
using polatory::interpolant;
using polatory::io::read_points;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::distance_filter;
using polatory::point_cloud::normal_estimator;
using polatory::point_cloud::sdf_data_generator;
using polatory::rbf::biharmonic;

int main(int argc, char *argv[]) {
  if (argc < 7) {
    std::cerr << "Usage: " << argv[0] << " in_file" << std::endl
              << "  filter_distance fit_incrementally(0|1) fitting_accuracy" << std::endl
              << "  mesh_resolution out_obj_file" << std::endl;
    return 1;
  }

  auto in_file = argv[1];
  auto filter_distance = std::stod(argv[2]);
  auto fit_incrementally = static_cast<bool>(std::stoi(argv[3]));
  auto fitting_accuracy = std::stod(argv[4]);
  auto mesh_resolution = std::stod(argv[5]);
  auto out_obj_file = argv[6];

  // Read points.
  auto terrain_points = read_points(in_file);

  // Estimate normals.
  normal_estimator norm_est(terrain_points);
  norm_est.estimate_with_knn(20);
  norm_est.orient_normals_by_outward_vector(Eigen::Vector3d(0, 0, 1));
  auto terrain_normals = norm_est.normals();

  // Generate SDF data.
  sdf_data_generator sdf_data(terrain_points, terrain_normals, 1.0, 2.0, 2.0);
  auto points = sdf_data.sdf_points();
  auto values = sdf_data.sdf_values();

  // Remove very close points.
  distance_filter filter(points, filter_distance);
  points = filter.filter_points(points);
  values = filter.filter_values(values);

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
  auto mesh_bbox = bbox3d::from_points(terrain_points);
  mesh_bbox = bbox3d(mesh_bbox.min() - 0.1 * mesh_bbox.size(), mesh_bbox.max() + 0.1 * mesh_bbox.size());
  polatory::isosurface::isosurface isosurf(mesh_bbox, mesh_resolution);
  rbf_field_function field_f(interpolant);

  isosurf.generate_from_seed_points(terrain_points, field_f);

  export_obj(out_obj_file, isosurf);

  return 0;
}
