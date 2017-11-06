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
#include "polatory/isosurface/rbf_field_function_25d.hpp"
#include "polatory/point_cloud/distance_filter.hpp"
#include "polatory/rbf/biharmonic.hpp"

using polatory::geometry::bbox3d;
using polatory::interpolant;
using polatory::io::read_points_2d_and_values;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function_25d;
using polatory::point_cloud::distance_filter;
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

  // Read points and normals.
  std::vector<Eigen::Vector3d> points;
  Eigen::VectorXd values;
  std::tie(points, values) = read_points_2d_and_values(in_file);

  // Remove very close points.
  distance_filter filter(points, filter_distance);
  points = filter.filter_points(points);
  values = filter.filter_values(values);

  // Define model.
  biharmonic rbf({ 1.0, 0.0 });
  interpolant interpolant(rbf, 2, 0);

  // Fit.
  if (fit_incrementally) {
    interpolant.fit_incrementally(points, values, fitting_accuracy);
  } else {
    interpolant.fit(points, values, fitting_accuracy);
  }
  std::cout << "Number of RBF centers: " << interpolant.centers().size() << std::endl;

  // Generate isosurface.
  for (size_t i = 0; i < points.size(); i++) {
    points[i](2) = values(i);
  }
  auto mesh_bbox = bbox3d::from_points(points);
  mesh_bbox = bbox3d(mesh_bbox.min() - 0.1 * mesh_bbox.size(), mesh_bbox.max() + 0.1 * mesh_bbox.size());
  polatory::isosurface::isosurface isosurf(mesh_bbox, mesh_resolution);
  rbf_field_function_25d field_f(interpolant);

  isosurf.generate_from_seed_points(points, field_f);

  export_obj(out_obj_file, isosurf);

  return 0;
}
