// Copyright (c) 2016, GSI and The Polatory Authors.

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "polatory/geometry/bbox3d.hpp"
#include "polatory/interpolant.hpp"
#include "polatory/isosurface/export_obj.hpp"
#include "polatory/isosurface/isosurface.hpp"
#include "polatory/isosurface/rbf_field_function.hpp"
#include "polatory/point_cloud/scattered_data_generator.hpp"
#include "polatory/point_cloud/distance_filter.hpp"
#include "polatory/rbf/biharmonic.hpp"

#include "read_table.hpp"

using polatory::geometry::bbox3d;
using polatory::interpolant;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::scattered_data_generator;
using polatory::point_cloud::distance_filter;
using polatory::rbf::biharmonic;

int main(int argc, char *argv[]) {
  if (argc < 15) return 1;
  std::string in_file(argv[1]);

  auto filter_distance = std::stod(argv[2]);
  auto min_normal_distance = std::stod(argv[3]);
  auto max_normal_distance = std::stod(argv[4]);
  auto fitting_accuracy = std::stod(argv[5]);

  bbox3d mesh_bbox(
    Eigen::Vector3d(std::stod(argv[6]), std::stod(argv[7]), std::stod(argv[8])),
    Eigen::Vector3d(std::stod(argv[9]), std::stod(argv[10]), std::stod(argv[11]))
  );
  auto mesh_resolution = std::stod(argv[12]);

  std::string out_scattered_data_file(argv[13]);
  std::string out_obj_file(argv[14]);

  // Read points and normals.
  std::vector<Eigen::Vector3d> cloud_points;
  std::vector<Eigen::Vector3d> cloud_normals;
  std::tie(cloud_points, cloud_normals) = read_points_and_normals(in_file);

  // Generate scattered data.
  scattered_data_generator scatter_gen(cloud_points, cloud_normals, min_normal_distance, max_normal_distance);
  auto points = scatter_gen.scattered_points();
  auto values = scatter_gen.scattered_values();

  // Remove very close points.
  distance_filter filter(points, filter_distance);
  points = filter.filtered_points();
  values = filter.filter_values(values);

  // Output scattered data (optional).
  std::ofstream ofs(out_scattered_data_file);
  if (!ofs) return 1;

  for (size_t i = 0; i < points.size(); i++) {
    const auto& p = points[i];
    ofs << p(0) << " " << p(1) << " " << p(2) << " " << values(i) << std::endl;
  }
  ofs.close();

  // Define model.
  biharmonic rbf({ 1.0, 0.0 });
  interpolant interpolant(rbf, 3, 0);

  // Fit.
  interpolant.fit_incrementally(points, values, fitting_accuracy);
  std::cout << "Number of RBF centers: " << interpolant.centers().size() << std::endl;

  // Generate isosurface.
  polatory::isosurface::isosurface isosurf(mesh_bbox, mesh_resolution);

  interpolant.set_evaluation_bbox(isosurf.evaluation_bbox());
  rbf_field_function field_f(interpolant);

  auto n_seed_points = interpolant.centers().size() / 10;

  std::vector<Eigen::Vector3d> seed_points(interpolant.centers().begin(),
                                           interpolant.centers().begin() + n_seed_points);
  isosurf.generate_from_seed_points(seed_points, field_f);

  export_obj(out_obj_file, isosurf);

  return 0;
}
