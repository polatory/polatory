// Copyright (c) 2016, GSI and The Polatory Authors.

#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "polatory/geometry/bbox3d.hpp"
#include "polatory/interpolant.hpp"
#include "polatory/io/read_table.hpp"
#include "polatory/isosurface/export_obj.hpp"
#include "polatory/isosurface/isosurface.hpp"
#include "polatory/isosurface/rbf_field_function.hpp"
#include "polatory/rbf/cov_spherical.hpp"

using polatory::geometry::bbox3d;
using polatory::interpolant;
using polatory::io::read_points_and_values;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::rbf::cov_spherical;

int main(int argc, char *argv[]) {
  if (argc < 3) return 1;
  std::string in_file(argv[1]);
  std::string out_dir(argv[2]);

  std::vector<Eigen::Vector3d> points;
  Eigen::VectorXd values;
  std::tie(points, values) = read_points_and_values(in_file);

  // Define model.
  cov_spherical rbf({ 0.0181493, 0.678264, 0.00383142 });
  interpolant interpolant(rbf, 3, 0);

  // Fit.
  interpolant.fit(points, values, 0.00001);

  // Generate isosurface of some values.
  bbox3d mesh_bbox(
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(1.0, 1.0, 1.0)
  );

  std::vector<std::pair<double, std::string>> isovalue_names{
    { 0.2, "0.2.obj" },
    { 0.4, "0.4.obj" },
    { 0.6, "0.6.obj" },
    { 0.8, "0.8.obj" }
  };

  auto resolution = 1e-2;
  isosurface isosurf(mesh_bbox, resolution);
  rbf_field_function field_f(interpolant);

  for (auto isovalue_name : isovalue_names) {
    isosurf.generate(field_f, isovalue_name.first);
    export_obj(out_dir + "\\" + isovalue_name.second, isosurf);
  }

  return 0;
}
