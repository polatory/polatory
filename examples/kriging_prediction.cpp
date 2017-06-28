// Copyright (c) 2016, GSI and The Polatory Authors.

#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "geometry/bbox3.hpp"
#include "interpolation.hpp"
#include "isosurface/export_obj.hpp"
#include "isosurface/isosurface.hpp"
#include "isosurface/rbf_field_function.hpp"
#include "rbf/spherical_variogram.hpp"

#include "read_table.hpp"

using polatory::geometry::bbox3d;
using polatory::interpolation::rbf_evaluator;
using polatory::interpolation::rbf_fitter;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::rbf::spherical_variogram;

int main(int argc, char *argv[])
{
   if (argc < 3) return 1;
   std::string in_file(argv[1]);
   std::string out_dir(argv[2]);

   std::vector<Eigen::Vector3d> points;
   Eigen::VectorXd values;
   std::tie(points, values) = read_points_and_values(in_file);

   // Define model.
   spherical_variogram rbf({ 0.0181493, 0.678264, 0.00383142 });
   int poly_degree = 0;

   // Fit.
   rbf_fitter fitter(rbf, poly_degree, points);
   auto weights = fitter.fit(values, 0.00001);

   // Generate isosurface of some values.
   Eigen::Vector3d box_min(0.0, 0.0, 0.0);
   Eigen::Vector3d box_max(1.0, 1.0, 1.0);

   Eigen::Vector3d eval_box_min(-0.1, -0.1, -0.1);
   Eigen::Vector3d eval_box_max(1.1, 1.1, 1.1);

   rbf_evaluator<> eval(rbf, poly_degree, points, bbox3d(eval_box_min, eval_box_max));
   eval.set_weights(weights);
   rbf_field_function field_f(eval);

   std::vector<std::pair<double, std::string>> isovalue_names{
      { 0.2, "0.2.obj" },
      { 0.4, "0.4.obj" },
      { 0.6, "0.6.obj" },
      { 0.8, "0.8.obj" }
   };
   
   auto resolution = 1e-2;
   polatory::isosurface::isosurface isosurf(box_min, box_max, resolution);

   for (auto isovalue_name : isovalue_names) {
      isosurf.generate(field_f, isovalue_name.first);
      export_obj(out_dir + "\\" + isovalue_name.second, isosurf);
   }

   return 0;
}
