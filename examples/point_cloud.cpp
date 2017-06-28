// Copyright (c) 2016, GSI and The Polatory Authors.

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "common/vector_view.hpp"
#include "interpolation.hpp"
#include "isosurface/export_obj.hpp"
#include "isosurface/isosurface.hpp"
#include "isosurface/rbf_field_function.hpp"
#include "point_cloud/scattered_data_generator.hpp"
#include "point_cloud/distance_filter.hpp"
#include "rbf/linear_variogram.hpp"

#include "read_table.hpp"

using polatory::common::make_view;
using polatory::interpolation::rbf_evaluator;
using polatory::interpolation::rbf_incremental_fitter;
using polatory::isosurface::export_obj;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::scattered_data_generator;
using polatory::point_cloud::distance_filter;
using polatory::rbf::linear_variogram;

int main(int argc, char *argv[])
{
   if (argc < 4) return 1;
   std::string in_file(argv[1]);
   std::string out_scattered_data_file(argv[2]);
   std::string out_obj_file(argv[3]);

   std::vector<Eigen::Vector3d> cloud_points;
   std::vector<Eigen::Vector3d> cloud_normals;
   std::tie(cloud_points, cloud_normals) = read_points_and_normals(in_file);

   // Generate scattered data.
   scattered_data_generator scatter_gen(cloud_points, cloud_normals, 1e-4, 5e-4);
   auto points = scatter_gen.scattered_points();
   auto values = scatter_gen.scattered_values();

   // Remove very close points.
   distance_filter filter(points, 1e-11);
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
   linear_variogram rbf({ 1.0, 0.0 });
   int poly_degree = 0;

   // Fit.
   std::vector<size_t> point_indices;
   Eigen::VectorXd weights;

   rbf_incremental_fitter fitter(rbf, poly_degree, points);
   std::tie(point_indices, weights) = fitter.fit(values, 5e-5);

   std::cout << "Number of RBF centers: " << weights.size() << std::endl;

   // Generate isosurface.
   Eigen::Vector3d box_min(-0.1, -0.1, -0.1);
   Eigen::Vector3d box_max(0.1, 0.1, 0.1);

   rbf_evaluator<> eval(rbf, poly_degree, make_view(points, point_indices));
   eval.set_weights(weights);
   rbf_field_function field_f(eval);

   double resolution = 5e-4;
   polatory::isosurface::isosurface isosurf(box_min, box_max, resolution);

   std::vector<Eigen::Vector3d> seed_points(cloud_points.begin(), cloud_points.begin() + 10);
   isosurf.generate_from_seed_points(seed_points, field_f);

   export_obj(out_obj_file, isosurf);

   return 0;
}
