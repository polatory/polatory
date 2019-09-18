// Copyright (c) 2016, GSI and The Polatory Authors.

#include <exception>
#include <iostream>
#include <tuple>

#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::common::concatenate_cols;
using polatory::common::take_cols;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::geometry::vectors3d;
using polatory::interpolant;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::model;
using polatory::point_cloud::distance_filter;
using polatory::point_cloud::normal_estimator;
using polatory::point_cloud::sdf_data_generator;
using polatory::rbf::biharmonic3d;
using polatory::read_table;
using polatory::tabled;
using polatory::write_table;

int main(int argc, const char *argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load points (x,y,z).
    tabled table = read_table(opts.in_file);
    points3d terrain_points = take_cols(table, 0, 1, 2);

    // Estimate normals.
    vectors3d terrain_normals = normal_estimator(terrain_points)
      .estimate_with_knn(20, opts.min_plane_factor)
      .orient_by_outward_vector({ 0, 0, 1 });

    // Generate SDF (signed distance function) data.
    sdf_data_generator sdf_data(terrain_points, terrain_normals, opts.min_sdf_distance, opts.max_sdf_distance, opts.sdf_multiplication);
    points3d points = sdf_data.sdf_points();
    valuesd values = sdf_data.sdf_values();

    // Remove very close points.
    std::tie(points, values) = distance_filter(points, opts.min_distance)
      .filtered(points, values);

    // Output SDF data (optional).
    if (!opts.sdf_data_file.empty()) {
      write_table(opts.sdf_data_file, concatenate_cols(points, values));
    }

    // Define model.
    model model(biharmonic3d({ 1.0 }), opts.poly_dimension, opts.poly_degree);
    model.set_nugget(opts.smooth);
    interpolant interpolant(model);

    // Fit.
    if (opts.incremental_fit) {
      interpolant.fit_incrementally(points, values, opts.absolute_tolerance);
    } else {
      interpolant.fit(points, values, opts.absolute_tolerance);
    }
    std::cout << "Number of RBF centers: " << interpolant.centers().rows() << std::endl;

    // Generate isosurface.
    isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    rbf_field_function field_fn(interpolant);

    isosurf.generate_from_seed_points(terrain_points, field_fn)
      .export_obj(opts.mesh_file);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
