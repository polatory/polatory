// Copyright (c) 2016, GSI and The Polatory Authors.

#include <algorithm>
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
using polatory::point_cloud::distance_filter;
using polatory::point_cloud::sdf_data_generator;
using polatory::rbf::biharmonic;
using polatory::read_table;
using polatory::write_table;

int main(int argc, const char *argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Read points and normals.
    auto table = read_table(opts.in_file);
    points3d cloud_points = take_cols(table, 0, 1, 2);
    vectors3d cloud_normals = take_cols(table, 3, 4, 5);

    // Generate SDF data.
    sdf_data_generator sdf_data(cloud_points, cloud_normals, opts.min_sdf_distance, opts.max_sdf_distance, opts.sdf_multiplication);
    points3d points = sdf_data.sdf_points();
    valuesd values = sdf_data.sdf_values();

    // Remove very close points.
    std::tie(points, values) = distance_filter(points, opts.filter_distance)
      .filtered(points, values);

    // Output SDF data (optional).
    if (!opts.sdf_data_file.empty()) {
      write_table(opts.sdf_data_file, concatenate_cols(points, values));
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
    std::cout << "Number of RBF centers: " << interpolant.centers().rows() << std::endl;

    // Generate isosurface.
    polatory::isosurface::isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    rbf_field_function field_fn(interpolant);

    auto n_seed_points = std::max(size_t(cloud_points.rows() / 10), size_t(100));
    points3d seed_points = cloud_points.topRows(n_seed_points);
    isosurf.generate_from_seed_points(seed_points, field_fn)
      .export_obj(opts.mesh_file);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
