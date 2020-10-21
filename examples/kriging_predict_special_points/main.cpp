// Copyright (c) 2016, GSI and The Polatory Authors.

#include <exception>
#include <iostream>
#include <tuple>

#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::common::take_cols;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolant;
using polatory::model;
using polatory::point_cloud::distance_filter;
using polatory::read_table;
using polatory::tabled;
using polatory::write_table;

int main(int argc, const char *argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load points (x,y,z) and values (value).
    tabled table = read_table(opts.in_file);
    points3d points = take_cols(table, 0, 1, 2);
    valuesd values = table.col(3);

    // Remove very close points.
    std::tie(points, values) = distance_filter(points, opts.min_distance)
      .filtered(points, values);

    // Define the model.
    auto rbf = make_rbf(opts.rbf_name, opts.rbf_params);
    rbf->set_anisotropy(opts.aniso);
    model model(*rbf, opts.poly_dimension, opts.poly_degree);
    model.set_nugget(opts.nugget);

    // Fit.
    interpolant interpolant(model);
    interpolant.fit(points, values, opts.absolute_tolerance);

    // Get prediction values.
    tabled prediction_points_table = read_table(opts.prediction_points_file);
    points3d prediction_points = take_cols(prediction_points_table, 0, 1, 2);
    auto prediction_values = interpolant.evaluate(prediction_points);

    // Output.
    write_table(opts.out_file, prediction_values);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
