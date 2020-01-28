// Copyright (c) 2016, GSI and The Polatory Authors.

#include <exception>
#include <iostream>

#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::common::concatenate_cols;
using polatory::common::take_cols;
using polatory::geometry::points3d;
using polatory::geometry::vectors3d;
using polatory::point_cloud::normal_estimator;
using polatory::read_table;
using polatory::tabled;
using polatory::write_table;

int main(int argc, const char *argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load points (x,y,z).
    tabled table = read_table(opts.in_file);
    points3d points = take_cols(table, 0, 1, 2);

    // Estimate normals.
    vectors3d normals = normal_estimator(points)
      .estimate_with_knn(opts.knn, opts.min_plane_factor)
      .orient_by_outward_vector({ 0, 0, 1 });

    // Output points with normals.
    write_table(opts.out_file, concatenate_cols(points, normals));

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
