#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::read_table;
using polatory::tabled;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::geometry::points3d;
using polatory::geometry::vectors3d;
using polatory::point_cloud::normal_estimator;

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load points (x,y,z).
    tabled table = read_table(opts.in_file);
    points3d points = table(Eigen::all, {0, 1, 2});

    // Estimate normals.
    vectors3d normals = normal_estimator(points)
                            .estimate_with_knn(opts.ks, opts.min_plane_factor)
                            .orient_closed_surface(opts.k_orient);

    // Output points with normals.
    write_table(opts.out_file, concatenate_cols(points, normals));

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
