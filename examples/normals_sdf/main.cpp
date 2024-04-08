#include <Eigen/Core>
#include <algorithm>
#include <exception>
#include <iostream>
#include <numeric>
#include <polatory/polatory.hpp>
#include <random>
#include <vector>

#include "parse_options.hpp"

using polatory::index_t;
using polatory::matrixd;
using polatory::read_table;
using polatory::vectord;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::geometry::points3d;
using polatory::geometry::vectors3d;
using polatory::point_cloud::sdf_data_generator;

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load points (x,y,z) and normals (nx,ny,nz).
    matrixd table = read_table(opts.in_file);

    // Shuffle the points so that the off-surface points will not be clustered.
    std::vector<index_t> indices(table.rows());
    std::iota(indices.begin(), indices.end(), index_t{0});
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    table = table(indices, Eigen::all).eval();

    points3d surface_points = table(Eigen::all, {0, 1, 2});
    vectors3d surface_normals = table(Eigen::all, {3, 4, 5});

    // Generate SDF (signed distance function) data.
    sdf_data_generator sdf_data(surface_points, surface_normals, opts.min_offset, opts.max_offset,
                                opts.sdf_multiplication);
    points3d points = sdf_data.sdf_points();
    vectord values = sdf_data.sdf_values();

    // Save the SDF data.
    write_table(opts.out_file, concatenate_cols(points, values));

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
