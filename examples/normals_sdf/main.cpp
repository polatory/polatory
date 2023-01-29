#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::read_table;
using polatory::tabled;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::common::take_cols;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::geometry::vectors3d;
using polatory::point_cloud::sdf_data_generator;

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load points (x,y,z) and normals (nx,ny,nz).
    tabled table = read_table(opts.in_file);
    points3d surface_points = take_cols(table, 0, 1, 2);
    vectors3d surface_normals = take_cols(table, 3, 4, 5);

    // Generate SDF (signed distance function) data.
    sdf_data_generator sdf_data(surface_points, surface_normals, opts.min_offset, opts.max_offset,
                                opts.sdf_multiplication);
    points3d points = sdf_data.sdf_points();
    valuesd values = sdf_data.sdf_values();

    // Save the SDF data.
    write_table(opts.out_file, concatenate_cols(points, values));

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
