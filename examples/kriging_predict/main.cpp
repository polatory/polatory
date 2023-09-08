#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <tuple>

#include "parse_options.hpp"

using polatory::interpolant;
using polatory::model;
using polatory::read_table;
using polatory::tabled;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::distance_filter;

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load points (x,y,z) and values (value).
    tabled table = read_table(opts.in_file);
    points3d points = table(Eigen::all, {0, 1, 2});
    valuesd values = table.col(3);

    // Remove very close points.
    std::tie(points, values) = distance_filter(points, opts.min_distance)(points, values);

    // Define the model.
    auto rbf = make_rbf(opts.rbf_name, opts.rbf_params);
    rbf->set_anisotropy(opts.aniso);
    model model(*rbf, opts.poly_dimension, opts.poly_degree);
    model.set_nugget(opts.nugget);

    // Fit.
    interpolant interpolant(model);
    interpolant.fit(points, values, opts.absolute_tolerance, opts.max_iter);

    // Generate isosurfaces of given values.
    isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
    rbf_field_function field_fn(interpolant);

    for (const auto& isovalue_name : opts.mesh_values_files) {
      isosurf.generate(field_fn, isovalue_name.first).export_obj(isovalue_name.second);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
