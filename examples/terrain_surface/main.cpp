#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <tuple>

#include "../common/make_model.hpp"
#include "parse_options.hpp"

using polatory::interpolant;
using polatory::model;
using polatory::read_table;
using polatory::tabled;
using polatory::vectord;
using polatory::common::concatenate_cols;
using polatory::geometry::points2d;
using polatory::geometry::points3d;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function_25d;
using polatory::point_cloud::distance_filter;

void main_impl(model<2>&& model, const options& opts) {
  // Load points (x,y,0) and values (z).
  tabled table = read_table(opts.in_file);
  points2d points = table(Eigen::all, {0, 1});
  vectord values = table.col(2);

  // Remove very close points.
  std::tie(points, values) = distance_filter(points, opts.min_distance)(points, values);

  // Fit.
  interpolant<2> interpolant(model);
  if (opts.reduce) {
    interpolant.fit_incrementally(points, values, opts.absolute_tolerance, opts.max_iter);
  } else {
    interpolant.fit(points, values, opts.absolute_tolerance, opts.max_iter);
  }

  // Generate the isosurface.
  isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
  rbf_field_function_25d field_fn(interpolant);

  points3d points_3d = concatenate_cols(points, values);
  isosurf.generate_from_seed_points(points_3d, field_fn).export_obj(opts.mesh_file);
}

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);
    main_impl(make_model<2>(opts.model_opts), opts);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
