#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <optional>
#include <polatory/polatory.hpp>
#include <tuple>
#include <utility>

#include "../common/make_model.hpp"
#include "parse_options.hpp"

using polatory::interpolant;
using polatory::model;
using polatory::read_table;
using polatory::tabled;
using polatory::vectord;
using polatory::geometry::points3d;
using polatory::geometry::vectors3d;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::distance_filter;

void main_impl(model<3>&& model, const options& opts) {
  // Load points (x,y,z) and values (value).
  tabled table(0, 4);
  if (!opts.in_file.empty()) {
    table = read_table(opts.in_file);
  }
  points3d points = table(Eigen::all, {0, 1, 2});
  vectord values = table.col(3);
  std::optional<vectord> values_lb;
  std::optional<vectord> values_ub;
  if (opts.ineq) {
    values_lb = table.col(4);
    values_ub = table.col(5);
  }

  // Load gradient data.
  tabled grad_table(0, 6);
  if (!opts.grad_in_file.empty()) {
    grad_table = read_table(opts.grad_in_file);
  }
  points3d grad_points = grad_table(Eigen::all, {0, 1, 2});
  vectors3d grad_values = grad_table(Eigen::all, {3, 4, 5});

  // Remove very close points.
  distance_filter filter(points, opts.min_distance);
  std::tie(points, values) = filter(points, values);
  if (opts.ineq) {
    *values_lb = filter(*values_lb);
    *values_ub = filter(*values_ub);
  }

  distance_filter grad_filter(grad_points, opts.min_distance);
  std::tie(grad_points, grad_values) = grad_filter(grad_points, grad_values);

  vectord rhs(values.size() + 3 * grad_values.rows());
  rhs << values, grad_values.reshaped<Eigen::RowMajor>();

  // Fit.
  interpolant<3> interpolant(model);
  if (opts.ineq) {
    interpolant.fit_inequality(points, values, *values_lb, *values_ub, opts.absolute_tolerance,
                               opts.max_iter);
  } else if (opts.reduce) {
    interpolant.fit_incrementally(points, grad_points, rhs, opts.absolute_tolerance,
                                  opts.grad_absolute_tolerance, opts.max_iter);
  } else {
    interpolant.fit(points, grad_points, rhs, opts.absolute_tolerance, opts.grad_absolute_tolerance,
                    opts.max_iter);
  }

  // Generate isosurfaces of given values.
  isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
  rbf_field_function field_fn(interpolant);

  for (const auto& isovalue_name : opts.mesh_values_files) {
    isosurf.generate(field_fn, isovalue_name.first).export_obj(isovalue_name.second);
  }
}

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);
    main_impl(make_model<3>(opts.model_opts), opts);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
