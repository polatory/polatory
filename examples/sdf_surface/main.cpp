#include <exception>
#include <iostream>
#include <optional>
#include <polatory/polatory.hpp>
#include <tuple>

#include "parse_options.hpp"

using polatory::index_t;
using polatory::interpolant;
using polatory::model;
using polatory::read_table;
using polatory::tabled;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::geometry::vectors3d;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::point_cloud::distance_filter;
using polatory::rbf::biharmonic2d;
using polatory::rbf::biharmonic3d;
using polatory::rbf::cov_exponential;
using polatory::rbf::cov_spheroidal3;
using polatory::rbf::cov_spheroidal5;
using polatory::rbf::cov_spheroidal7;
using polatory::rbf::cov_spheroidal9;
using polatory::rbf::inverse_multiquadric1;
using polatory::rbf::multiquadric1;
using polatory::rbf::multiquadric3;
using polatory::rbf::triharmonic3d;

template <class Rbf>
void main_impl(const options& opts) {
  // Load points (x,y,z) and values (value).
  tabled table = read_table(opts.in_file);
  points3d points = table(Eigen::all, {0, 1, 2});
  valuesd values = table.col(3);
  std::optional<valuesd> values_lb;
  std::optional<valuesd> values_ub;
  if (opts.ineq) {
    values_lb = table.col(4);
    values_ub = table.col(5);
  }

  // Load gradient data.
  tabled grad_table(0, 6);
  if (opts.grad_in_file != "") {
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

  // Define the model.
  Rbf rbf(opts.rbf_params);
  rbf.set_anisotropy(opts.aniso);
  model model(rbf, opts.poly_degree);
  model.set_nugget(opts.nugget);

  valuesd rhs(values.size() + 3 * grad_values.rows());
  rhs << values, grad_values.reshaped<Eigen::RowMajor>();

  // Fit.
  interpolant interpolant(model);
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

  // Generate the isosurface.
  auto n_surface_points = (values.array() == 0.0).count();
  points3d surface_points(n_surface_points, 3);
  index_t si{};
  for (index_t i = 0; i < points.rows(); i++) {
    if (values(i) == 0.0) {
      surface_points.row(si++) = points.row(i);
    }
  }

  isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
  rbf_field_function field_fn(interpolant);

  isosurf.generate_from_seed_points(surface_points, field_fn).export_obj(opts.mesh_file);
}

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    if (opts.rbf_name == "bh2") {
      main_impl<biharmonic2d<3>>(opts);
    } else if (opts.rbf_name == "bh3") {
      main_impl<biharmonic3d<3>>(opts);
    } else if (opts.rbf_name == "exp") {
      main_impl<cov_exponential<3>>(opts);
    } else if (opts.rbf_name == "sp3") {
      main_impl<cov_spheroidal3<3>>(opts);
    } else if (opts.rbf_name == "sp5") {
      main_impl<cov_spheroidal5<3>>(opts);
    } else if (opts.rbf_name == "sp7") {
      main_impl<cov_spheroidal7<3>>(opts);
    } else if (opts.rbf_name == "sp9") {
      main_impl<cov_spheroidal9<3>>(opts);
    } else if (opts.rbf_name == "imq1") {
      main_impl<inverse_multiquadric1<3>>(opts);
    } else if (opts.rbf_name == "mq1") {
      main_impl<multiquadric1<3>>(opts);
    } else if (opts.rbf_name == "mq3") {
      main_impl<multiquadric3<3>>(opts);
    } else if (opts.rbf_name == "th3") {
      main_impl<triharmonic3d<3>>(opts);
    } else {
      throw std::runtime_error("Unknown RBF name: " + opts.rbf_name);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
