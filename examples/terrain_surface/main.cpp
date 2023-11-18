#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <tuple>

#include "parse_options.hpp"

using polatory::interpolant;
using polatory::model;
using polatory::read_table;
using polatory::tabled;
using polatory::common::concatenate_cols;
using polatory::common::valuesd;
using polatory::geometry::points2d;
using polatory::geometry::points3d;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function_25d;
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
  // Load points (x,y,0) and values (z).
  tabled table = read_table(opts.in_file);
  points2d points = table(Eigen::all, {0, 1});
  valuesd values = table.col(2);

  // Remove very close points.
  std::tie(points, values) = distance_filter(points, opts.min_distance)(points, values);

  // Define the model.
  Rbf rbf(opts.rbf_params);
  model model(rbf, opts.poly_degree);
  model.set_nugget(opts.smooth);

  // Fit.
  interpolant interpolant(model);
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

    if (opts.rbf_name == "bh2") {
      main_impl<biharmonic2d<2>>(opts);
    } else if (opts.rbf_name == "bh3") {
      main_impl<biharmonic3d<2>>(opts);
    } else if (opts.rbf_name == "exp") {
      main_impl<cov_exponential<2>>(opts);
    } else if (opts.rbf_name == "sp3") {
      main_impl<cov_spheroidal3<2>>(opts);
    } else if (opts.rbf_name == "sp5") {
      main_impl<cov_spheroidal5<2>>(opts);
    } else if (opts.rbf_name == "sp7") {
      main_impl<cov_spheroidal7<2>>(opts);
    } else if (opts.rbf_name == "sp9") {
      main_impl<cov_spheroidal9<2>>(opts);
    } else if (opts.rbf_name == "imq1") {
      main_impl<inverse_multiquadric1<2>>(opts);
    } else if (opts.rbf_name == "mq1") {
      main_impl<multiquadric1<2>>(opts);
    } else if (opts.rbf_name == "mq3") {
      main_impl<multiquadric3<2>>(opts);
    } else if (opts.rbf_name == "th3") {
      main_impl<triharmonic3d<2>>(opts);
    } else {
      throw std::runtime_error("Unknown RBF name: " + opts.rbf_name);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
