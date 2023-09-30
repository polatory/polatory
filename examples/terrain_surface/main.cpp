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
using polatory::rbf::multiquadric1;

template <class Rbf>
void main_impl(const options& opts) {
  using Model = model<Rbf>;

  // Load points (x,y,0) and values (z).
  tabled table = read_table(opts.in_file);
  points3d points = concatenate_cols(table(Eigen::all, {0, 1}), valuesd::Zero(table.rows()));
  valuesd values = table.col(2);

  // Remove very close points.
  std::tie(points, values) = distance_filter(points, opts.min_distance)(points, values);

  // Define the model.
  Rbf rbf(opts.rbf_params);
  Model model(rbf, 2, opts.poly_degree);
  model.set_nugget(opts.smooth);

  // Fit.
  interpolant<Model> interpolant(model);
  if (opts.reduce) {
    interpolant.fit_incrementally(points, values, opts.absolute_tolerance, opts.max_iter);
  } else {
    interpolant.fit(points, values, opts.absolute_tolerance, opts.max_iter);
  }

  // Generate the isosurface.
  isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
  rbf_field_function_25d<Model> field_fn(interpolant);

  points.col(2) = values;
  isosurf.generate_from_seed_points(points, field_fn).export_obj(opts.mesh_file);
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
    } else if (opts.rbf_name == "mq1") {
      main_impl<multiquadric1<3>>(opts);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
