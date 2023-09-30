#include <exception>
#include <iostream>
#include <optional>
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

  // Remove very close points.
  distance_filter filter(points, opts.min_distance);
  std::tie(points, values) = filter(points, values);
  if (opts.ineq) {
    *values_lb = filter(*values_lb);
    *values_ub = filter(*values_ub);
  }

  // Define the model.
  Rbf rbf(opts.rbf_params);
  rbf.set_anisotropy(opts.aniso);
  Model model(rbf, opts.poly_degree);
  model.set_nugget(opts.nugget);

  // Fit.
  interpolant<Model> interpolant(model);
  if (opts.ineq) {
    interpolant.fit_inequality(points, values, *values_lb, *values_ub, opts.absolute_tolerance,
                               opts.max_iter);
  } else {
    interpolant.fit(points, values, opts.absolute_tolerance, opts.max_iter);
  }

  // Generate isosurfaces of given values.
  isosurface isosurf(opts.mesh_bbox, opts.mesh_resolution);
  rbf_field_function<Model> field_fn(interpolant);

  for (const auto& isovalue_name : opts.mesh_values_files) {
    isosurf.generate(field_fn, isovalue_name.first).export_obj(isovalue_name.second);
  }
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
