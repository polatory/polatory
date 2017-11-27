// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/io/read_table.hpp>
#include <polatory/kriging/cross_validation.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/rbf/cov_quasi_spherical9.hpp>

#include "parse_options.hpp"

using polatory::geometry::points3d;
using polatory::interpolant;
using polatory::io::read_points_and_values;
using polatory::kriging::k_fold_cross_validation;
using polatory::point_cloud::distance_filter;
using polatory::rbf::cov_quasi_spherical9;

int main(int argc, const char *argv[]) {
  auto opts = parse_options(argc, argv);

  points3d points;
  Eigen::VectorXd values;
  std::tie(points, values) = read_points_and_values(opts.in_file);

  // Remove very close points.
  std::tie(points, values) = distance_filter(points, opts.filter_distance)
    .filtered(points, values);

  // Define model.
  cov_quasi_spherical9 rbf({ opts.psill, opts.range, opts.nugget });
  auto residuals = k_fold_cross_validation(rbf, opts.poly_dimension, opts.poly_degree,
                                           points, values, opts.absolute_tolerance, opts.k);

  std::cout << "Estimated mean absolute error: " << std::endl
            << std::setw(12) << residuals.lpNorm<1>() / points.rows() << std::endl
            << "Estimated root mean square error: " << std::endl
            << std::setw(12) << std::sqrt(residuals.squaredNorm() / points.rows()) << std::endl;

  return 0;
}
