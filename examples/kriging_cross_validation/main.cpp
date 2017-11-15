// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>

#include <Eigen/Core>

#include <polatory/interpolant.hpp>
#include <polatory/io/read_table.hpp>
#include <polatory/kriging/cross_validation.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/rbf/cov_quasi_spherical9.hpp>

#include "parse_options.hpp"

using polatory::interpolant;
using polatory::io::read_points_and_values;
using polatory::kriging::k_fold_cross_validation;
using polatory::point_cloud::distance_filter;
using polatory::rbf::cov_quasi_spherical9;

int main(int argc, const char *argv[]) {
  auto opts = parse_options(argc, argv);

  std::vector<Eigen::Vector3d> points;
  Eigen::VectorXd values;
  std::tie(points, values) = read_points_and_values(opts.in_file);

  // Remove very close points.
  distance_filter filter(points, opts.filter_distance);
  points = filter.filter_points(points);
  values = filter.filter_values(values);

  // Define model.
  cov_quasi_spherical9 rbf({ opts.psill, opts.range, opts.nugget });
  auto residuals = k_fold_cross_validation(rbf, opts.poly_dimension, opts.poly_degree,
                                           points, values, opts.absolute_tolerance, opts.k);

  std::cout << "Estimated mean absolute error: " << std::endl
            << std::setw(12) << residuals.lpNorm<1>() / points.size() << std::endl
            << "Estimated root mean square error: " << std::endl
            << std::setw(12) << std::sqrt(residuals.squaredNorm() / points.size()) << std::endl;

  return 0;
}
