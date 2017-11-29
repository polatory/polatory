// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>

#include <polatory/interpolant.hpp>
#include <polatory/kriging/cross_validation.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/rbf/cov_quasi_spherical9.hpp>
#include <polatory/table.hpp>

#include "parse_options.hpp"

using polatory::common::take_cols;
using polatory::interpolant;
using polatory::kriging::k_fold_cross_validation;
using polatory::point_cloud::distance_filter;
using polatory::rbf::cov_quasi_spherical9;
using polatory::read_table;

int main(int argc, const char *argv[]) {
  auto opts = parse_options(argc, argv);

  auto table = read_table(opts.in_file);
  auto points = take_cols(table, 0, 1, 2);
  auto values = table.col(3);

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
