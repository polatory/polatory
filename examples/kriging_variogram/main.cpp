// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iomanip>
#include <iostream>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/kriging/variogram_fitting.hpp>
#include <polatory/rbf/cov_quasi_spherical9.hpp>
#include <polatory/table.hpp>

#include "parse_options.hpp"

using polatory::common::take_cols;
using polatory::geometry::points3d;
using polatory::kriging::empirical_variogram;
using polatory::kriging::variogram_fitting;
using polatory::kriging::variogram_fitting_weights;
using polatory::rbf::cov_quasi_spherical9;
using polatory::read_table;

int main(int argc, const char *argv[]) {
  auto opts = parse_options(argc, argv);

  auto table = read_table(opts.in_file);
  auto points = take_cols(table, 0, 1, 2);
  auto values = table.col(3);

  empirical_variogram emp_variog(points, values, opts.bin_width, opts.n_bins);
  const auto bin_distance = emp_variog.bin_distance();
  const auto bin_num_pairs = emp_variog.bin_num_pairs();
  const auto bin_variance = emp_variog.bin_variance();

  std::cout << "Empirical variogram:" << std::endl
            << std::setw(12) << "n_pairs" << std::setw(12) << "distance" << std::setw(12) << "gamma" << std::endl;
  for (int bin = 0; bin < bin_distance.size(); bin++) {
    std::cout << std::setw(12) << bin_num_pairs[bin] << std::setw(12) << bin_distance[bin] << std::setw(12) <<  bin_variance[bin] << std::endl;
  }

  cov_quasi_spherical9 variog({ opts.psill, opts.range, opts.nugget });
  variogram_fitting fit(emp_variog, variog, variogram_fitting_weights::equal);

  auto params = fit.parameters();
  std::cout << "Fitted parameters:" << std::endl
            << std::setw(12) << "psill" << std::setw(12) << "range" << std::setw(12) << "nugget" << std::endl
            << std::setw(12) << params[0] << std::setw(12) << params[1] << std::setw(12) << params[2] << std::endl;

  return 0;
}
