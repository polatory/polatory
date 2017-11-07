// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "polatory/io/read_table.hpp"
#include "polatory/kriging/empirical_variogram.hpp"
#include "polatory/kriging/variogram_fitting.hpp"
#include "polatory/rbf/cov_quasi_spherical9.hpp"

using polatory::io::read_points_and_values;
using polatory::kriging::empirical_variogram;
using polatory::kriging::variogram_fitting;
using polatory::kriging::variogram_fitting_weights;
using polatory::rbf::rbf_base;
using polatory::rbf::cov_quasi_spherical9;

int main(int argc, char *argv[]) {
  if (argc < 2) return 1;
  std::string in_file(argv[1]);

  std::vector<Eigen::Vector3d> points;
  Eigen::VectorXd values;
  std::tie(points, values) = read_points_and_values(in_file);

  int n_bins = 15;
  double bin_range = 0.1;

  empirical_variogram emp_variog(points, values, bin_range, n_bins);
  const auto bin_distance = emp_variog.bin_distance();
  const auto bin_num_pairs = emp_variog.bin_num_pairs();
  const auto bin_variance = emp_variog.bin_variance();

  std::cout << "np\tdist\tgamma" << std::endl;
  for (int bin = 0; bin < n_bins; bin++) {
    std::cout << bin_num_pairs[bin] << "\t" << bin_distance[bin] << "\t" <<  bin_variance[bin] << std::endl;
  }

  cov_quasi_spherical9 variog({ 0.02, 0.6, 0.0 });
  variogram_fitting fit(emp_variog, &variog, variogram_fitting_weights::cressie);

  auto params = fit.parameters();
  std::cout << "Fitted params: " << params[0] << ", " << params[1] << ", " << params[2] << "\n";

  return 0;
}
