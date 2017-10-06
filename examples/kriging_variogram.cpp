// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "polatory/kriging/empirical_variogram.hpp"
#include "polatory/kriging/variogram_fitting.hpp"
#include "polatory/rbf/spherical_variogram.hpp"

#include "read_table.hpp"

using polatory::kriging::empirical_variogram;
using polatory::kriging::variogram_fitting;
using polatory::kriging::variogram_fitting_weights;
using polatory::rbf::rbf_base;
using polatory::rbf::spherical_variogram;

int main(int argc, char *argv[]) {
  if (argc < 2) return 1;
  std::string in_file(argv[1]);

  std::vector<Eigen::Vector3d> points;
  Eigen::VectorXd values;
  std::tie(points, values) = read_points_and_values(in_file);

  int n_bins = 15;
  double bin_range = 0.1;

  empirical_variogram emp_variog(points, values, bin_range, n_bins);
  const auto bin_variances = emp_variog.bin_variances();
  const auto bin_num_pairs = emp_variog.bin_num_pairs();
  for (int bin = 0; bin < n_bins; bin++) {
    std::cout << bin_variances[bin] << " " << bin_num_pairs[bin] << std::endl;
  }

  spherical_variogram variog({ 0.02, 0.6, 0.0 });
  variogram_fitting fit(emp_variog, &variog, variogram_fitting_weights::cressie);

  auto params = fit.parameters();
  std::cout << "Fitted params: " << params[0] << ", " << params[1] << ", " << params[2] << "\n";

  return 0;
}
