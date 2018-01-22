// Copyright (c) 2016, GSI and The Polatory Authors.

#include <exception>
#include <iomanip>
#include <iostream>

#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::kriging::empirical_variogram;
using polatory::kriging::variogram_fitting;
using polatory::rbf::cov_quasi_spherical9;

int main(int argc, const char *argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    empirical_variogram emp_variog(opts.in_file);

    cov_quasi_spherical9 variog({ opts.psill, opts.range, opts.nugget });
    variogram_fitting fit(emp_variog, variog, opts.weight_fn);

    auto params = fit.parameters();
    std::cout << "Fitted parameters:" << std::endl
              << std::setw(12) << "psill" << std::setw(12) << "range" << std::setw(12) << "nugget" << std::endl
              << std::setw(12) << params[0] << std::setw(12) << params[1] << std::setw(12) << params[2] << std::endl;

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
