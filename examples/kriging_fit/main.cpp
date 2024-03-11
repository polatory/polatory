#include <exception>
#include <iomanip>
#include <iostream>
#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::model;
using polatory::kriging::empirical_variogram;
using polatory::kriging::variogram_fitting;
using polatory::rbf::cov_cauchy3;
using polatory::rbf::cov_cauchy5;
using polatory::rbf::cov_cauchy7;
using polatory::rbf::cov_cauchy9;
using polatory::rbf::cov_exponential;
using polatory::rbf::cov_spheroidal3;
using polatory::rbf::cov_spheroidal5;
using polatory::rbf::cov_spheroidal7;
using polatory::rbf::cov_spheroidal9;

template <class Rbf>
void main_impl(const options& opts) {
  // Load the empirical variogram.
  empirical_variogram emp_variog(opts.in_file);

  // Define the model.
  Rbf rbf(opts.rbf_params);
  model model(rbf, -1);
  model.set_nugget(opts.nugget);

  // Fit model parameters.
  variogram_fitting fit(emp_variog, model, opts.weight_fn);

  auto params = fit.parameters();
  std::cout << "Fitted parameters:" << std::endl
            << std::setw(12) << "nugget" << std::setw(12) << "psill" << std::setw(12) << "range"
            << std::endl
            << std::setw(12) << params[0] << std::setw(12) << params[1] << std::setw(12)
            << params[2] << std::endl;
}

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    if (opts.rbf_name == "ca3") {
      main_impl<cov_cauchy3<3>>(opts);
    } else if (opts.rbf_name == "ca5") {
      main_impl<cov_cauchy5<3>>(opts);
    } else if (opts.rbf_name == "ca7") {
      main_impl<cov_cauchy7<3>>(opts);
    } else if (opts.rbf_name == "ca9") {
      main_impl<cov_cauchy9<3>>(opts);
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
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
