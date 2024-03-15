#include <exception>
#include <iomanip>
#include <iostream>
#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::model;
using polatory::kriging::empirical_variogram;
using polatory::kriging::variogram_fitting;

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);
    // Load the empirical variogram.
    empirical_variogram emp_variog(opts.in_file);

    // Define the model.
    auto rbf = make_rbf<3>(opts.rbf_name, opts.rbf_params);
    model<3> model(rbf, -1);
    model.set_nugget(opts.nugget);

    // Fit model parameters.
    variogram_fitting fit(emp_variog, model, opts.weight_fn);

    auto params = fit.parameters();
    std::cout << "Fitted parameters:" << std::endl
              << std::setw(12) << "nugget" << std::setw(12) << "psill" << std::setw(12) << "range"
              << std::endl
              << std::setw(12) << params[0] << std::setw(12) << params[1] << std::setw(12)
              << params[2] << std::endl;

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
