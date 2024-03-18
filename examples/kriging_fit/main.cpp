#include <exception>
#include <iomanip>
#include <iostream>
#include <polatory/polatory.hpp>
#include <utility>

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
    model<3> model(std::move(rbf), -1);
    model.set_nugget(opts.nugget);

    // Fit model parameters.
    variogram_fitting fit(emp_variog, model, opts.weight_fn);

    std::cout << "Fitted parameters:" << std::endl;

    auto names = model.parameter_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
      std::cout << std::setw(12) << names.at(i);
    }
    std::cout << std::endl;

    auto params = fit.parameters();
    for (std::size_t i = 0; i < params.size(); ++i) {
      std::cout << std::setw(12) << params.at(i);
    }
    std::cout << std::endl;

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
