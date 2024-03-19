#include <exception>
#include <iomanip>
#include <iostream>
#include <polatory/polatory.hpp>
#include <utility>

#include "parse_options.hpp"

using polatory::model;
using polatory::kriging::empirical_variogram;
using polatory::kriging::variogram_fitting;
using polatory::rbf::rbf_proxy;

template <int Dim>
void main_impl(rbf_proxy<Dim>&& rbf, const options& opts) {
  using EmpiricalVariogram = empirical_variogram<Dim>;
  using Model = model<Dim>;
  using VariogramFitting = variogram_fitting<Dim>;

  // Load the empirical variogram.
  EmpiricalVariogram emp_variog(opts.in_file);

  // Define the model.
  Model model(std::move(rbf), -1);
  model.set_nugget(opts.nugget);

  // Fit model parameters.
  VariogramFitting fit(emp_variog, model, opts.weight_fn);

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
}

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);
    switch (opts.dim) {
      case 1:
        main_impl(make_rbf<1>(opts.rbf_name, opts.rbf_params), opts);
        break;
      case 2:
        main_impl(make_rbf<2>(opts.rbf_name, opts.rbf_params), opts);
        break;
      case 3:
        main_impl(make_rbf<3>(opts.rbf_name, opts.rbf_params), opts);
        break;
      default:
        throw std::runtime_error("Unsupported dimension: " + std::to_string(opts.dim));
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
