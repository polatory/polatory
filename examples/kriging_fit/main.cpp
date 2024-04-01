#include <exception>
#include <iostream>
#include <polatory/kriging.hpp>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/make_model.hpp"
#include "parse_options.hpp"

using polatory::model;
using polatory::common::load;
using polatory::kriging::variogram;
using polatory::kriging::variogram_fitting;

template <int Dim>
void main_impl(model<Dim>&& model, const options& opts) {
  using Variogram = variogram<Dim>;
  using VariogramFitting = variogram_fitting<Dim>;

  // Load the empirical variogram.
  std::vector<Variogram> variogs;
  load(opts.in_file, variogs);

  // Fit model parameters.
  VariogramFitting fit(variogs, model, opts.weight_fn);

  std::cout << fit.brief_report() << std::endl;

  model = fit.model();
  std::cout << model.description() << std::endl;
}

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);
    switch (opts.dim) {
      case 1:
        main_impl(make_model<1>(opts.model_opts), opts);
        break;
      case 2:
        main_impl(make_model<2>(opts.model_opts), opts);
        break;
      case 3:
        main_impl(make_model<3>(opts.model_opts), opts);
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
