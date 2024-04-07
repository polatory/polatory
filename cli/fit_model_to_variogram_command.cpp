#include <Eigen/Core>
#include <algorithm>
#include <boost/program_options.hpp>
#include <format>
#include <iostream>
#include <limits>
#include <polatory/kriging.hpp>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "../examples/common/make_model.hpp"
#include "../examples/common/model_options.hpp"
#include "commands.hpp"

using polatory::model;
using polatory::kriging::variogram_fitting;
using polatory::kriging::variogram_set;

namespace {

struct options {
  std::string in_file;
  int dim{};
  std::string model_file;
  model_options model_opts;
  polatory::kriging::weight_function weight_fn{
      polatory::kriging::weight_function::kNumPairsOverDistanceSquared};
  int num_trials{};
  std::string out_file;
};

template <int Dim>
void run_impl(const options& opts) {
  using Model = model<Dim>;
  using VariogramFitting = variogram_fitting<Dim>;
  using VariogramSet = variogram_set<Dim>;

  auto variog_set = VariogramSet::load(opts.in_file);

  auto model =
      !opts.model_file.empty() ? Model::load(opts.model_file) : make_model<Dim>(opts.model_opts);

  auto best_model = model;
  auto best_cost = std::numeric_limits<double>::infinity();
  for (auto i = 0; i < opts.num_trials; ++i) {
    VariogramFitting fit(variog_set, model, opts.weight_fn);
    if (fit.final_cost() < best_cost) {
      best_model = fit.model();
      best_cost = fit.final_cost();
    }
  }

  std::cout << best_model.description() << std::endl;

  if (!opts.out_file.empty()) {
    best_model.save(opts.out_file);
  }
}

}  // namespace

void fit_model_to_variogram_command::run(const std::vector<std::string>& args,
                                         const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;
  int weights{};

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input variogram file")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("model", po::value(&opts.model_file)->value_name("FILE"),
       "Input model file")  //
      ("weights", po::value(&weights)->default_value(1)->value_name("0 to 5"),
       R"(Weight function for least squares fitting, one of
  0: N_j
  1: N_j / h_j^2
  2: N_j / (gamma(h_j))^2
  3: 1
  4: 1 / h_j^2
  5: 1 / (gamma(h_j))^2
where
  N_j: number of pairs in the j-th bin
  h_j: representative distance of the j-th bin
  gamma: model variogram)")  //
      ("num-trials", po::value(&opts.num_trials)->default_value(30)->value_name("N"),
       "Number of trials")  //
      ("out", po::value(&opts.out_file)->value_name("FILE"),
       "Output model file")  //
      ;

  if (std::find(args.begin(), args.end(), "--model") == args.end()) {
    auto model_opts_desc = make_model_options_description(opts.model_opts);
    opts_desc.add(model_opts_desc);
  }

  if (global_opts.help) {
    std::cout << std::format("Usage: polatory {} [OPTIONS]\n", kName) << opts_desc;
    return;
  }

  po::variables_map vm;
  try {
    po::store(po::command_line_parser{args}
                  .options(opts_desc)
                  .style(po::command_line_style::unix_style ^ po::command_line_style::allow_short)
                  .run(),
              vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cout << e.what() << '\n'
              << std::format("Usage: polatory {} [OPTIONS]\n", kName) << opts_desc;
    throw;
  }

  switch (weights) {
    case 0:
      opts.weight_fn = polatory::kriging::weight_function::kNumPairs;
      break;
    case 1:
      opts.weight_fn = polatory::kriging::weight_function::kNumPairsOverDistanceSquared;
      break;
    case 2:
      opts.weight_fn = polatory::kriging::weight_function::kNumPairsOverModelGammaSquared;
      break;
    case 3:
      opts.weight_fn = polatory::kriging::weight_function::kOne;
      break;
    case 4:
      opts.weight_fn = polatory::kriging::weight_function::kOneOverDistanceSquared;
      break;
    case 5:
      opts.weight_fn = polatory::kriging::weight_function::kOneOverModelGammaSquared;
      break;
    default:
      throw std::runtime_error("--weight must be 0 to 5.");
  }

  switch (opts.dim) {
    case 1:
      run_impl<1>(opts);
      break;
    case 2:
      run_impl<2>(opts);
      break;
    case 3:
      run_impl<3>(opts);
      break;
    default:
      throw std::runtime_error(std::format("Unsupported dimension: {}.", opts.dim));
  }
}
