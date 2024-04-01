#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <polatory/kriging.hpp>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/common.hpp"
#include "../common/model_options.hpp"

struct options {
  std::string in_file;
  int dim{};
  model_options model_opts;
  polatory::kriging::weight_function weight_fn{
      polatory::kriging::weight_function::kNumPairsOverDistanceSquared};
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;
  int weights;

  auto model_opts_desc = make_model_options_description(opts.model_opts);

  po::options_description general_opts_desc("General", 80, 50);
  general_opts_desc.add_options()                                            //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),       //
       "Input file produced by kriging_variogram")                           //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),         //
       "Dimension of the spatial domain")                                    //
      ("weights", po::value(&weights)->default_value(1)->value_name("0-5"),  //
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
  gamma: model variogram)");

  po::options_description opts_desc;
  opts_desc.add(general_opts_desc).add(model_opts_desc);

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(
                  argc, argv, opts_desc,
                  po::command_line_style::unix_style ^ po::command_line_style::allow_short),
              vm);
    po::notify(vm);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl
              << "Usage: " << argv[0] << " [OPTION]..." << std::endl
              << opts_desc;
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
      throw std::runtime_error("weight must be within the range of 0 to 5.");
  }

  return opts;
}
