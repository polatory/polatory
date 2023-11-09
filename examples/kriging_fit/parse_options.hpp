#pragma once

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/common.hpp"

struct options {
  std::string in_file;
  std::string rbf_name;
  std::vector<double> rbf_params;
  double nugget;
  polatory::kriging::weight_function weight_fn;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<std::string> rbf_vec;
  int weights;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()                                                               //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),                  //
       "Input file (an output file from kriging_variogram)")                            //
      ("rbf", po::value(&rbf_vec)->multitoken()->required()->value_name("..."),         //
       cov_list)                                                                        //
      ("nugget", po::value(&opts.nugget)->default_value(0.0, "0.")->value_name("VAL"),  //
       "Initial value of the nugget")                                                   //
      ("weights", po::value(&weights)->default_value(1)->value_name("0-5"),             //
       R"(Weight function for least squares fitting, one of
  0: N_j
  1: N_j / h_j^2
  2: N_j / (\\gamma(h_j))^2
  3: 1
  4: 1 / h_j^2
  5: 1 / (\\gamma(h_j))^2)");

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

  opts.rbf_name = rbf_vec.at(0);
  for (std::size_t i = 1; i < rbf_vec.size(); i++) {
    opts.rbf_params.push_back(boost::lexical_cast<double>(rbf_vec.at(i)));
  }

  switch (weights) {
    case 0:
      opts.weight_fn = polatory::kriging::weight_functions::n_pairs;
      break;
    case 1:
      opts.weight_fn = polatory::kriging::weight_functions::n_pairs_over_distance_squared;
      break;
    case 2:
      opts.weight_fn = polatory::kriging::weight_functions::n_pairs_over_model_gamma_squared;
      break;
    case 3:
      opts.weight_fn = polatory::kriging::weight_functions::one;
      break;
    case 4:
      opts.weight_fn = polatory::kriging::weight_functions::one_over_distance_squared;
      break;
    case 5:
      opts.weight_fn = polatory::kriging::weight_functions::one_over_model_gamma_squared;
      break;
    default:
      throw std::runtime_error("weight must be within the range of 0 to 5.");
  }

  return opts;
}
