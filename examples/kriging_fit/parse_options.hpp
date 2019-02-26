// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <exception>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <polatory/polatory.hpp>

struct options {
  std::string in_file;
  double psill;
  double range;
  double nugget;
  polatory::rbf::weight_function weight_fn;
};

inline
options parse_options(int argc, const char *argv[]) {
  namespace po = boost::program_options;

  options opts;
  int weights;

  po::options_description opts_desc("");
  opts_desc.add_options()
    ("in", po::value<std::string>(&opts.in_file)->required(),
     "Input file (output file of kriging_variogram)")
    ("psill", po::value<double>(&opts.psill)->required(),
     "Initial value of the partial sill")
    ("range", po::value<double>(&opts.range)->required(),
     "Initial value of the range")
    ("nugget", po::value<double>(&opts.nugget)->default_value(0),
     "Initial value of the nugget")
    ("weights", po::value<int>(&weights)->default_value(1),
     "Weight function for least squares fitting, one of\n"
       "  0: N_j\n"
       "  1: N_j / h_j^2\n"
       "  2: N_j / (\\gamma(h_j))^2\n"
       "  3: 1\n"
       "  4: 1 / h_j^2\n"
       "  5: 1 / (\\gamma(h_j))^2");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, opts_desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);
    po::notify(vm);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl
              << "Usage: " << argv[0] << " [OPTION]..." << std::endl
              << opts_desc;
    throw;
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
    throw polatory::common::invalid_argument("0 <= weight <= 5");
  }

  return opts;
}
