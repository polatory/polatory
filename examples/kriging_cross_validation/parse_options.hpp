// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "../common/common.hpp"

struct options {
  std::string in_file;
  double min_distance;
  std::string rbf_name;
  std::vector<double> rbf_params;
  polatory::geometry::linear_transformation3d aniso;
  double nugget;
  int poly_dimension;
  int poly_degree;
  double absolute_tolerance;
  int k;
};

inline
options parse_options(int argc, const char *argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<std::string> rbf_vec;
  std::vector<double> aniso_vec;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()
    ("in", po::value<std::string>(&opts.in_file)->required()
      ->value_name("FILE"),
     "Input file in CSV format:\n  X,Y,Z,VAL")
    ("min-dist", po::value<double>(&opts.min_distance)->default_value(1e-10)
      ->value_name("VAL"),
     "Minimum separation distance of input points")
    ("rbf", po::value<std::vector<std::string>>(&rbf_vec)->multitoken()->required()
      ->value_name("..."),
     rbf_cov_list)
    ("aniso", po::value<std::vector<double>>(&aniso_vec)->multitoken()->default_value({ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, "1. 0. 0. 0. 1. 0. 0. 0. 1.")
      ->value_name("A11 A12 A13 A21 A22 A23 A31 A32 A33"),
     "Elements of the anisotropy matrix")
    ("nugget", po::value<double>(&opts.nugget)->default_value(0.0, "0.")
      ->value_name("VAL"),
     "Nugget of the model")
    ("dim", po::value<int>(&opts.poly_dimension)->default_value(3)
      ->value_name("1|2|3"),
     "Dimension of the drift polynomial")
    ("deg", po::value<int>(&opts.poly_degree)->default_value(0)
      ->value_name("-1|0|1|2"),
     "Degree of the drift polynomial")
    ("tol", po::value<double>(&opts.absolute_tolerance)->required()
      ->value_name("VAL"),
     "Absolute tolerance of fitting")
    ("k", po::value<int>(&opts.k)->default_value(5)
      ->value_name("K"),
     "Number of folds in the cross-validation");

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

  opts.rbf_name = rbf_vec[0];
  for (size_t i = 1; i < rbf_vec.size(); i++) {
    opts.rbf_params.push_back(std::stod(rbf_vec[i]));
  }

  if (aniso_vec.size() != 9)
    std::runtime_error("9 arguments must be given to --aniso.");

  opts.aniso = polatory::geometry::linear_transformation3d(aniso_vec.data());

  return opts;
}
