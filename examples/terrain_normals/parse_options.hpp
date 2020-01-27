// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <exception>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

struct options {
  std::string in_file;
  int knn;
  double min_plane_factor;
  std::string out_file;
};

inline
options parse_options(int argc, const char *argv[]) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()
    ("in", po::value<std::string>(&opts.in_file)->required()
      ->value_name("<file>"),
     "Input file in x,y,z format")
    ("knn", po::value<int>(&opts.knn)->default_value(20)
      ->value_name("<value>"),
     "Number of points in k-NN search")
    ("min-plane-factor", po::value<double>(&opts.min_plane_factor)->default_value(1.8)
      ->value_name("<value>"),
     "Threshold of acceptance for estimated normal vectors")
    ("out", po::value<std::string>(&opts.out_file)
      ->value_name("<file>"),
     "Output file in x,y,z,value format");

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

  return opts;
}
