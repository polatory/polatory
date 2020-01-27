// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <exception>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

struct options {
  std::string in_file;
  double min_offset;
  double max_offset;
  double sdf_multiplication;
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
     "Input file in x,y,z,nx,ny,nz format")
    ("min-offset", po::value<double>(&opts.min_offset)->default_value(0.0, "0.0")
      ->value_name("<value>"),
     "Minimum offset distance of off-surface points")
    ("offset", po::value<double>(&opts.max_offset)->required()
      ->value_name("<value>"),
     "Default offset distance of off-surface points, average distance between adjacent points is appropriate")
    ("mult", po::value<double>(&opts.sdf_multiplication)->default_value(2.0, "2.0")
      ->value_name("1.0-3.0"),
     "Multiplication factor of sdf data")
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
