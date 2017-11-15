// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

struct options {
  std::string in_file;
  double filter_distance;
  double psill;
  double range;
  double nugget;
  int poly_dimension;
  int poly_degree;
  double absolute_tolerance;
  int k;
};

options parse_options(int argc, const char *argv[]) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("");
  opts_desc.add_options()
    ("in", po::value<std::string>(&opts.in_file)->required(),
     "input file")
    ("filter-dist", po::value<double>(&opts.filter_distance)->default_value(1e-10),
     "filter distance threshold")
    ("psill", po::value<double>(&opts.psill)->required(),
     "partial sill of the variogram")
    ("range", po::value<double>(&opts.range)->required(),
     "range of the variogram")
    ("nugget", po::value<double>(&opts.nugget)->default_value(0),
     "nugget of the variogram")
    ("dim", po::value<int>(&opts.poly_dimension)->default_value(3),
     "dimension of the drift polynomial")
    ("deg", po::value<int>(&opts.poly_degree)->default_value(0),
     "degree of the drift polynomial")
    ("tol", po::value<double>(&opts.absolute_tolerance)->required(),
     "absolute tolerance of fitting")
    ("k", po::value<int>(&opts.k)->default_value(5),
     "k of k-fold cross validation");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, opts_desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);
    po::notify(vm);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl
              << "Usage: " << argv[0] << " [OPTION]..." << std::endl
              << opts_desc;
    std::exit(1);
  }

  return opts;
}
