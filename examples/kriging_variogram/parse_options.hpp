#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <string>

struct options {
  std::string in_file;
  int dim;
  double lag_distance;
  int num_lags;
  bool aniso;
  std::string out_file;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()                                                          //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),             //
       "Input file in CSV format:\n  X[,Y[,Z]],VAL")                               //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),               //
       "Dimension of input points")                                                //
      ("lag-dist", po::value(&opts.lag_distance)->required()->value_name("DIST"),  //
       "Lag distance")                                                             //
      ("num-lags", po::value(&opts.num_lags)->default_value(15)->value_name("N"),  //
       "Number of lags")                                                           //
      ("aniso", po::bool_switch(&opts.aniso),                                      //
       "Use anisotropic directions")                                               //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),           //
       "Output file to be read by kriging_fit");

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

  return opts;
}
