#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <string>

#include "../common/model_options.hpp"

struct options {
  std::string in_file;
  int dim;
  double min_distance;
  model_options model_opts;
  double absolute_tolerance;
  int max_iter;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;

  po::options_description general_opts_desc("General options");
  general_opts_desc.add_options()                                                            //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),                       //
       "Input file in CSV format:\n  X[,Y[,Z]],VAL,SET_ID")                                  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),                         //
       "Dimension of input points")                                                          //
      ("min-dist", po::value(&opts.min_distance)->default_value(1e-10)->value_name("DIST"),  //
       "Minimum separation distance of input points")                                        //
      ("tol", po::value(&opts.absolute_tolerance)->required()->value_name("TOL"),            //
       "Absolute tolerance of fitting")                                                      //
      ("max-iter", po::value(&opts.max_iter)->default_value(100)->value_name("N"),           //
       "Maximum number of iterations");

  auto model_opts_desc = make_model_options_description(opts.model_opts);

  po::options_description opts_desc(80, 50);
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

  return opts;
}
