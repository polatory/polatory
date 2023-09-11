#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <polatory/types.hpp>
#include <string>

struct options {
  std::string in_file;
  std::vector<polatory::index_t> ks;
  double min_plane_factor;
  std::string out_file;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()                                               //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),  //
       "Input file in CSV format:\n  X,Y,Z")                            //
      ("k",
       po::value(&opts.ks)
           ->multitoken()
           ->default_value(std::vector<polatory::index_t>{10, 30, 100, 300},
                           "10 30 100 300")
           ->value_name("K [K2...]"),                            //
       "Number of points for kNN search for normal estimation")  //
      ("min-plane-factor",
       po::value(&opts.min_plane_factor)->default_value(1.8)->value_name("FACTOR"),  //
       "Threshold of acceptance for estimated normals")                              //
      ("out", po::value(&opts.out_file)->value_name("FILE"),                         //
       "Output file in CSV format:\n  X,Y,Z,VAL");

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
