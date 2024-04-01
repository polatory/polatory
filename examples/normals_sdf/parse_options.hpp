#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <string>

struct options {
  std::string in_file;
  double min_offset;
  double max_offset;
  double sdf_multiplication;
  std::string out_file;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X,Y,Z,NX,NY,NZ")  //
      ("min-offset",
       po::value(&opts.min_offset)->default_value(0.0, "0.0")->value_name("OFFSET"),  //
       "Minimum offset distance of off-surface points")                               //
      ("offset", po::value(&opts.max_offset)->required()->value_name("OFFSET"),       //
       "Default offset distance of off-surface points, the average distance between adjacent "
       "points is a reasonable choice")  //
      ("mult",
       po::value(&opts.sdf_multiplication)->default_value(2.0, "2.0")->value_name("1.0 to 3.0"),  //
       "Multiplication factor of sdf data")                                                       //
      ("out", po::value(&opts.out_file)->value_name("FILE"),                                      //
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
