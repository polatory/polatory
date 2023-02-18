#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <polatory/polatory.hpp>
#include <iostream>
#include <cstring>

struct options {
  std::string in_file; // x, y, z  point cloud
  int knn;                      // search the most nearest knn points
  double p_outside_x; // an arbitrary point outside the surface
  double p_outside_y; 
  double p_outside_z;
  double min_plane_factor;
  std::string out_file; // x, y, z, nx, ny, nz  the point cloud with normal
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
      "Input file in CSV format:\n  X,Y,Z")
      ("knn", po::value(&opts.knn)->default_value(20)->value_name("VAL"),
      "Number of points in k-NN search")
      ("p_outside_x", po::value(&opts.p_outside_x)->required()->value_name("VAL"),
      "The X VAL of p_outside")
      ("p_outside_y", po::value(&opts.p_outside_y)->required()->value_name("VAL"),
       "The Y VAL of p_outside")
      ("p_outside_z", po::value(&opts.p_outside_z)->required()->value_name("VAL"), 
      "The Z VAL of p_outside")(
      "min-plane-factor", po::value(&opts.min_plane_factor)->default_value(1.8)->value_name("VAL"),
      "Threshold of acceptance for estimated normal vectors")
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
      "Output file in CSV format:\n  X,Y,Z,VAL");

  po::variables_map vm;

  try {
    po::store(po::parse_command_line(
                  argc, argv, opts_desc,
                  po::command_line_style::unix_style ^ po::command_line_style::allow_short),
              vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl
              << "Usage: " << argv[0] << " [OPTION]..." << std::endl
              << opts_desc;
    throw;
  }

  return opts;
}