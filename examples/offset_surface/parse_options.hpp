#pragma once

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <optional>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/common.hpp"

struct options {
  std::string in;
  std::string mesh_in;
  double min_distance;
  double absolute_tolerance;
  int max_iter;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::string mesh_out;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<std::string> rbf_vec;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()                                                                   //
      ("in", po::value(&opts.in)->required()->value_name("FILE"),                           //
       "Input file in CSV format:\n  X,Y,Z")                                                //
      ("mesh-in", po::value(&opts.mesh_in)->required()->value_name("FILE"),                 //
       "Reference mesh in OBJ format")                                                      //
      ("min-dist", po::value(&opts.min_distance)->default_value(1e-10)->value_name("VAL"),  //
       "Minimum separation distance of input points")                                       //
      ("tol", po::value(&opts.absolute_tolerance)->required()->value_name("VAL"),           //
       "Absolute tolerance of the fitting")                                                 //
      ("max-iter", po::value(&opts.max_iter)->default_value(32)->value_name("N"),           //
       "Maximum number of iterations")                                                      //
      ("mesh-bbox",
       po::value(&opts.mesh_bbox)
           ->multitoken()
           ->required()
           ->value_name("XMIN YMIN ZMIN XMAX YMAX ZMAX"),                            //
       "Output mesh bounding box")                                                   //
      ("mesh-res", po::value(&opts.mesh_resolution)->required()->value_name("VAL"),  //
       "Output mesh resolution")                                                     //
      ("mesh-out", po::value(&opts.mesh_out)->required()->value_name("FILE"),        //
       "Output mesh file in OBJ format");

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
