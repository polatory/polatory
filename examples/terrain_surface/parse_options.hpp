#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/common.hpp"
#include "../common/model_options.hpp"

struct options {
  std::string in_file;
  double min_distance;
  model_options model_opts;
  double absolute_tolerance;
  int max_iter;
  bool reduce;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::string mesh_file;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;

  auto model_opts_desc = make_model_options_description(opts.model_opts);

  po::options_description general_opts_desc("General", 80, 50);
  general_opts_desc.add_options()                                                           //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),                      //
       "Input file in CSV format:\n  X,Y,Z")                                                //
      ("min-dist", po::value(&opts.min_distance)->default_value(1e-10)->value_name("VAL"),  //
       "Minimum separation distance of input points")                                       //
      ("tol", po::value(&opts.absolute_tolerance)->required()->value_name("VAL"),           //
       "Absolute tolerance of the fitting")                                                 //
      ("max-iter", po::value(&opts.max_iter)->default_value(100)->value_name("N"),          //
       "Maximum number of iterations")                                                      //
      ("reduce", po::bool_switch(&opts.reduce),                                             //
       "Try to reduce the number of RBF centers (incremental fitting)")                     //
      ("mesh-bbox",
       po::value(&opts.mesh_bbox)
           ->multitoken()
           ->required()
           ->value_name("XMIN YMIN ZMIN XMAX YMAX ZMAX"),                                     //
       "Output mesh bounding box")                                                            //
      ("mesh-res", po::value(&opts.mesh_resolution)->required()->value_name("VAL"),           //
       "Output mesh resolution")                                                              //
      ("mesh-out", po::value(&opts.mesh_file)->multitoken()->required()->value_name("FILE"),  //
       "Output mesh file in OBJ format");

  po::options_description opts_desc;
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
