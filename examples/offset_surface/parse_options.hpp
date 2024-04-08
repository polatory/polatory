#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>

#include "../common/bbox.hpp"

struct options {
  std::string in;
  std::string mesh_in;
  double min_distance;
  double absolute_tolerance;
  int max_iter;
  bool reduce;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::string mesh_out;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()                                                                    //
      ("in", po::value(&opts.in)->required()->value_name("FILE"),                            //
       "The points to offset the mesh to in CSV format:\n  X,Y,Z")                           //
      ("mesh-in", po::value(&opts.mesh_in)->required()->value_name("FILE"),                  //
       "The mesh to offset in OBJ format")                                                   //
      ("min-dist", po::value(&opts.min_distance)->default_value(1e-10)->value_name("DIST"),  //
       "Minimum separation distance of input points")                                        //
      ("tol", po::value(&opts.absolute_tolerance)->required()->value_name("TOL"),            //
       "Absolute tolerance of the fitting")                                                  //
      ("max-iter", po::value(&opts.max_iter)->default_value(100)->value_name("N"),           //
       "Maximum number of iterations")                                                       //
      ("reduce", po::bool_switch(&opts.reduce),                                              //
       "Try to reduce the number of RBF centers (incremental fitting)")                      //
      ("mesh-bbox",
       po::value(&opts.mesh_bbox)
           ->multitoken()
           ->required()
           ->value_name("X_MIN Y_MIN Z_MIN X_MAX Y_MAX Z_MAX"),                      //
       "Output mesh bounding box")                                                   //
      ("mesh-res", po::value(&opts.mesh_resolution)->required()->value_name("RES"),  //
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
  } catch (const po::error&) {
    std::cout << "usage: offset_surface [OPTIONS]\n" << opts_desc;
    throw;
  }

  return opts;
}
