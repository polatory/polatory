#pragma once

#include <boost/program_options.hpp>
#include <iostream>
#include <limits>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>

#include "../common/bbox.hpp"

struct Options {
  std::string in;
  std::string mesh_in;
  double min_distance{};
  double tolerance{};
  int max_iter{};
  double accuracy{};
  bool reduce{};
  polatory::geometry::Bbox3 mesh_bbox;
  double mesh_resolution{};
  std::string mesh_out;
};

inline Options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  Options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in)->required()->value_name("FILE"),
       "The points to offset the mesh to in CSV format:\n  X,Y,Z")  //
      ("mesh-in", po::value(&opts.mesh_in)->required()->value_name("FILE"),
       "The mesh to offset in OBJ format")  //
      ("tol", po::value(&opts.tolerance)->required()->value_name("TOL"),
       "Absolute fitting tolerance")  //
      ("max-iter", po::value(&opts.max_iter)->default_value(100)->value_name("N"),
       "Maximum number of iterations")  //
      ("acc",
       po::value(&opts.accuracy)
           ->default_value(std::numeric_limits<double>::infinity(), "ANY")
           ->value_name("ACC"),
       "Absolute evaluation accuracy")  //
      ("reduce", po::bool_switch(&opts.reduce),
       "Try to reduce the number of RBF centers (incremental fitting)")  //
      ("mesh-bbox",
       po::value(&opts.mesh_bbox)
           ->multitoken()
           ->required()
           ->value_name("X_MIN Y_MIN Z_MIN X_MAX Y_MAX Z_MAX"),
       "Output mesh bounding box")  //
      ("mesh-res", po::value(&opts.mesh_resolution)->required()->value_name("RES"),
       "Output mesh resolution")  //
      ("mesh-out", po::value(&opts.mesh_out)->required()->value_name("FILE"),
       "Output mesh file in OBJ format")  //
      ;

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
