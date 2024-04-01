#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <string>
#include <utility>
#include <vector>

#include "../common/common.hpp"
#include "../common/model_options.hpp"

struct options {
  std::string in_file;
  std::string grad_in_file;
  double min_distance;
  model_options model_opts;
  double absolute_tolerance;
  double grad_absolute_tolerance;
  int max_iter;
  bool ineq;
  bool reduce;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::vector<std::pair<double, std::string>> mesh_values_files;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<double> mesh_vals_vec;
  std::vector<std::string> mesh_files_vec;

  auto model_opts_desc = make_model_options_description(opts.model_opts);

  po::options_description general_opts_desc("General", 80, 50);
  general_opts_desc.add_options()                                              //
      ("in", po::value(&opts.in_file)->default_value("")->value_name("FILE"),  //
       "Input file in CSV format:\n  X,Y,Z,VAL[,LOWER,UPPER]")                 //
      ("grad-in",
       po::value(&opts.grad_in_file)->default_value("")->value_name("FILE"),                //
       "Gradient data input file in CSV format:\n  X,Y,Z,DX,DY,DZ")                         //
      ("min-dist", po::value(&opts.min_distance)->default_value(1e-10)->value_name("VAL"),  //
       "Minimum separation distance of input points")                                       //
      ("tol", po::value(&opts.absolute_tolerance)->required()->value_name("VAL"),           //
       "Absolute tolerance of the fitting")                                                 //
      ("grad-tol",
       po::value(&opts.grad_absolute_tolerance)->default_value(1.0, "1.")->value_name("VAL"),  //
       "Gradient data absolute tolerance of the fitting")                                      //
      ("max-iter", po::value(&opts.max_iter)->default_value(100)->value_name("N"),             //
       "Maximum number of iterations")                                                         //
      ("ineq", po::bool_switch(&opts.ineq),                                                    //
       "Use inequality constraints")                                                           //
      ("reduce", po::bool_switch(&opts.reduce),                                                //
       "Try to reduce the number of RBF centers (incremental fitting)")                        //
      ("mesh-bbox",
       po::value(&opts.mesh_bbox)
           ->multitoken()
           ->required()
           ->value_name("XMIN YMIN ZMIN XMAX YMAX ZMAX"),                                         //
       "Output mesh bounding box")                                                                //
      ("mesh-res", po::value(&opts.mesh_resolution)->required()->value_name("VAL"),               //
       "Output mesh resolution")                                                                  //
      ("mesh-isoval", po::value(&mesh_vals_vec)->multitoken()->required()->value_name("VAL..."),  //
       "Output mesh isovalues")                                                                   //
      ("mesh-out", po::value(&mesh_files_vec)->multitoken()->required()->value_name("FILE..."),   //
       "Output mesh files in OBJ format");

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

  for (std::size_t i = 0; i < mesh_vals_vec.size(); i++) {
    opts.mesh_values_files.emplace_back(mesh_vals_vec.at(i), mesh_files_vec.at(i));
  }

  return opts;
}
