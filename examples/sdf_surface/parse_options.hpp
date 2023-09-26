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
  std::string in_file;
  std::string grad_in_file;
  double min_distance;
  std::string rbf_name;
  std::vector<double> rbf_params;
  polatory::geometry::linear_transformation3d aniso;
  double nugget;
  int poly_degree;
  double absolute_tolerance;
  double grad_absolute_tolerance;
  int max_iter;
  bool ineq;
  bool reduce;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::string mesh_file;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<std::string> rbf_vec;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()                                               //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),  //
       "Input file in CSV format:\n  X,Y,Z,VAL[,LOWER,UPPER]")          //
      ("grad-in",
       po::value(&opts.grad_in_file)->default_value("")->value_name("FILE"),                   //
       "Gradient data input file in CSV format:\n  X,Y,Z,DX,DY,DZ")                            //
      ("min-dist", po::value(&opts.min_distance)->default_value(1e-10)->value_name("VAL"),     //
       "Minimum separation distance of input points")                                          //
      ("rbf", po::value(&rbf_vec)->multitoken()->required()->value_name("..."), rbf_cov_list)  //
      ("aniso",
       po::value(&opts.aniso)
           ->multitoken()
           ->default_value(polatory::geometry::linear_transformation3d::Identity(),
                           "1. 0. 0. 0. 1. 0. 0. 0. 1.")
           ->value_name("A11 A12 A13 A21 A22 A23 A31 A32 A33"),                              //
       "Elements of the anisotropy matrix")                                                  //
      ("nugget", po::value(&opts.nugget)->default_value(0, "0.")->value_name("VAL"),         //
       "Nugget of the model")                                                                //
      ("deg", po::value(&opts.poly_degree)->default_value(0)->value_name("-1|0|1|2"),        //
       "Degree of the drift polynomial")                                                     //
      ("tol", po::value(&opts.absolute_tolerance)->required()->value_name("VAL"),            //
       "Absolute tolerance of the fitting")                                                  //
      ("grad-tol", po::value(&opts.grad_absolute_tolerance)->required()->value_name("VAL"),  //
       "Gradient data absolute tolerance of the fitting")                                    //
      ("max-iter", po::value(&opts.max_iter)->default_value(32)->value_name("N"),            //
       "Maximum number of iterations")                                                       //
      ("ineq", po::bool_switch(&opts.ineq),                                                  //
       "Use inequality constraints")                                                         //
      ("reduce", po::bool_switch(&opts.reduce),                                              //
       "Try to reduce the number of RBF centers (incremental fitting)")                      //
      ("mesh-bbox",
       po::value(&opts.mesh_bbox)
           ->multitoken()
           ->required()
           ->value_name("XMIN YMIN ZMIN XMAX YMAX ZMAX"),                            //
       "Output mesh bounding box")                                                   //
      ("mesh-res", po::value(&opts.mesh_resolution)->required()->value_name("VAL"),  //
       "Output mesh resolution")                                                     //
      ("mesh-out", po::value(&opts.mesh_file)->required()->value_name("FILE"),       //
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

  opts.rbf_name = rbf_vec.at(0);
  for (std::size_t i = 1; i < rbf_vec.size(); i++) {
    opts.rbf_params.push_back(boost::lexical_cast<double>(rbf_vec.at(i)));
  }

  return opts;
}
