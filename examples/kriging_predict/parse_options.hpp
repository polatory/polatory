#pragma once

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <string>
#include <utility>
#include <vector>

#include "../common/common.hpp"

struct options {
  std::string in_file;
  double min_distance;
  std::string rbf_name;
  std::vector<double> rbf_params;
  polatory::geometry::linear_transformation3d aniso;
  double nugget;
  int poly_dimension;
  int poly_degree;
  double absolute_tolerance;
  int max_iter;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::vector<std::pair<double, std::string>> mesh_values_files;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<std::string> rbf_vec;
  std::vector<double> mesh_vals_vec;
  std::vector<std::string> mesh_files_vec;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()                                                                   //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),                      //
       "Input file in CSV format:\n  X,Y,Z,VAL")                                            //
      ("min-dist", po::value(&opts.min_distance)->default_value(1e-10)->value_name("VAL"),  //
       "Minimum separation distance of input points")                                       //
      ("rbf", po::value(&rbf_vec)->multitoken()->required()->value_name("..."),             //
       rbf_cov_list)                                                                        //
      ("aniso",
       po::value(&opts.aniso)
           ->multitoken()
           ->default_value(polatory::geometry::linear_transformation3d::Identity(),
                           "1. 0. 0. 0. 1. 0. 0. 0. 1.")
           ->value_name("A11 A12 A13 A21 A22 A23 A31 A32 A33"),                        //
       "Elements of the anisotropy matrix")                                            //
      ("nugget", po::value(&opts.nugget)->default_value(0, "0.")->value_name("VAL"),   //
       "Nugget of the model")                                                          //
      ("dim", po::value(&opts.poly_dimension)->default_value(3)->value_name("1|2|3"),  //
       "Dimension of the drift polynomial")                                            //
      ("deg", po::value(&opts.poly_degree)->default_value(0)->value_name("-1|0|1|2"),  //
       "Degree of the drift polynomial")                                               //
      ("tol", po::value(&opts.absolute_tolerance)->required()->value_name("VAL"),      //
       "Absolute tolerance of the fitting")                                            //
      ("max-iter", po::value(&opts.max_iter)->default_value(32)->value_name("N"),      //
       "Maximum number of iterations")                                                 //
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

  for (std::size_t i = 0; i < mesh_vals_vec.size(); i++) {
    opts.mesh_values_files.emplace_back(mesh_vals_vec.at(i), mesh_files_vec.at(i));
  }

  return opts;
}
