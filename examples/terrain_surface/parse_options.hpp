// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <polatory/polatory.hpp>

#include "../common/common.hpp"

struct options {
  std::string in_file;
  double min_distance;
  std::string rbf_name;
  std::vector<double> rbf_params;
  double smooth;
  int poly_degree;
  double absolute_tolerance;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::string mesh_file;
};

inline
options parse_options(int argc, const char *argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<std::string> rbf_vec;
  std::vector<double> bbox_vec;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()
    ("in", po::value<std::string>(&opts.in_file)->required()
      ->value_name("FILE"),
     "Input file in CSV format:\n  X,Y,Z")
    ("min-dist", po::value<double>(&opts.min_distance)->default_value(1e-10)
      ->value_name("VAL"),
     "Minimum separation distance of input points")
    ("rbf", po::value<std::vector<std::string>>(&rbf_vec)->multitoken()->required()
      ->value_name("..."),
     rbf_cov_list)
    ("smooth", po::value<double>(&opts.smooth)->default_value(0.0, "0.")
      ->value_name("VAL"),
     "Amount of spline smoothing")
    ("deg", po::value<int>(&opts.poly_degree)->default_value(0)
      ->value_name("-1|0|1|2"),
     "Degree of the polynomial")
    ("tol", po::value<double>(&opts.absolute_tolerance)->required()
      ->value_name("VAL"),
     "Absolute tolerance of the fitting")
    ("mesh-bbox", po::value<std::vector<double>>(&bbox_vec)->multitoken()->required()
      ->value_name("XMIN YMIN ZMIN XMAX YMAX ZMAX"),
     "Output mesh bounding box")
    ("mesh-res", po::value<double>(&opts.mesh_resolution)->required()
      ->value_name("VAL"),
     "Output mesh resolution")
    ("mesh-out", po::value<std::string>(&opts.mesh_file)->multitoken()->required()
      ->value_name("FILE"),
     "Output mesh file in OBJ format");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, opts_desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);
    po::notify(vm);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl
              << "Usage: " << argv[0] << " [OPTION]..." << std::endl
              << opts_desc;
    throw;
  }

  opts.rbf_name = rbf_vec[0];
  for (size_t i = 1; i < rbf_vec.size(); i++) {
    opts.rbf_params.push_back(std::stod(rbf_vec[i]));
  }

  if (bbox_vec.size() != 6)
    std::runtime_error("6 arguments must be given to --mesh-bbox.");

  opts.mesh_bbox = polatory::geometry::bbox3d(
    { bbox_vec[0], bbox_vec[1], bbox_vec[2] },
    { bbox_vec[3], bbox_vec[4], bbox_vec[5] });

  return opts;
}
