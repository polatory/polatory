// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <polatory/polatory.hpp>

struct options {
  std::string in_file;
  double min_distance;
  double psill;
  double range;
  double nugget;
  int poly_dimension;
  int poly_degree;
  double absolute_tolerance;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::vector<std::pair<double, std::string>> mesh_values_files;
};

inline
options parse_options(int argc, const char *argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<double> bbox_vec;
  std::vector<double> mesh_vals_vec;
  std::vector<std::string> mesh_files_vec;

  po::options_description opts_desc("");
  opts_desc.add_options()
    ("in", po::value<std::string>(&opts.in_file)->required(),
     "Input file (x,y,z,value format)")
    ("min-dist", po::value<double>(&opts.min_distance)->default_value(1e-10),
     "Minimum distance for preserving close points")
    ("psill", po::value<double>(&opts.psill)->required(),
     "Partial sill of the variogram")
    ("range", po::value<double>(&opts.range)->required(),
     "Range of the variogram")
    ("nugget", po::value<double>(&opts.nugget)->default_value(0),
     "Nugget of the variogram")
    ("dim", po::value<int>(&opts.poly_dimension)->default_value(3),
     "Dimension of the drift polynomial")
    ("deg", po::value<int>(&opts.poly_degree)->default_value(0),
     "Degree of the drift polynomial")
    ("tol", po::value<double>(&opts.absolute_tolerance)->required(),
     "Absolute tolerance of fitting")
    ("mesh-bbox", po::value<std::vector<double>>(&bbox_vec)->multitoken()->required(),
     "Output mesh bbox: xmin ymin zmin xmax ymax zmax")
    ("mesh-res", po::value<double>(&opts.mesh_resolution)->required(),
     "Output mesh resolution")
    ("mesh-isoval", po::value<std::vector<double>>(&mesh_vals_vec)->multitoken()->required(),
     "Output mesh isovalues: value1 [value2 [...]]")
    ("mesh-out", po::value<std::vector<std::string>>(&mesh_files_vec)->multitoken()->required(),
     "Output mesh files (OBJ format): file1 [file2 [...]]");

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

  opts.mesh_bbox = polatory::geometry::bbox3d(
    { bbox_vec[0], bbox_vec[1], bbox_vec[2] },
    { bbox_vec[3], bbox_vec[4], bbox_vec[5] }
  );

  for (size_t i = 0; i < mesh_vals_vec.size(); i++) {
    opts.mesh_values_files.emplace_back(mesh_vals_vec[i], mesh_files_vec[i]);
  }

  return opts;
}
