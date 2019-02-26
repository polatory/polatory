// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <polatory/polatory.hpp>

struct options {
  std::string in_file;
  double min_distance;
  double smooth;
  int poly_dimension;
  int poly_degree;
  bool incremental_fit;
  double absolute_tolerance;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::string mesh_file;
};

inline
options parse_options(int argc, const char *argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<double> bbox_vec;

  po::options_description opts_desc("");
  opts_desc.add_options()
    ("in", po::value<std::string>(&opts.in_file)->required(),
     "Input file (x,y,z format)")
    ("min-dist", po::value<double>(&opts.min_distance)->default_value(1e-10),
     "Minimum distance for preserving close points")
    ("smooth", po::value<double>(&opts.smooth)->default_value(0.0),
     "Amount of spline smoothing")
    ("dim", po::value<int>(&opts.poly_dimension)->default_value(2),
     "Dimension of the polynomial")
    ("deg", po::value<int>(&opts.poly_degree)->default_value(0),
     "Degree of the polynomial")
    ("incremental", po::bool_switch(&opts.incremental_fit),
     "Add RBF centers incrementally")
    ("tol", po::value<double>(&opts.absolute_tolerance)->required(),
     "Absolute tolerance of fitting")
    ("mesh-bbox", po::value<std::vector<double>>(&bbox_vec)->multitoken()->required(),
     "Output mesh bbox: xmin ymin zmin xmax ymax zmax")
    ("mesh-res", po::value<double>(&opts.mesh_resolution)->required(),
     "Output mesh resolution")
    ("mesh-out", po::value<std::string>(&opts.mesh_file)->multitoken()->required(),
     "Output mesh file (OBJ format)");

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

  return opts;
}
