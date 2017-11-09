// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <Eigen/Core>

#include "polatory/geometry/bbox3d.hpp"

struct options {
  std::string in_file;
  double min_sdf_distance;
  double max_sdf_distance;
  double filter_distance;
  double rho;
  int poly_dimension;
  int poly_degree;
  bool incremental_fit;
  double absolute_tolerance;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::string mesh_file;
  std::string sdf_data_file;
};

options parse_options(int argc, const char *argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<double> bbox_vec;

  po::options_description opts_desc("");
  opts_desc.add_options()
    ("in", po::value<std::string>(&opts.in_file)->required(),
     "input file")
    ("min-sdf-dist", po::value<double>(&opts.min_sdf_distance)->required(),
     "minimum shift distance of off-surface points")
    ("max-sdf-dist", po::value<double>(&opts.max_sdf_distance)->required(),
     "maximum shift distance of off-surface points")
    ("filter-dist", po::value<double>(&opts.filter_distance)->default_value(1e-10),
     "filter distance threshold")
    ("rho", po::value<double>(&opts.rho)->default_value(0),
     "spline smoothing")
    ("dim", po::value<int>(&opts.poly_dimension)->default_value(3),
     "dimension of polynomial")
    ("deg", po::value<int>(&opts.poly_degree)->default_value(0),
     "degree of polynomial")
    ("incremental-fit", po::bool_switch(&opts.incremental_fit),
     "add RBF centers incrementally")
    ("tol", po::value<double>(&opts.absolute_tolerance)->required(),
     "absolute tolerance of fitting")
    ("mesh-bbox", po::value<std::vector<double>>(&bbox_vec)->multitoken()->required(),
     "output mesh bbox: xmin ymin zmin xmax ymax zmax")
    ("mesh-res", po::value<double>(&opts.mesh_resolution)->required(),
     "output mesh resolution")
    ("mesh-file", po::value<std::string>(&opts.mesh_file)->multitoken()->required(),
     "output mesh filename")
    ("sdf-data-file", po::value<std::string>(&opts.sdf_data_file),
     "SDF data output filename");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, opts_desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);
    po::notify(vm);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl
              << "Usage: " << argv[0] << " [OPTION]..." << std::endl
              << opts_desc;
    std::exit(1);
  }

  opts.mesh_bbox = polatory::geometry::bbox3d(
    Eigen::Vector3d(bbox_vec[0], bbox_vec[1], bbox_vec[2]),
    Eigen::Vector3d(bbox_vec[3], bbox_vec[4], bbox_vec[5])
  );

  return opts;
}
