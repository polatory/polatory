// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>
#include <Eigen/Core>

#include "polatory/geometry/bbox3d.hpp"

struct options {
  std::string in_file;
  double filter_distance;
  double psill;
  double range;
  double nugget;
  int poly_dimension;
  int poly_degree;
  double absolute_tolerance;
  polatory::geometry::bbox3d mesh_bbox;
  double mesh_resolution;
  std::vector<std::pair<double, std::string>> mesh_values_names;
};

options parse_options(int argc, const char *argv[]) {
  namespace po = boost::program_options;

  po::options_description opts_desc("");
  opts_desc.add_options()
    ("in", po::value<std::string>()->required(), "input file")
    ("filter-dist", po::value<double>()->default_value(1e-10), "filter distance threshold")
    ("psill", po::value<double>()->required(), "partial sill of the variogram")
    ("range", po::value<double>()->required(), "range of the variogram")
    ("nugget", po::value<double>()->default_value(0), "nugget of the variogram")
    ("dim", po::value<int>()->default_value(3), "dimension of drift")
    ("deg", po::value<int>()->default_value(0), "degree of drift")
    ("tol", po::value<double>()->required(), "absolute tolerance of fitting")
    ("out-mesh-bbox", po::value<std::vector<double>>()->multitoken()->required(), "output mesh bbox: xmin ymin zmin xmax ymax zmax")
    ("out-mesh-resol", po::value<double>()->required(), "output mesh resolution")
    ("out-mesh-vals", po::value<std::vector<double>>()->multitoken()->required(), "output mesh isovalues: val1 [val2 [...]]")
    ("out-mesh-files", po::value<std::vector<std::string>>()->multitoken()->required(), "output mesh filenames: file1 [file2 [...]]");

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

  options opts;
  opts.in_file = vm["in"].as<std::string>();
  opts.filter_distance = vm["filter-dist"].as<double>();
  opts.psill = vm["psill"].as<double>();
  opts.range = vm["range"].as<double>();
  opts.nugget = vm["nugget"].as<double>();
  opts.poly_dimension = vm["dim"].as<int>();
  opts.poly_degree = vm["deg"].as<int>();
  opts.absolute_tolerance = vm["tol"].as<double>();

  auto bbox_vec = vm["out-mesh-bbox"].as<std::vector<double>>();
  opts.mesh_bbox = polatory::geometry::bbox3d(
    Eigen::Vector3d(bbox_vec[0], bbox_vec[1], bbox_vec[2]),
    Eigen::Vector3d(bbox_vec[3], bbox_vec[4], bbox_vec[5])
  );
  opts.mesh_resolution = vm["out-mesh-resol"].as<double>();

  auto mesh_vals_vec = vm["out-mesh-vals"].as<std::vector<double>>();
  auto mesh_files_vec = vm["out-mesh-files"].as<std::vector<std::string>>();
  for (size_t i = 0; i < mesh_vals_vec.size(); i++) {
    opts.mesh_values_names.push_back(std::make_pair(mesh_vals_vec[i], mesh_files_vec[i]));
  }

  return opts;
}
