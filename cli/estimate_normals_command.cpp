#include <boost/program_options.hpp>
#include <iostream>
#include <polatory/polatory.hpp>
#include <string>
#include <vector>

#include "commands.hpp"

using polatory::index_t;
using polatory::matrixd;
using polatory::read_table;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::geometry::vector3d;
using polatory::point_cloud::normal_estimator;

namespace {

struct options {
  std::string in_file;
  std::vector<index_t> ks;
  std::vector<double> radii;
  double threshold{};
  std::vector<double> point{};
  std::vector<double> direction{};
  int closed{};
  std::string out_file;
};

void run_impl(const options& opts) {
  matrixd table = read_table(opts.in_file);
  points3d points = table(Eigen::all, {0, 1, 2});

  normal_estimator estimator(points);

  if (!opts.ks.empty()) {
    estimator.estimate_with_knn(opts.ks);
  } else if (!opts.radii.empty()) {
    estimator.estimate_with_radius(opts.radii);
  } else {
    estimator.estimate_with_knn(std::vector<index_t>{10, 30, 100, 300});
  }

  estimator.filter_by_plane_factor(opts.threshold);

  if (opts.point.size() == 3) {
    estimator.orient_toward_point(point3d{opts.point[0], opts.point[1], opts.point[2]});
  } else if (opts.direction.size() == 3) {
    estimator.orient_toward_direction(
        vector3d{opts.direction[0], opts.direction[1], opts.direction[2]});
  } else if (opts.closed) {
    estimator.orient_closed_surface(100);
  }

  const auto& normals = estimator.normals();

  write_table(opts.out_file, concatenate_cols(points, normals));
}

}  // namespace

void estimate_normals_command::run(const std::vector<std::string>& args,
                                   const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X,Y,Z")  //
      ("k", po::value(&opts.ks)->multitoken()->value_name("K ..."),
       "Number of points for k-NN search during normal estimation")  //
      ("radius", po::value(&opts.radii)->multitoken()->value_name("RADIUS ..."),
       "Radius for radius search during normal estimation")  //
      ("threshold", po::value(&opts.threshold)->default_value(1.8, "1.8")->value_name("THRES"),
       "Threshold for plane factor filtering, set to 1.0 to disable filtering")  //
      ("point", po::value(&opts.point)->multitoken()->value_name("X Y Z"),
       "Orient normals toward the point")  //
      ("direction", po::value(&opts.direction)->multitoken()->value_name("X Y Z"),
       "Orient normals toward the direction")  //
      ("closed", po::value(&opts.closed)->value_name("K"),
       "Orient normals for closed surface(s) with specified number of points for k-NN search")  //
      ("out", po::value(&opts.out_file)->value_name("FILE"),
       "Output file in CSV format:\n  X,Y,Z,NX,NY,NZ")  //
      ;

  if (global_opts.help) {
    std::cout << std::format("usage: polatory {} [OPTIONS]\n", kName) << opts_desc;
    return;
  }

  po::variables_map vm;
  try {
    po::store(po::command_line_parser{args}
                  .options(opts_desc)
                  .style(po::command_line_style::unix_style ^ po::command_line_style::allow_short)
                  .run(),
              vm);
    po::notify(vm);
  } catch (const po::error&) {
    std::cout << std::format("usage: polatory {} [OPTIONS]\n", kName) << opts_desc;
    throw;
  }

  auto num_estimation_opts = vm.count("k") + vm.count("radius");
  if (num_estimation_opts == 0) {
    opts.ks = {10, 30, 100, 300};
  } else if (num_estimation_opts > 1) {
    throw std::runtime_error("only either --k or --radius can be specified");
  }

  auto num_orientation_opts = vm.count("point") + vm.count("direction") + vm.count("closed");
  if (num_orientation_opts == 0) {
    opts.closed = 100;
  } else if (num_orientation_opts > 1) {
    throw std::runtime_error("only one of --point, --direction, or --closed can be specified");
  }

  if (vm.count("point") && opts.point.size() != 3) {
    throw std::runtime_error("--point takes exactly 3 values");
  }

  if (vm.count("direction") && opts.direction.size() != 3) {
    throw std::runtime_error("--direction takes exactly 3 values");
  }

  run_impl(opts);
}
