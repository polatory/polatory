#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "commands.hpp"

using polatory::Index;
using polatory::MatX;
using polatory::read_table;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::geometry::Point3;
using polatory::geometry::Points3;
using polatory::geometry::Vector3;
using polatory::numeric::to_double;
using polatory::point_cloud::NormalEstimator;

namespace {

enum class NormalEstimationMethod { kKNN, kRadius };

enum class OrientationEstimationMethod { kPoint, kDirection, kClosed };

struct Options {
  std::string in_file;
  NormalEstimationMethod normal_method{};
  std::vector<Index> ks;
  std::vector<double> radii;
  double threshold{};
  OrientationEstimationMethod orientation_method{};
  Point3 point;
  Vector3 direction;
  Index k_closed{};
  std::string out_file;
};

void run_impl(const Options& opts) {
  MatX table = read_table(opts.in_file);
  Points3 points = table(Eigen::all, {0, 1, 2});

  NormalEstimator estimator(points);

  switch (opts.normal_method) {
    case NormalEstimationMethod::kKNN:
      estimator.estimate_with_knn(opts.ks);
      break;
    case NormalEstimationMethod::kRadius:
      estimator.estimate_with_radius(opts.radii);
      break;
  }

  estimator.filter_by_plane_factor(opts.threshold);

  switch (opts.orientation_method) {
    case OrientationEstimationMethod::kPoint:
      estimator.orient_toward_point(opts.point);
      break;
    case OrientationEstimationMethod::kDirection:
      estimator.orient_toward_direction(opts.direction);
      break;
    case OrientationEstimationMethod::kClosed:
      estimator.orient_closed_surface(opts.k_closed);
      break;
  }

  const auto& normals = estimator.normals();

  write_table(opts.out_file, concatenate_cols<MatX>(points, normals));
}

}  // namespace

void EstimateNormalsCommand::run(const std::vector<std::string>& args,
                                 const GlobalOptions& global_opts) {
  namespace po = boost::program_options;

  Options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X,Y,Z")  //
      ("k", po::value(&opts.ks)->multitoken()->value_name("K ..."),
       "Use k-NN search with the specified number of points for normal estimation\n"
       "When multiple values are supplied, the best one is selected per point\n"
       "This option with 10 30 100 300 is default")  //
      ("radius", po::value(&opts.radii)->multitoken()->value_name("RADIUS ..."),
       "Use radius search with the specified radius for normal estimation\n"
       "When multiple values are supplied, the best one is selected per point")  //
      ("threshold", po::value(&opts.threshold)->default_value(1.8, "1.8")->value_name("THRES"),
       "Threshold for plane factor filtering, set to 1.0 to disable filtering")  //
      ("point", po::value(&opts.point)->multitoken()->value_name("X Y Z"),
       "Orient normals toward the point")  //
      ("direction", po::value(&opts.direction)->multitoken()->value_name("X Y Z"),
       "Orient normals toward the direction")  //
      ("closed", po::value(&opts.k_closed)->value_name("K"),
       "Orient normals of closed surface(s) using k-NN search with the specified number of points\n"
       "This option with 100 is default")  //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
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
  if (num_estimation_opts > 1) {
    throw std::runtime_error("only either --k or --radius can be specified");
  }
  if (vm.count("k") == 1) {
    opts.normal_method = NormalEstimationMethod::kKNN;
  } else if (vm.count("radius") == 1) {
    opts.normal_method = NormalEstimationMethod::kRadius;
  } else {
    opts.normal_method = NormalEstimationMethod::kKNN;
    opts.ks = {10, 30, 100, 300};
  }

  auto num_orientation_opts = vm.count("point") + vm.count("direction") + vm.count("closed");
  if (num_orientation_opts > 1) {
    throw std::runtime_error("only one of --point, --direction, or --closed can be specified");
  }
  if (vm.count("point") == 1) {
    opts.orientation_method = OrientationEstimationMethod::kPoint;
  } else if (vm.count("direction") == 1) {
    opts.orientation_method = OrientationEstimationMethod::kDirection;
  } else if (vm.count("closed") == 1) {
    opts.orientation_method = OrientationEstimationMethod::kClosed;
  } else {
    opts.orientation_method = OrientationEstimationMethod::kClosed;
    opts.k_closed = 100;
  }

  run_impl(opts);
}

namespace Eigen {

inline void validate(boost::any& v, const std::vector<std::string>& values,
                     polatory::geometry::Vector3*, int) {
  namespace po = boost::program_options;

  if (values.size() != 3) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  v = polatory::geometry::Vector3{to_double(values.at(0)), to_double(values.at(1)),
                                  to_double(values.at(2))};
}

}  // namespace Eigen
