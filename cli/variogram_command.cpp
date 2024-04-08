#include <Eigen/Core>
#include <boost/program_options.hpp>
#include <format>
#include <numbers>
#include <polatory/kriging.hpp>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "commands.hpp"

using polatory::matrixd;
using polatory::read_table;
using polatory::vectord;
using polatory::geometry::pointsNd;
using polatory::kriging::detrend;
using polatory::kriging::variogram;
using polatory::kriging::variogram_calculator;

namespace {

struct options {
  std::string in_file;
  int dim{};
  int detrend{};
  double lag_distance{};
  int num_lags{};
  double lag_tolerance{};
  double angle_tolerance{};
  bool aniso{};
  std::string out_file;
};

template <int Dim>
void run_impl(const options& opts) {
  using Points = pointsNd<Dim>;
  using VariogramCalculator = variogram_calculator<Dim>;

  matrixd table = read_table(opts.in_file);
  Points points = table(Eigen::all, Eigen::seqN(0, Dim));
  vectord values = table.col(Dim);

  if (opts.detrend >= 0) {
    values = detrend(points, values, opts.detrend);
  }

  VariogramCalculator calc(opts.lag_distance, opts.num_lags);
  calc.set_lag_tolerance(opts.lag_tolerance);
  calc.set_angle_tolerance(opts.angle_tolerance);
  if (opts.aniso) {
    calc.set_directions(VariogramCalculator::kAnisotropicDirections);
  }
  auto variog_set = calc.calculate(points, values);

  variog_set.save(opts.out_file);
}

}  // namespace

void variogram_command::run(const std::vector<std::string>& args,
                            const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X[,Y[,Z]],VAL")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("detrend", po::value(&opts.detrend)->default_value(-1, "-1")->value_name("-1|0|1|2"),
       "Detrend polynomial of specified degree")  //
      ("lag-dist", po::value(&opts.lag_distance)->required()->value_name("DIST"),
       "Lag distance")  //
      ("num-lags", po::value(&opts.num_lags)->default_value(15)->value_name("N"),
       "Number of lags")  //
      ("lag-tol",
       po::value(&opts.lag_tolerance)
           ->default_value(variogram_calculator<1>::kAutomaticLagTolerance, "AUTO")
           ->value_name("TOL"),
       "Lag tolerance")  //
      ("angle-tol",
       po::value(&opts.angle_tolerance)
           ->default_value(variogram_calculator<1>::kAutomaticAngleTolerance, "AUTO")
           ->value_name("TOL"),
       "Angle tolerance in degrees")  //
      ("aniso", po::bool_switch(&opts.aniso),
       "Use anisotropic directions")  //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
       "Output variogram file")  //
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

  if (opts.angle_tolerance != variogram_calculator<1>::kAutomaticAngleTolerance) {
    auto deg = std::numbers::pi / 180.0;
    opts.angle_tolerance *= deg;
  }

  switch (opts.dim) {
    case 1:
      run_impl<1>(opts);
      break;
    case 2:
      run_impl<2>(opts);
      break;
    case 3:
      run_impl<3>(opts);
      break;
    default:
      throw std::runtime_error(std::format("unsupported dimension: {}", opts.dim));
  }
}
