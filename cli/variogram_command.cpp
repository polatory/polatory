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

using polatory::MatX;
using polatory::read_table;
using polatory::VecX;
using polatory::geometry::Points;
using polatory::kriging::detrend;
using polatory::kriging::NormalScoreTransformation;
using polatory::kriging::VariogramCalculator;

namespace {

struct Options {
  std::string in_file;
  int dim{};
  int detrend{};
  bool normal_score{};
  double lag_distance{};
  int num_lags{};
  double lag_tolerance{};
  double angle_tolerance{};
  bool aniso{};
  std::string out_file;
};

template <int Dim>
void run_impl(const Options& opts) {
  using Points = Points<Dim>;
  using VariogramCalculator = VariogramCalculator<Dim>;

  MatX table = read_table(opts.in_file);
  Points points = table(Eigen::all, Eigen::seqN(0, Dim));
  VecX values = table.col(Dim);

  if (opts.detrend >= 0) {
    values = detrend(points, values, opts.detrend);
  }

  NormalScoreTransformation nst;
  if (opts.normal_score) {
    values = nst.transform(values);
  }

  VariogramCalculator calc(opts.lag_distance, opts.num_lags);
  calc.set_lag_tolerance(opts.lag_tolerance);
  calc.set_angle_tolerance(opts.angle_tolerance);
  if (opts.aniso) {
    calc.set_directions(VariogramCalculator::kAnisotropicDirections);
  }
  auto variog_set = calc.calculate(points, values);

  if (opts.normal_score) {
    variog_set.back_transform(nst);
  }

  variog_set.save(opts.out_file);
}

}  // namespace

void VariogramCommand::run(const std::vector<std::string>& args, const GlobalOptions& global_opts) {
  namespace po = boost::program_options;

  Options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X[,Y[,Z]],VAL")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("detrend", po::value(&opts.detrend)->default_value(-1, "-1")->value_name("-1|0|1|2"),
       "Detrend polynomial of specified degree")  //
      ("normal-score", po::bool_switch(&opts.normal_score),
       "Perform normal score transformation and back-transform variogram")  //
      ("lag-dist", po::value(&opts.lag_distance)->required()->value_name("DIST"),
       "Lag distance")  //
      ("num-lags", po::value(&opts.num_lags)->default_value(15)->value_name("N"),
       "Number of lags")  //
      ("lag-tol",
       po::value(&opts.lag_tolerance)
           ->default_value(VariogramCalculator<1>::kAutomaticLagTolerance, "AUTO")
           ->value_name("TOL"),
       "Lag tolerance")  //
      ("angle-tol",
       po::value(&opts.angle_tolerance)
           ->default_value(VariogramCalculator<1>::kAutomaticAngleTolerance, "AUTO")
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

  if (opts.angle_tolerance != VariogramCalculator<1>::kAutomaticAngleTolerance) {
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
