#include <Eigen/Core>
#include <algorithm>
#include <boost/program_options.hpp>
#include <format>
#include <iostream>
#include <optional>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "../examples/common/make_model.hpp"
#include "../examples/common/model_options.hpp"
#include "commands.hpp"

using polatory::interpolant;
using polatory::matrixd;
using polatory::model;
using polatory::read_table;
using polatory::vectord;
using polatory::geometry::pointsNd;

namespace {

struct options {
  std::string in_file;
  std::string grad_in_file;
  int dim{};
  std::string model_file;
  model_options model_opts;
  std::string initial_interpolant_file;
  double absolute_tolerance{};
  double grad_absolute_tolerance{};
  int max_iter{};
  bool ineq{};
  bool reduce{};
  std::string out_file;
};

template <int Dim>
void run_impl(const options& opts) {
  using Interpolant = interpolant<Dim>;
  using Model = model<Dim>;
  using Points = pointsNd<Dim>;

  matrixd table = read_table(opts.in_file);
  Points points = table(Eigen::all, Eigen::seqN(0, Dim));
  vectord values = table.col(Dim);
  std::optional<vectord> values_lb;
  std::optional<vectord> values_ub;
  if (opts.ineq) {
    values_lb = table.col(Dim + 1);
    values_ub = table.col(Dim + 2);
  }

  Points grad_points;
  vectord grad_values;
  if (!opts.grad_in_file.empty()) {
    matrixd grad_table = read_table(opts.grad_in_file);
    grad_points = grad_table(Eigen::all, Eigen::seqN(0, Dim));
    grad_values = grad_table(Eigen::all, Eigen::seqN(Dim, Dim));
  }

  vectord rhs(values.size() + grad_values.size());
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();

  auto model =
      !opts.model_file.empty() ? Model::load(opts.model_file) : make_model<Dim>(opts.model_opts);

  std::optional<Interpolant> initial;
  if (!opts.initial_interpolant_file.empty()) {
    initial = Interpolant::load(opts.initial_interpolant_file);
  }

  Interpolant inter(std::move(model));
  if (opts.ineq) {
    inter.fit_inequality(points, values, *values_lb, *values_ub, opts.absolute_tolerance,
                         opts.max_iter);
  } else if (opts.reduce) {
    inter.fit_incrementally(points, grad_points, rhs, opts.absolute_tolerance,
                            opts.grad_absolute_tolerance, opts.max_iter);
  } else {
    inter.fit(points, grad_points, rhs, opts.absolute_tolerance, opts.grad_absolute_tolerance,
              opts.max_iter, initial ? &*initial : nullptr);
  }

  inter.save(opts.out_file);
}

}  // namespace

void fit_command::run(const std::vector<std::string>& args, const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X[,Y[,Z]],VAL[,LOWER,UPPER]")  //
      ("grad-in", po::value(&opts.grad_in_file)->value_name("FILE"),
       "Gradient data input file in CSV format:\n  X[,Y[,Z]],DX[,DY[,DZ]]")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("model", po::value(&opts.model_file)->value_name("FILE"),
       "Input model file")  //
      ("initial", po::value(&opts.initial_interpolant_file)->value_name("FILE"),
       "Input interpolant file to be used as the initial solution")  //
      ("tol", po::value(&opts.absolute_tolerance)->required()->value_name("TOL"),
       "Absolute tolerance of the fitting")  //
      ("grad-tol",
       po::value(&opts.grad_absolute_tolerance)
           ->default_value(-1.0, "SAME AS --tol")
           ->value_name("TOL"),
       "Gradient data absolute tolerance of the fitting")  //
      ("max-iter", po::value(&opts.max_iter)->default_value(100)->value_name("N"),
       "Maximum number of iterations")  //
      ("ineq", po::bool_switch(&opts.ineq),
       "Use inequality constraints")  //
      ("reduce", po::bool_switch(&opts.reduce),
       "Try to reduce the number of RBF centers (incremental fitting)")  //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
       "Output interpolant file")  //
      ;

  if (std::find(args.begin(), args.end(), "--model") == args.end()) {
    auto model_opts_desc = make_model_options_description(opts.model_opts);
    opts_desc.add(model_opts_desc);
  }

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

  if (!opts.grad_in_file.empty() && opts.ineq) {
    throw std::runtime_error("--grad-in cannot be used in conjunction with --ineq");
  }

  if (!opts.initial_interpolant_file.empty() && (opts.ineq || opts.reduce)) {
    throw std::runtime_error("--initial cannot be used in conjunction with --ineq or --reduce");
  }

  if (opts.grad_absolute_tolerance == -1.0) {
    opts.grad_absolute_tolerance = opts.absolute_tolerance;
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
