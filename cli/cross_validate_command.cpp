#include <Eigen/Core>
#include <boost/program_options.hpp>
#include <format>
#include <iomanip>
#include <iostream>
#include <polatory/kriging.hpp>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>

#include "../examples/common/make_model.hpp"
#include "../examples/common/model_options.hpp"
#include "commands.hpp"

using polatory::MatX;
using polatory::Model;
using polatory::read_table;
using polatory::VecX;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::geometry::Points;
using polatory::kriging::cross_validate;

namespace {

struct Options {
  std::string in_file;
  int dim{};
  std::string model_file;
  ModelOptions model_opts;
  double tolerance{};
  int max_iter{};
  double accuracy{};
  std::string out_file;
};

template <int Dim>
void run_impl(const Options& opts) {
  using Model = Model<Dim>;
  using Points = Points<Dim>;

  MatX table = read_table(opts.in_file);
  Points points = table(Eigen::all, Eigen::seqN(0, Dim));
  VecX values = table.col(Dim);
  Eigen::VectorXi set_ids = table.col(Dim + 1).cast<int>();

  auto model =
      !opts.model_file.empty() ? Model::load(opts.model_file) : make_model<Dim>(opts.model_opts);

  auto predictions = cross_validate<Dim>(model, points, values, set_ids, opts.tolerance,
                                         opts.max_iter, opts.accuracy);

  write_table(opts.out_file, concatenate_cols<MatX>(table, predictions));
}

}  // namespace

void CrossValidateCommand::run(const std::vector<std::string>& args,
                               const GlobalOptions& global_opts) {
  namespace po = boost::program_options;

  Options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X[,Y[,Z]],VAL,SET_ID,...")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("model", po::value(&opts.model_file)->value_name("FILE"),
       "Input model file")  //
      ("tol", po::value(&opts.tolerance)->required()->value_name("TOL"),
       "Absolute fitting tolerance")  //
      ("max-iter", po::value(&opts.max_iter)->default_value(100)->value_name("N"),
       "Maximum number of iterations")  //
      ("acc",
       po::value(&opts.accuracy)
           ->default_value(std::numeric_limits<double>::infinity(), "ANY")
           ->value_name("ACC"),
       "Absolute evaluation accuracy")  //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
       "Output file in CSV format:\n  X[,Y[,Z]],VAL,SET_ID,...,PREDICTION")  //
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
