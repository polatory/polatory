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

using polatory::matrixd;
using polatory::model;
using polatory::read_table;
using polatory::vectord;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::geometry::pointsNd;
using polatory::kriging::cross_validate;

namespace {

struct options {
  std::string in_file;
  int dim;
  std::string model_file;
  model_options model_opts;
  double absolute_tolerance;
  int max_iter;
  std::string out_file;
};

template <int Dim>
void run_impl(const options& opts) {
  using Model = model<Dim>;
  using Points = pointsNd<Dim>;

  matrixd table = read_table(opts.in_file);
  Points points = table(Eigen::all, Eigen::seqN(0, Dim));
  vectord values = table.col(Dim);
  Eigen::VectorXi set_ids = table.col(Dim + 1).cast<int>();

  auto model =
      !opts.model_file.empty() ? Model::load(opts.model_file) : make_model<Dim>(opts.model_opts);

  auto predictions =
      cross_validate<Dim>(model, points, values, set_ids, opts.absolute_tolerance, opts.max_iter);

  write_table(opts.out_file, concatenate_cols(table, predictions));
}

}  // namespace

void cross_validate_command::run(const std::vector<std::string>& args,
                                 const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X[,Y[,Z]],VAL,SET_ID,...")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("model", po::value(&opts.model_file)->value_name("FILE"),
       "Input model file")  //
      ("tol", po::value(&opts.absolute_tolerance)->required()->value_name("TOL"),
       "Absolute fitting tolerance")  //
      ("max-iter", po::value(&opts.max_iter)->default_value(100)->value_name("N"),
       "Maximum number of iterations")  //
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
