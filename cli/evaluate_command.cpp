#include <boost/program_options.hpp>
#include <polatory/polatory.hpp>
#include <string>
#include <vector>

#include "commands.hpp"

using polatory::interpolant;
using polatory::matrixd;
using polatory::read_table;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::geometry::pointsNd;

namespace {

struct options {
  std::string interpolant_file;
  std::string points_file;
  int dim{};
  bool grads{};
  std::string out_file;
};

template <int Dim>
void run_impl(const options& opts) {
  using Interpolant = interpolant<Dim>;
  using Points = pointsNd<Dim>;

  auto inter = Interpolant::load(opts.interpolant_file);

  matrixd table = read_table(opts.points_file);
  Points points = table(Eigen::all, Eigen::seqN(0, Dim));

  auto n = points.rows();
  if (opts.grads) {
    auto values = inter.evaluate(points, points);
    write_table(opts.out_file,
                concatenate_cols(points, values.head(n),
                                 values.tail(Dim * n).template reshaped<Eigen::RowMajor>(n, Dim)));
  } else {
    auto values = inter.evaluate(points);
    write_table(opts.out_file, concatenate_cols(points, values));
  }
}

}  // namespace

void evaluate_command::run(const std::vector<std::string>& args,
                           const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.interpolant_file)->required()->value_name("FILE"),
       "Input interpolant file")  //
      ("points", po::value(&opts.points_file)->value_name("FILE"),
       "Input evaluation points file in CSV format:\n  X[,Y[,Z]]")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("grads", po::bool_switch(&opts.grads),
       "Evaluate gradients as well as values")  //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
       "Output file in CSV format:\n  X[,Y[,Z]],VAL[,DX[,DY[,DZ]]]")  //
      ;

  if (global_opts.help) {
    std::cout << std::format("Usage: polatory {} [OPTIONS]\n", kName) << opts_desc;
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
  } catch (const po::error& e) {
    std::cout << e.what() << '\n'
              << std::format("Usage: polatory {} [OPTIONS]\n", kName) << opts_desc;
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
      throw std::runtime_error(std::format("Unsupported dimension: {}.", opts.dim));
  }
}
