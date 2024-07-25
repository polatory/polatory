#include <Eigen/Core>
#include <boost/program_options.hpp>
#include <format>
#include <polatory/polatory.hpp>
#include <string>
#include <vector>

#include "commands.hpp"

using polatory::matrixd;
using polatory::read_table;
using polatory::write_table;
using polatory::geometry::pointsNd;
using polatory::point_cloud::distance_filter;

namespace {

struct options {
  std::string in_file;
  int dim{};
  double dist{};
  std::string out_file;
};

template <int Dim>
void run_impl(const options& opts) {
  using Points = pointsNd<Dim>;

  matrixd table = read_table(opts.in_file);
  Points points = table(Eigen::all, Eigen::seqN(0, Dim));

  table = distance_filter(points).filter(opts.dist)(table);

  write_table(opts.out_file, table);
}

}  // namespace

void unique_command::run(const std::vector<std::string>& args, const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X[,Y[,Z]]...")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("dist", po::value(&opts.dist)->default_value(0.0, "0.0")->value_name("DIST"),
       "Minimum distance for identifying points")  //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
       "Output file in CSV format:\n  X[,Y[,Z]]...")  //
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
