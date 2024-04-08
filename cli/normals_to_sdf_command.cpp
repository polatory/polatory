#include <Eigen/Core>
#include <algorithm>
#include <boost/program_options.hpp>
#include <format>
#include <numeric>
#include <polatory/polatory.hpp>
#include <string>
#include <vector>

#include "commands.hpp"

using polatory::index_t;
using polatory::matrixd;
using polatory::read_table;
using polatory::vectord;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::geometry::points3d;
using polatory::geometry::vectors3d;
using polatory::point_cloud::sdf_data_generator;

namespace {

struct options {
  std::string in_file;
  double min_offset{};
  double max_offset{};
  double mult{};
  std::string out_file;
};

void run_impl(const options& opts) {
  matrixd table = read_table(opts.in_file);

  // Shuffle the points so that the off-surface points will not be clustered.
  std::vector<index_t> indices(table.rows());
  std::iota(indices.begin(), indices.end(), index_t{0});
  std::shuffle(indices.begin(), indices.end(), std::mt19937{});

  points3d surface_points = table(indices, {0, 1, 2});
  vectors3d surface_normals = table(indices, {3, 4, 5});

  sdf_data_generator sdf_data(surface_points, surface_normals, opts.min_offset, opts.max_offset,
                              opts.mult);

  const auto& points = sdf_data.sdf_points();
  const auto& values = sdf_data.sdf_values();

  write_table(opts.out_file, concatenate_cols(points, values));
}

}  // namespace

void normals_to_sdf_command::run(const std::vector<std::string>& args,
                                 const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X,Y,Z,NX,NY,NZ")  //
      ("min-offset", po::value(&opts.min_offset)->default_value(0.0, "0.0")->value_name("OFFSET"),
       "Minimum offset distance of off-surface points")  //
      ("offset", po::value(&opts.max_offset)->required()->value_name("OFFSET"),
       "Default offset distance of off-surface points")  //
      ("mult", po::value(&opts.mult)->default_value(2.0, "2.0")->value_name("1.0 to 3.0"),
       "The size of output data as a multiple the input data")  //
      ("out", po::value(&opts.out_file)->value_name("FILE"),
       "Output file in CSV format:\n  X,Y,Z,VAL")  //
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
              << std::format("Usage: polatory {} [OPTIONS]\n", kName) << std::endl
              << opts_desc;
    throw;
  }

  run_impl(opts);
}
