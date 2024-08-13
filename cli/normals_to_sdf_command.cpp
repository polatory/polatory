#include <Eigen/Core>
#include <algorithm>
#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <format>
#include <numeric>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "commands.hpp"

using polatory::Index;
using polatory::Mat3;
using polatory::MatX;
using polatory::read_table;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::geometry::Points3;
using polatory::geometry::Vectors3;
using polatory::numeric::to_double;
using polatory::point_cloud::SdfDataGenerator;

namespace {

struct Options {
  std::string in_file;
  double min_offset{};
  double max_offset{};
  double ratio{};
  Mat3 aniso;
  std::string out_file;
};

void run_impl(const Options& opts) {
  MatX table = read_table(opts.in_file);

  Points3 points = table(Eigen::all, {0, 1, 2});
  Vectors3 normals = table(Eigen::all, {3, 4, 5});

  auto n_normals = normals.rows();
  auto n_normals_to_keep = static_cast<Index>(std::round(opts.ratio * n_normals));
  std::vector<Index> indices(n_normals);
  std::iota(indices.begin(), indices.end(), 0);
  // Prevent clustering of off-surface points.
  std::shuffle(indices.begin(), indices.end(), std::mt19937{});
  indices.resize(n_normals - n_normals_to_keep);
  normals(indices, Eigen::all) *= 0.0;

  SdfDataGenerator sdf_data(points, normals, opts.min_offset, opts.max_offset, opts.aniso);

  const auto& sdf_points = sdf_data.sdf_points();
  const auto& sdf_values = sdf_data.sdf_values();

  write_table(opts.out_file, concatenate_cols(sdf_points, sdf_values));
}

}  // namespace

void NormalsToSdfCommand::run(const std::vector<std::string>& args,
                              const GlobalOptions& global_opts) {
  namespace po = boost::program_options;

  Options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input file in CSV format:\n  X,Y,Z,NX,NY,NZ")  //
      ("min-offset", po::value(&opts.min_offset)->default_value(0.0, "0.0")->value_name("OFFSET"),
       "Minimum offset distance of off-surface points")  //
      ("offset", po::value(&opts.max_offset)->required()->value_name("OFFSET"),
       "Default offset distance of off-surface points")  //
      ("aniso",
       po::value(&opts.aniso)
           ->multitoken()
           ->default_value(Mat3::Identity(), "1 0 0 0 1 0 0 0 1")
           ->value_name("A_11 A_12 ... A_33"),
       "Elements of the anisotropy matrix")  //
      ("ratio", po::value(&opts.ratio)->default_value(0.5, "0.5")->value_name("0.0 to 1.0"),
       "Ratio of normals to use for generating off-surface points")  //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
       "Output file in CSV format:\n  X,Y,Z,VAL")  //
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

  if (!(opts.ratio >= 0.0 && opts.ratio <= 1.0)) {
    throw std::runtime_error("--ratio must be within 0.0 to 1.0");
  }

  run_impl(opts);
}

namespace Eigen {

inline void validate(boost::any& v, const std::vector<std::string>& values, polatory::Mat3*, int) {
  namespace po = boost::program_options;

  if (values.size() != 9) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  polatory::Mat3 aniso;
  aniso << to_double(values.at(0)), to_double(values.at(1)), to_double(values.at(2)),
      to_double(values.at(3)), to_double(values.at(4)), to_double(values.at(5)),
      to_double(values.at(6)), to_double(values.at(7)), to_double(values.at(8));

  v = aniso;
}

}  // namespace Eigen
