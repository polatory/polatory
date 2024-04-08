#include <boost/program_options.hpp>
#include <format>
#include <iostream>
#include <polatory/polatory.hpp>
#include <string>
#include <vector>

#include "../examples/common/bbox.hpp"
#include "commands.hpp"

using polatory::interpolant;
using polatory::matrixd;
using polatory::read_table;
using polatory::geometry::bbox3d;
using polatory::geometry::points3d;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::isosurface::surface;

namespace {

struct options {
  std::string in_file;
  std::string seed_points_file;
  double isovalue{};
  bbox3d bbox;
  double resolution{};
  std::string out_file;
};

void run_impl(const options& opts) {
  points3d seed_points;

  if (!opts.seed_points_file.empty()) {
    matrixd table = read_table(opts.seed_points_file);
    seed_points = table(Eigen::all, {0, 1, 2});
  }

  auto inter = interpolant<3>::load(opts.in_file);
  auto bbox = opts.bbox.is_empty() ? inter.bbox() : opts.bbox;

  isosurface isosurf(bbox, opts.resolution);
  rbf_field_function field_fn(inter);

  auto surface = seed_points.rows() > 0
                     ? isosurf.generate_from_seed_points(seed_points, field_fn, opts.isovalue)
                     : isosurf.generate(field_fn, opts.isovalue);

  surface.export_obj(opts.out_file);
}

}  // namespace

void isosurface_command::run(const std::vector<std::string>& args,
                             const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input 3D interpolant file")  //
      ("seeds", po::value(&opts.seed_points_file)->value_name("FILE"),
       "Input seed points file in CSV format:\n  X,Y,Z")  //
      ("bbox",
       po::value(&opts.bbox)
           ->multitoken()
           ->default_value({}, "AUTO")
           ->value_name("X_MIN Y_MIN Z_MIN X_MAX Y_MAX Z_MAX"),
       "Output mesh bounding box")  //
      ("res", po::value(&opts.resolution)->required()->value_name("RES"),
       "Output mesh resolution")  //
      ("isoval", po::value(&opts.isovalue)->default_value(0.0, "0.0")->value_name("VAL"),
       "Output mesh isovalue")  //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
       "Output mesh file in OBJ format")  //
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

  run_impl(opts);
}
