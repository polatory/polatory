#include <Eigen/Core>
#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <format>
#include <iostream>
#include <limits>
#include <polatory/polatory.hpp>
#include <string>
#include <vector>

#include "../examples/common/bbox.hpp"
#include "commands.hpp"

using polatory::index_t;
using polatory::interpolant;
using polatory::matrixd;
using polatory::read_table;
using polatory::geometry::bbox3d;
using polatory::geometry::matrix3d;
using polatory::geometry::points3d;
using polatory::isosurface::isosurface;
using polatory::isosurface::rbf_field_function;
using polatory::isosurface::surface;
using polatory::numeric::to_double;

namespace {

struct options {
  std::string in_file;
  std::string seed_points_file;
  double accuracy{};
  double grad_accuracy{};
  double isovalue{};
  bbox3d bbox;
  double resolution{};
  matrix3d aniso;
  int refine{};
  std::string out_file;
};

void run_impl(const options& opts) {
  auto inter = interpolant<3>::load(opts.in_file);
  auto bbox = opts.bbox.is_empty() ? inter.bbox() : opts.bbox;

  isosurface isosurf(bbox, opts.resolution, opts.aniso);
  rbf_field_function field_fn(inter, opts.accuracy, opts.grad_accuracy);

  points3d seed_points;
  if (!opts.seed_points_file.empty()) {
    matrixd table = read_table(opts.seed_points_file);
    seed_points = table(Eigen::all, {0, 1, 2});
  }

  auto surface = seed_points.rows() > 0 ? isosurf.generate_from_seed_points(
                                              seed_points, field_fn, opts.isovalue, opts.refine)
                                        : isosurf.generate(field_fn, opts.isovalue, opts.refine);

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
      ("acc",
       po::value(&opts.accuracy)
           ->default_value(std::numeric_limits<double>::infinity(), "ANY")
           ->value_name("ACC"),
       "Absolute evaluation accuracy")  //
      ("grad-acc",
       po::value(&opts.grad_accuracy)
           ->default_value(std::numeric_limits<double>::infinity(), "ANY")
           ->value_name("ACC"),
       "Absolute gradient evaluation accuracy")  //
      ("bbox",
       po::value(&opts.bbox)
           ->multitoken()
           ->default_value({}, "AUTO")
           ->value_name("X_MIN Y_MIN Z_MIN X_MAX Y_MAX Z_MAX"),
       "Output mesh bounding box")  //
      ("res", po::value(&opts.resolution)->required()->value_name("RES"),
       "Output mesh resolution")  //
      ("aniso",
       po::value(&opts.aniso)
           ->multitoken()
           ->default_value(matrix3d::Identity(), "1 0 0 0 1 0 0 0 1")
           ->value_name("A_11 A_12 ... A_33"),
       "Elements of the anisotropy matrix")  //
      ("isoval", po::value(&opts.isovalue)->default_value(0.0, "0.0")->value_name("VAL"),
       "Output mesh isovalue")  //
      ("refine", po::value(&opts.refine)->default_value(1)->value_name("N"),
       "Number of vertex refinement passes")  //
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

namespace Eigen {

inline void validate(boost::any& v, const std::vector<std::string>& values, matrix3d*, int) {
  namespace po = boost::program_options;

  if (values.size() != 9) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  matrix3d aniso;
  aniso << to_double(values.at(0)), to_double(values.at(1)), to_double(values.at(2)),
      to_double(values.at(3)), to_double(values.at(4)), to_double(values.at(5)),
      to_double(values.at(6)), to_double(values.at(7)), to_double(values.at(8));

  v = aniso;
}

}  // namespace Eigen
