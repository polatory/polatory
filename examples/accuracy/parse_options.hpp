#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <string>

#include "../common/model_options.hpp"

struct options {
  std::string in_file;
  int dim;
  polatory::index_t n_eval_points;
  polatory::index_t n_grad_eval_points;
  int order;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;

  po::options_description general_opts_desc("General options", 80, 50);
  general_opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->value_name("FILE"),
       "Input interpolant file")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of RBF centers and evaluation points")  //
      ("n-eval", po::value(&opts.n_eval_points)->default_value(0)->value_name("N"),
       "Number of evaluation points")  //
      ("n-grad-eval", po::value(&opts.n_grad_eval_points)->default_value(0)->value_name("N"),
       "Number of evaluation points for gradients")  //
      ("order",
       po::value(&opts.order)->default_value(polatory::precision::kPrecise)->value_name("ORDER"),
       "Order of the interpolators of fast multipole method")  //
      ;

  po::options_description opts_desc(80, 50);
  opts_desc.add(general_opts_desc);

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(
                  argc, argv, opts_desc,
                  po::command_line_style::unix_style ^ po::command_line_style::allow_short),
              vm);
    po::notify(vm);
  } catch (std::exception& e) {
    std::cout << "usage: accuracy [OPTIONS]\n" << opts_desc;
    throw;
  }

  return opts;
}
