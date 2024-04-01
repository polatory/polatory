#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <string>
#include <vector>

#include "../common/common.hpp"
#include "../common/model_options.hpp"

struct options {
  int dim;
  polatory::index_t n_points;
  polatory::index_t n_grad_points;
  polatory::index_t n_eval_points;
  polatory::index_t n_grad_eval_points;
  model_options model_opts;
  int order;
  bool perf;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;

  auto model_opts_desc = make_model_options_description(opts.model_opts);

  po::options_description general_opts_desc("General", 80, 50);
  general_opts_desc.add_options()                                                          //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),                       //
       "Dimension of the domain")                                                          //
      ("n", po::value(&opts.n_points)->required()->value_name("N"),                        //
       "Number of RBF centers")                                                            //
      ("ng", po::value(&opts.n_grad_points)->default_value(0)->value_name("N"),            //
       "Number of RBF centers for gradients")                                              //
      ("n-eval", po::value(&opts.n_eval_points)->required()->value_name("N"),              //
       "Number of evaluation points")                                                      //
      ("ng-eval", po::value(&opts.n_grad_eval_points)->default_value(0)->value_name("N"),  //
       "Number of evaluation points for gradients")                                        //
      ("order",
       po::value(&opts.order)
           ->default_value(polatory::precision::kPrecise)
           ->value_name("ORDER"),                              //
       "Order of the interpolators of fast multipole method")  //
      ("perf", po::bool_switch(&opts.perf),                    //
       "Run fast evaluation only and do not compute accuracy (for performance measurement)");

  po::options_description opts_desc;
  opts_desc.add(general_opts_desc).add(model_opts_desc);

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(
                  argc, argv, opts_desc,
                  po::command_line_style::unix_style ^ po::command_line_style::allow_short),
              vm);
    po::notify(vm);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl
              << "Usage: " << argv[0] << " [OPTION]..." << std::endl
              << opts_desc;
    throw;
  }

  return opts;
}
