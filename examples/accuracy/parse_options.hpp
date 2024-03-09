#pragma once

#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <string>
#include <utility>
#include <vector>

#include "../common/common.hpp"

struct options {
  int dim;
  polatory::index_t n_points;
  polatory::index_t n_grad_points;
  polatory::index_t n_eval_points;
  polatory::index_t n_eval_grad_points;
  std::string rbf_name;
  std::vector<double> rbf_params;
  int poly_degree;
};

inline options parse_options(int argc, const char* argv[]) {
  namespace po = boost::program_options;

  options opts;
  std::vector<std::string> rbf_vec;
  std::vector<double> mesh_vals_vec;
  std::vector<std::string> mesh_files_vec;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()                                                                      //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),                           //
       "Dimension of the domain")                                                              //
      ("n", po::value(&opts.n_points)->required()->value_name("N"),                            //
       "Number of RBF centers")                                                                //
      ("n-grad", po::value(&opts.n_grad_points)->default_value(0)->value_name("N"),            //
       "Number of RBF gradient centers")                                                       //
      ("n-eval", po::value(&opts.n_eval_points)->required()->value_name("N"),                  //
       "Number of evaluation points")                                                          //
      ("n-eval-grad", po::value(&opts.n_eval_grad_points)->default_value(0)->value_name("N"),  //
       "Number of evaluation gradient points")                                                 //
      ("rbf", po::value(&rbf_vec)->multitoken()->required()->value_name("..."),                //
       rbf_cov_list)                                                                           //
      ("deg", po::value(&opts.poly_degree)->default_value(0)->value_name("-1|0|1|2"),          //
       "Degree of the drift polynomial");

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

  opts.rbf_name = rbf_vec.at(0);
  for (std::size_t i = 1; i < rbf_vec.size(); i++) {
    opts.rbf_params.push_back(polatory::numeric::to_double(rbf_vec.at(i)));
  }

  return opts;
}
