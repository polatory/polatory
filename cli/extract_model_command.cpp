#include <boost/program_options.hpp>
#include <format>
#include <iostream>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "../examples/common/make_model.hpp"
#include "../examples/common/model_options.hpp"
#include "commands.hpp"

using polatory::interpolant;

namespace {

struct options {
  std::string in_file;
  int dim{};
  std::string out_file;
};

template <int Dim>
void run_impl(const options& opts) {
  using Interpolant = interpolant<Dim>;

  auto inter = Interpolant::load(opts.in_file);

  inter.model().save(opts.out_file);
}

}  // namespace

void extract_model_command::run(const std::vector<std::string>& args,
                                const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input interpolant file")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
       "Output model file")  //
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
