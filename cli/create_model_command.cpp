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

namespace {

struct Options {
  int dim{};
  ModelOptions model_opts;
  std::string out_file;
};

template <int Dim>
void run_impl(const Options& opts) {
  auto model = make_model<Dim>(opts.model_opts);
  model.save(opts.out_file);
}

}  // namespace

void CreateModelCommand::run(const std::vector<std::string>& args,
                             const GlobalOptions& global_opts) {
  namespace po = boost::program_options;

  Options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("out", po::value(&opts.out_file)->required()->value_name("FILE"),
       "Output model file")  //
      ;

  auto model_opts_desc = make_model_options_description(opts.model_opts);
  opts_desc.add(model_opts_desc);

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
