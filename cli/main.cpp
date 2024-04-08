#include <boost/program_options.hpp>
#include <exception>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "commands.hpp"

int main(int argc, const char* argv[]) {
  try {
    namespace po = boost::program_options;

    global_options opts;

    po::options_description opts_desc("Global options", 80, 50);
    opts_desc.add_options()  //
        ("help,h", po::bool_switch(&opts.help),
         "Display this help");  //

    auto parsed = po::command_line_parser(argc, argv).options(opts_desc).allow_unregistered().run();

    po::variables_map vm;
    po::store(parsed, vm);
    po::notify(vm);

    auto args = po::collect_unrecognized(parsed.options, po::include_positional);

    if (args.empty()) {
      std::cout << "usage: polatory [OPTIONS] COMMAND [ARGS]" << std::endl << opts_desc;
      return opts.help ? 0 : 1;
    }

    auto command = args.at(0);
    args.erase(args.begin());

    if (command == create_model_command::kName) {
      create_model_command::run(args, opts);
    } else if (command == cross_validate_command::kName) {
      cross_validate_command::run(args, opts);
    } else if (command == estimate_normals_command::kName) {
      estimate_normals_command::run(args, opts);
    } else if (command == evaluate_command::kName) {
      evaluate_command::run(args, opts);
    } else if (command == extract_model_command::kName) {
      extract_model_command::run(args, opts);
    } else if (command == fit_command::kName) {
      fit_command::run(args, opts);
    } else if (command == fit_model_to_variogram_command::kName) {
      fit_model_to_variogram_command::run(args, opts);
    } else if (command == isosurface_command::kName) {
      isosurface_command::run(args, opts);
    } else if (command == normals_to_sdf_command::kName) {
      normals_to_sdf_command::run(args, opts);
    } else if (command == show_model_command::kName) {
      show_model_command::run(args, opts);
    } else if (command == show_variogram_command::kName) {
      show_variogram_command::run(args, opts);
    } else if (command == surface_25d_command::kName) {
      surface_25d_command::run(args, opts);
    } else if (command == unique_command::kName) {
      unique_command::run(args, opts);
    } else if (command == variogram_command::kName) {
      variogram_command::run(args, opts);
    } else {
      throw std::runtime_error(std::format("unknown command: '{}'", command));
      return 1;
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "unknown error" << std::endl;
    return 1;
  }
}
