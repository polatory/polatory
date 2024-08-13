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

    GlobalOptions opts;

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

    if (command == CreateModelCommand::kName) {
      CreateModelCommand::run(args, opts);
    } else if (command == CrossValidateCommand::kName) {
      CrossValidateCommand::run(args, opts);
    } else if (command == EstimateNormalsCommand::kName) {
      EstimateNormalsCommand::run(args, opts);
    } else if (command == EvaluateCommand::kName) {
      EvaluateCommand::run(args, opts);
    } else if (command == ExtractModelCommand::kName) {
      ExtractModelCommand::run(args, opts);
    } else if (command == FitCommand::kName) {
      FitCommand::run(args, opts);
    } else if (command == FitModelToVariogramCommand::kName) {
      FitModelToVariogramCommand::run(args, opts);
    } else if (command == IsosurfaceCommand::kName) {
      IsosurfaceCommand::run(args, opts);
    } else if (command == NormalsToSdfCommand::kName) {
      NormalsToSdfCommand::run(args, opts);
    } else if (command == ShowModelCommand::kName) {
      ShowModelCommand::run(args, opts);
    } else if (command == ShowVariogramCommand::kName) {
      ShowVariogramCommand::run(args, opts);
    } else if (command == Surface25DCommand::kName) {
      Surface25DCommand::run(args, opts);
    } else if (command == UniqueCommand::kName) {
      UniqueCommand::run(args, opts);
    } else if (command == VariogramCommand::kName) {
      VariogramCommand::run(args, opts);
    } else {
      throw std::runtime_error(std::format("unknown command: '{}'", command));
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
