#include <boost/program_options.hpp>
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>
#include <polatory/kriging.hpp>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "commands.hpp"

using polatory::index_t;
using polatory::kriging::variogram_set;

namespace {

int format_precision(double x) {
  if (x == 0.0) {
    return 0;
  }

  return std::max(0, 5 - static_cast<int>(std::floor(std::log10(std::abs(x)))));
}

struct options {
  std::string in_file;
  int dim{};
  int id{};
};

template <int Dim>
void run_impl(const options& opts) {
  using VariogramSet = variogram_set<Dim>;

  auto variog_set = VariogramSet::load(opts.in_file);
  auto deg = std::numbers::pi / 180.0;

  if (opts.id == -1) {
    std::cout << std::format("  {:>4}  {:>10}  {:>10}  {:>10}\n", "id", "azimuth", "elevation",
                             "num_pairs");
    for (index_t i = 0; i < variog_set.num_variograms(); ++i) {
      const auto& v = variog_set.variograms().at(i);
      const auto& dir = v.direction();

      auto elev = std::acos(dir(2)) / deg;
      auto az = std::atan2(dir(0), dir(1)) / deg;
      if (elev < 0.0) {
        elev = -elev;
        az += 180.0;
      }
      if (az < 0.0) {
        az += 360.0;
      }

      std::cout << std::format("  {:>4}  {:>10.3f}  {:>10.3f}  {:>10}\n", i, az, elev,
                               v.num_pairs());
    }
  } else {
    const auto& v = variog_set.variograms().at(opts.id);
    auto max_distance = *std::max_element(v.bin_distance().begin(), v.bin_distance().end());
    auto max_gamma = *std::max_element(v.bin_gamma().begin(), v.bin_gamma().end());
    auto distance_prec = format_precision(max_distance);
    auto gamma_prec = format_precision(max_gamma);

    std::cout << std::format("  {:>10}  {:>10}  {:>10}\n", "distance", "gamma", "num_pairs");
    for (index_t i = 0; i < v.num_bins(); ++i) {
      std::cout << std::format("  {:>10.{}f}  {:>10.{}f}  {:>10}\n", v.bin_distance().at(i),
                               distance_prec, v.bin_gamma().at(i), gamma_prec,
                               v.bin_num_pairs().at(i));
    }
  }
}

}  // namespace

void show_variogram_command::run(const std::vector<std::string>& args,
                                 const global_options& global_opts) {
  namespace po = boost::program_options;

  options opts;

  po::options_description opts_desc("Options", 80, 50);
  opts_desc.add_options()  //
      ("in", po::value(&opts.in_file)->required()->value_name("FILE"),
       "Input variogram file")  //
      ("dim", po::value(&opts.dim)->required()->value_name("1|2|3"),
       "Dimension of input points")  //
      ("id", po::value(&opts.id)->default_value(-1, "NONE")->value_name("ID"),
       "ID of the variogram")  //
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
