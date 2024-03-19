#include <Eigen/Core>
#include <exception>
#include <iomanip>
#include <iostream>
#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::index_t;
using polatory::read_table;
using polatory::tabled;
using polatory::common::valuesd;
using polatory::geometry::pointsNd;
using polatory::kriging::empirical_variogram;

template <int Dim>
void main_impl(const options& opts) {
  using EmpiricalVariogram = empirical_variogram<Dim>;
  using Points = pointsNd<Dim>;

  // Load points (x,y,z) and values (value).
  tabled table = read_table(opts.in_file);
  Points points = table(Eigen::all, Eigen::seqN(0, Dim));
  valuesd values = table.col(3);

  // Compute the empirical variogram.
  EmpiricalVariogram emp_variog(points, values, opts.bin_width, opts.num_bins);
  const auto& bin_distance = emp_variog.bin_distance();
  const auto& bin_gamma = emp_variog.bin_gamma();
  const auto& bin_num_pairs = emp_variog.bin_num_pairs();

  std::cout << "Empirical variogram:" << std::endl
            << std::setw(12) << "distance" << std::setw(12) << "gamma" << std::setw(12)
            << "num_pairs" << std::endl;
  auto num_bins = static_cast<index_t>(bin_num_pairs.size());
  for (index_t bin = 0; bin < num_bins; bin++) {
    std::cout << std::setw(12) << bin_distance.at(bin) << std::setw(12) << bin_gamma.at(bin)
              << std::setw(12) << bin_num_pairs.at(bin) << std::endl;
  }

  // Save the empirical variogram.
  if (!opts.out_file.empty()) {
    emp_variog.save(opts.out_file);
  }
}

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);
    switch (opts.dim) {
      case 1:
        main_impl<1>(opts);
        break;
      case 2:
        main_impl<2>(opts);
        break;
      case 3:
        main_impl<3>(opts);
        break;
      default:
        throw std::runtime_error("Unsupported dimension: " + std::to_string(opts.dim));
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
