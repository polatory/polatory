// Copyright (c) 2016, GSI and The Polatory Authors.

#include <exception>
#include <iomanip>
#include <iostream>

#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::common::take_cols;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::index_t;
using polatory::kriging::empirical_variogram;
using polatory::read_table;

int main(int argc, const char *argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    auto table = read_table(opts.in_file);
    points3d points = take_cols(table, 0, 1, 2);
    valuesd values = table.col(3);

    empirical_variogram emp_variog(points, values, opts.bin_width, opts.n_bins);
    const auto& bin_distance = emp_variog.bin_distance();
    const auto& bin_gamma = emp_variog.bin_gamma();
    const auto& bin_num_pairs = emp_variog.bin_num_pairs();

    std::cout << "Empirical variogram:" << std::endl
              << std::setw(12) << "n_pairs" << std::setw(12) << "distance" << std::setw(12) << "gamma" << std::endl;
    auto n_bins = static_cast<index_t>(bin_num_pairs.size());
    for (index_t bin = 0; bin < n_bins; bin++) {
      std::cout << std::setw(12) << bin_num_pairs[bin] << std::setw(12) << bin_distance[bin] << std::setw(12) <<  bin_gamma[bin] << std::endl;
    }

    if (!opts.out_file.empty()) {
      emp_variog.save(opts.out_file);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
