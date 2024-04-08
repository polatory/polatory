#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <polatory/kriging.hpp>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>

#include "parse_options.hpp"

using polatory::index_t;
using polatory::matrixd;
using polatory::read_table;
using polatory::vectord;
using polatory::common::save;
using polatory::geometry::pointsNd;
using polatory::kriging::variogram;
using polatory::kriging::variogram_calculator;

template <int Dim>
void main_impl(const options& opts) {
  using Points = pointsNd<Dim>;
  using VariogramCalculator = variogram_calculator<Dim>;

  // Load points (x,y,z) and values (value).
  matrixd table = read_table(opts.in_file);
  Points points = table(Eigen::all, Eigen::seqN(0, Dim));
  vectord values = table.col(Dim);

  // Compute the empirical variogram.
  VariogramCalculator calc(opts.lag_distance, opts.num_lags);
  if (opts.aniso) {
    calc.set_directions(VariogramCalculator::kAnisotropicDirections);
  }
  auto variogs = calc.calculate(points, values);

  // Save the empirical variogram.
  save(opts.out_file, variogs);
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
