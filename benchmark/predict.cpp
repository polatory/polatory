// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/interpolant.hpp>
#include <polatory/io/read_table.hpp>
#include <polatory/io/write_table.hpp>
#include <polatory/rbf/cov_exponential.hpp>

using polatory::interpolant;
using polatory::io::read_points;
using polatory::io::read_values;
using polatory::io::write_values;
using polatory::polynomial::basis_base;
using polatory::rbf::cov_exponential;

int main(int argc, char *argv[]) {
  auto points = read_points(argv[1]);
  auto values = read_values(argv[2]);
  auto prediction_points = read_points(argv[3]);

  double absolute_tolerance = 1e-4;

  cov_exponential cov({ 1.0, 0.02, 0.0 });
  const auto poly_dimension = 3;
  const auto poly_degree = 0;

  interpolant ip(cov, poly_dimension, poly_degree);

  ip.fit(points, values, absolute_tolerance);
  auto prediction_values = ip.evaluate_points(prediction_points);

  write_values(argv[4], prediction_values);

  return 0;
}
