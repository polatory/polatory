// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/polatory.hpp>

using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolant;
using polatory::model;
using polatory::polynomial::polynomial_basis_base;
using polatory::rbf::cov_exponential;
using polatory::read_table;
using polatory::write_table;

int main(int argc, char *argv[]) {
  points3d points = read_table(argv[1]);
  valuesd values = read_table(argv[2]);
  points3d prediction_points = read_table(argv[3]);

  double absolute_tolerance = 1e-4;

  const auto poly_dimension = 3;
  const auto poly_degree = 0;
  model model(cov_exponential({ 1.0, 0.02 }), poly_dimension, poly_degree);

  interpolant interpolant(model);

  interpolant.fit(points, values, absolute_tolerance);
  auto prediction_values = interpolant.evaluate_points(prediction_points);

  write_table(argv[4], prediction_values);

  return 0;
}
