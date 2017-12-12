// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/common/types.hpp>
#include <polatory/config.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/table.hpp>

using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolant;
using polatory::polynomial::basis_base;
using polatory::rbf::cov_exponential;
using polatory::read_table;
using polatory::write_table;

int main(int argc, char *argv[]) {
  points3d points = read_table(argv[1]);
  valuesd values = read_table(argv[2]);
  points3d prediction_points = read_table(argv[3]);

  double absolute_tolerance = 1e-4;

  cov_exponential cov({ 1.0, 0.02, 0.0 });
  const auto poly_dimension = 3;
  const auto poly_degree = 0;

  interpolant ip(cov, poly_dimension, poly_degree);

  ip.fit(points, values, absolute_tolerance);
  auto prediction_values = ip.evaluate_points(prediction_points);

  write_table(argv[4], prediction_values);

  return 0;
}
