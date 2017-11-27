// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/kriging/cross_validation.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/exception.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>

namespace polatory {
namespace kriging {

Eigen::VectorXd k_fold_cross_validation(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                                        const geometry::points3d& points, const Eigen::VectorXd& values,
                                        double absolute_tolerance,
                                        int k) {
  auto n_points = points.rows();
  if (k <= 0 || k > n_points)
    throw common::invalid_argument("0 < k <= points.rows()");

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<size_t> indices(n_points);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), gen);

  auto bbox = geometry::bbox3d::from_points(points);
  Eigen::VectorXd residuals(n_points);

  double n_k = static_cast<double>(n_points) / k;
  for (size_t i = 0; i < k; i++) {
    size_t a = std::round(i * n_k);
    size_t b = std::round((i + 1) * n_k);
    size_t test_set_size = b - a;
    size_t train_set_size = n_points - test_set_size;

    std::vector<size_t> train_set(train_set_size);
    std::vector<size_t> test_set(test_set_size);

    std::copy(indices.begin(), indices.begin() + a, train_set.begin());
    std::copy(indices.begin() + a, indices.begin() + b, test_set.begin());
    std::copy(indices.begin() + b, indices.end(), train_set.begin() + a);

    auto train_points = common::take_rows(points, train_set);
    auto test_points = common::take_rows(points, test_set);

    auto train_values = common::take_rows(values, train_set);
    auto test_values = common::take_rows(values, test_set);

    interpolation::rbf_fitter fitter(rbf, poly_dimension, poly_degree, train_points);
    auto weights = fitter.fit(train_values, absolute_tolerance);

    interpolation::rbf_evaluator<> eval(rbf, poly_dimension, poly_degree, train_points, bbox);
    eval.set_weights(weights);
    auto test_values_fit = eval.evaluate_points(test_points);

    for (size_t j = 0; j < test_set_size; j++) {
      residuals(test_set[j]) = test_values(j) - test_values_fit(j);
    }
  }

  return residuals;
}

} // namespace kriging
} // namespace polatory
