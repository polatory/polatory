#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <random>
#include <stdexcept>

namespace polatory::kriging {

template <int Dim>
inline common::valuesd k_fold_cross_validation(const model<Dim>& model,
                                               const geometry::pointsNd<Dim>& points,
                                               const common::valuesd& values,
                                               double absolute_tolerance, int max_iter, index_t k) {
  auto n_points = points.rows();
  if (n_points < 2) {
    throw std::invalid_argument("points.row() must be greater than or equal to 2.");
  }

  if (k < 2 || k > n_points) {
    throw std::invalid_argument("k must be within 2 to points.rows().");
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<index_t> indices(n_points);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), gen);

  common::valuesd residuals(n_points);

  auto n_k = static_cast<double>(n_points) / static_cast<double>(k);
  for (index_t i = 0; i < k; i++) {
    auto a = static_cast<index_t>(std::round(static_cast<double>(i) * n_k));
    auto b = static_cast<index_t>(std::round(static_cast<double>(i + 1) * n_k));
    auto test_set_size = b - a;
    auto train_set_size = n_points - test_set_size;

    std::vector<index_t> train_set(train_set_size);
    std::vector<index_t> test_set(test_set_size);

    std::copy(indices.begin(), indices.begin() + a, train_set.begin());
    std::copy(indices.begin() + a, indices.begin() + b, test_set.begin());
    std::copy(indices.begin() + b, indices.end(), train_set.begin() + a);

    geometry::pointsNd<Dim> train_points = points(train_set, Eigen::all);
    geometry::pointsNd<Dim> test_points = points(test_set, Eigen::all);

    common::valuesd train_values = values(train_set, Eigen::all);
    common::valuesd test_values = values(test_set, Eigen::all);

    interpolant<Dim> interpolant(model);
    interpolant.fit(train_points, train_values, absolute_tolerance, max_iter);
    auto test_values_fit = interpolant.evaluate(test_points);

    for (index_t j = 0; j < test_set_size; j++) {
      residuals(test_set.at(j)) = test_values(j) - test_values_fit(j);
    }
  }

  return residuals;
}

}  // namespace polatory::kriging
