#include <algorithm>
#include <cmath>
#include <numeric>
#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/kriging/cross_validation.hpp>
#include <random>
#include <stdexcept>

namespace polatory::kriging {

common::valuesd k_fold_cross_validation(const model& model, const geometry::points3d& points,
                                        const common::valuesd& values, double absolute_tolerance,
                                        index_t k) {
  auto n_points = static_cast<index_t>(points.rows());
  if (n_points < 2) {
    throw std::invalid_argument("points.row() must be greater than or equal to 2.");
  }

  if (k < 2 || k > n_points) {
    throw std::invalid_argument("k must be within the range of 2 to points.rows().");
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<index_t> indices(n_points);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), gen);

  auto bbox = geometry::bbox3d::from_points(points);
  common::valuesd residuals(n_points);

  auto n_k = static_cast<double>(n_points) / k;
  for (index_t i = 0; i < k; i++) {
    auto a = static_cast<index_t>(std::round(i * n_k));
    auto b = static_cast<index_t>(std::round((i + 1) * n_k));
    auto test_set_size = b - a;
    auto train_set_size = n_points - test_set_size;

    std::vector<index_t> train_set(train_set_size);
    std::vector<index_t> test_set(test_set_size);

    std::copy(indices.begin(), indices.begin() + a, train_set.begin());
    std::copy(indices.begin() + a, indices.begin() + b, test_set.begin());
    std::copy(indices.begin() + b, indices.end(), train_set.begin() + a);

    auto train_points = common::take_rows(points, train_set);
    auto test_points = common::take_rows(points, test_set);

    auto train_values = common::take_rows(values, train_set);
    auto test_values = common::take_rows(values, test_set);

    interpolation::rbf_fitter fitter(model, train_points);
    auto weights = fitter.fit(train_values, absolute_tolerance);

    interpolation::rbf_evaluator<> eval(model, train_points, bbox);
    eval.set_weights(weights);
    auto test_values_fit = eval.evaluate(test_points);

    for (index_t j = 0; j < test_set_size; j++) {
      residuals(test_set[j]) = test_values(j) - test_values_fit(j);
    }
  }

  return residuals;
}

}  // namespace polatory::kriging
