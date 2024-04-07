#pragma once

#include <Eigen/Core>
#include <polatory/common/complementary_indices.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <unordered_set>
#include <vector>

namespace polatory::kriging {

template <int Dim>
inline vectord cross_validate(const model<Dim>& model, const geometry::pointsNd<Dim>& points,
                              const vectord& values, const Eigen::VectorXi& set_ids,
                              double absolute_tolerance, int max_iter) {
  auto n_points = points.rows();
  vectord predictions = vectord::Zero(n_points);

  std::unordered_set<int> ids(set_ids.begin(), set_ids.end());
  for (auto id : ids) {
    std::vector<index_t> test_set;
    for (index_t i = 0; i < n_points; i++) {
      if (set_ids(i) == id) {
        test_set.push_back(i);
      }
    }

    auto train_set = common::complementary_indices(test_set, n_points);

    geometry::pointsNd<Dim> train_points = points(train_set, Eigen::all);
    geometry::pointsNd<Dim> test_points = points(test_set, Eigen::all);

    vectord train_values = values(train_set, Eigen::all);

    interpolant<Dim> interpolant(model);
    interpolant.fit(train_points, train_values, absolute_tolerance, max_iter);
    auto test_values_fit = interpolant.evaluate(test_points);

    for (index_t j = 0; j < static_cast<index_t>(test_set.size()); j++) {
      predictions(test_set.at(j)) = test_values_fit(j);
    }
  }

  return predictions;
}

}  // namespace polatory::kriging
