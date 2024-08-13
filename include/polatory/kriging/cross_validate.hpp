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
VecX cross_validate(const Model<Dim>& model, const geometry::Points<Dim>& points,
                    const VecX& values, const Eigen::VectorXi& set_ids, double tolerance,
                    int max_iter, double accuracy) {
  auto n_points = points.rows();
  VecX predictions = VecX::Zero(n_points);

  std::unordered_set<int> ids(set_ids.begin(), set_ids.end());
  for (auto id : ids) {
    std::vector<Index> test_set;
    for (Index i = 0; i < n_points; i++) {
      if (set_ids(i) == id) {
        test_set.push_back(i);
      }
    }

    auto train_set = common::complementary_indices(test_set, n_points);

    geometry::Points<Dim> train_points = points(train_set, Eigen::all);
    geometry::Points<Dim> test_points = points(test_set, Eigen::all);

    VecX train_values = values(train_set, Eigen::all);

    Interpolant<Dim> interpolant(model);
    interpolant.fit(train_points, train_values, tolerance, max_iter, accuracy);
    auto test_values_fit = interpolant.evaluate(test_points, accuracy);

    for (Index j = 0; j < static_cast<Index>(test_set.size()); j++) {
      predictions(test_set.at(j)) = test_values_fit(j);
    }
  }

  return predictions;
}

}  // namespace polatory::kriging
