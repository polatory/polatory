// Copyright (c) 2016, GSI and The Polatory Authors.

#include "polatory/point_cloud/sdf_data_generator.hpp"

#include <cassert>

#include <boost/range/combine.hpp>

#include "polatory/common/quasi_random_sequence.hpp"
#include "polatory/point_cloud/kdtree.hpp"

namespace polatory {
namespace point_cloud {

sdf_data_generator::sdf_data_generator(
  const std::vector<Eigen::Vector3d>& points,
  const std::vector<Eigen::Vector3d>& normals,
  double min_distance,
  double max_distance,
  double ratio)
  : points_(points)
  , normals_(normals) {
  assert(points.size() == normals.size());
  assert(ratio > 0.0 && ratio <= 2.0);

  kdtree tree(points, true);

  std::vector<size_t> nn_indices;
  std::vector<double> nn_distances;

  auto reduced_indices = common::quasi_random_sequence((ratio / 2.0) * points.size());

  for (auto i : reduced_indices) {
    const auto& p = points[i];
    const auto& n = normals[i];

    auto d = max_distance;

    Eigen::Vector3d q = p + d * n;
    tree.knn_search(q, 1, nn_indices, nn_distances);
    while (nn_indices[0] != i && nn_distances[0] > 0.0) {
      d = 0.99 * (points[nn_indices[0]] - p).norm() / 2.0;
      q = p + d * n;
      tree.knn_search(q, 1, nn_indices, nn_distances);
    }

    if (d < min_distance)
      continue;

    ext_indices_.push_back(i);
    ext_distances_.push_back(d);
  }

  for (auto i : reduced_indices) {
    const auto& p = points[i];
    const auto& n = normals[i];

    auto d = max_distance;

    Eigen::Vector3d q = p - d * n;
    tree.knn_search(q, 1, nn_indices, nn_distances);
    while (nn_indices[0] != i && nn_distances[0] > 0.0) {
      d = 0.99 * (points[nn_indices[0]] - p).norm() / 2.0;
      q = p - d * n;
      tree.knn_search(q, 1, nn_indices, nn_distances);
    }

    if (d < min_distance)
      continue;

    int_indices_.push_back(i);
    int_distances_.push_back(d);
  }
}

std::vector<Eigen::Vector3d> sdf_data_generator::sdf_points() const {
  std::vector<Eigen::Vector3d> sdf_points(points_);
  sdf_points.reserve(total_size());

  for (auto i_d : boost::combine(ext_indices_, ext_distances_)) {
    size_t i;
    double d;
    boost::tie(i, d) = i_d;

    const auto& p = points_[i];
    const auto& n = normals_[i];
    sdf_points.push_back(p + d * n);
  }

  for (auto i_d : boost::combine(int_indices_, int_distances_)) {
    size_t i;
    double d;
    boost::tie(i, d) = i_d;

    const auto& p = points_[i];
    const auto& n = normals_[i];
    sdf_points.push_back(p - d * n);
  }

  return sdf_points;
}

Eigen::VectorXd sdf_data_generator::sdf_values() const {
  Eigen::VectorXd values = Eigen::VectorXd::Zero(total_size());

  values.segment(points_.size(), ext_indices_.size()) =
    Eigen::Map<const Eigen::VectorXd>(ext_distances_.data(), ext_indices_.size());

  values.tail(int_indices_.size()) =
    -Eigen::Map<const Eigen::VectorXd>(int_distances_.data(), int_indices_.size());

  return values;
}

size_t sdf_data_generator::total_size() const {
  return points_.size() + ext_indices_.size() + int_indices_.size();
}

} // namespace point_cloud
} // namespace polatory
