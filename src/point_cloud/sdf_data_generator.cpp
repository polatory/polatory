// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/point_cloud/sdf_data_generator.hpp>

#include <cassert>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/quasi_random_sequence.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>

namespace polatory {
namespace point_cloud {

sdf_data_generator::sdf_data_generator(
  const geometry::points3d& points,
  const geometry::vectors3d& normals,
  double min_distance,
  double max_distance,
  double ratio)
  : points_(points)
  , normals_(normals) {
  assert(points.rows() == normals.rows());
  assert(ratio > 0.0 && ratio <= 2.0);

  kdtree tree(points, true);

  std::vector<size_t> nn_indices;
  std::vector<double> nn_distances;

  auto reduced_indices = common::quasi_random_sequence((ratio / 2.0) * points.rows());

  for (auto i : reduced_indices) {
    auto p = points.row(i);
    auto n = normals.row(i);

    auto d = max_distance;

    geometry::point3d q = p + d * n;
    tree.knn_search(q, 1, nn_indices, nn_distances);
    while (nn_indices[0] != i && nn_distances[0] > 0.0) {
      d = 0.99 * (points.row(nn_indices[0]) - p).norm() / 2.0;
      q = p + d * n;
      tree.knn_search(q, 1, nn_indices, nn_distances);
    }

    if (d < min_distance)
      continue;

    ext_indices_.push_back(i);
    ext_distances_.push_back(d);
  }

  for (auto i : reduced_indices) {
    auto p = points.row(i);
    auto n = normals.row(i);

    auto d = max_distance;

    geometry::point3d q = p - d * n;
    tree.knn_search(q, 1, nn_indices, nn_distances);
    while (nn_indices[0] != i && nn_distances[0] > 0.0) {
      d = 0.99 * (points.row(nn_indices[0]) - p).norm() / 2.0;
      q = p - d * n;
      tree.knn_search(q, 1, nn_indices, nn_distances);
    }

    if (d < min_distance)
      continue;

    int_indices_.push_back(i);
    int_distances_.push_back(d);
  }
}

geometry::points3d sdf_data_generator::sdf_points() const {
  geometry::points3d sdf_points(total_size(), 3);
  sdf_points.topRows(points_.rows()) = points_;

  auto sdf_point_it = common::row_begin(sdf_points) + points_.rows();

  for (size_t i = 0; i < ext_indices_.size(); i++) {
    auto idx = ext_indices_[i];
    auto d = ext_distances_[i];

    auto p = points_.row(idx);
    auto n = normals_.row(idx);
    *sdf_point_it++ = p + d * n;
  }

  for (size_t i = 0; i < int_indices_.size(); i++) {
    auto idx = int_indices_[i];
    auto d = int_distances_[i];

    auto p = points_.row(i);
    auto n = normals_.row(i);
    *sdf_point_it++ = p - d * n;
  }

  return sdf_points;
}

common::valuesd sdf_data_generator::sdf_values() const {
  common::valuesd values = common::valuesd::Zero(total_size());

  values.segment(points_.rows(), ext_indices_.size()) =
    Eigen::Map<const common::valuesd>(ext_distances_.data(), ext_indices_.size());

  values.tail(int_indices_.size()) =
    -Eigen::Map<const common::valuesd>(int_distances_.data(), int_indices_.size());

  return values;
}

size_t sdf_data_generator::total_size() const {
  return points_.rows() + ext_indices_.size() + int_indices_.size();
}

} // namespace point_cloud
} // namespace polatory
