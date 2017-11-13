// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <Eigen/Core>

namespace polatory {
namespace point_cloud {

// Generates signed distance function data from given points and normals.
class sdf_data_generator {
public:
  sdf_data_generator(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    double min_distance,
    double max_distance,
    double ratio = 1.0);

  std::vector<Eigen::Vector3d> sdf_points() const;

  Eigen::VectorXd sdf_values() const;

private:
  size_t total_size() const;

  const std::vector<Eigen::Vector3d>& points_;
  const std::vector<Eigen::Vector3d>& normals_;

  std::vector<size_t> ext_indices_;
  std::vector<size_t> int_indices_;

  std::vector<double> ext_distances_;
  std::vector<double> int_distances_;
};

} // namespace point_cloud
} // namespace polatory
