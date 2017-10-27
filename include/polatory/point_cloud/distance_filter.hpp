// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <Eigen/Core>

namespace polatory {
namespace point_cloud {

class distance_filter {
public:
  distance_filter(const std::vector<Eigen::Vector3d>& points, double distance);

  const std::vector<size_t>& filtered_indices() const;

  std::vector<Eigen::Vector3d> filter_points(const std::vector<Eigen::Vector3d>& points) const;

  Eigen::VectorXd filter_values(const Eigen::VectorXd& values) const;

private:
  const size_t n_points_;

  std::vector<size_t> filtered_indices_;
};

} // namespace point_cloud
} // namespace polatory
