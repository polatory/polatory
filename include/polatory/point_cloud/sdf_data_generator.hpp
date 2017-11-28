// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace point_cloud {

// Generates signed distance function data from given points and normals.
class sdf_data_generator {
public:
  sdf_data_generator(
    const geometry::points3d& points,
    const geometry::vectors3d& normals,
    double min_distance,
    double max_distance,
    double ratio = 1.0);

  geometry::points3d sdf_points() const;

  common::valuesd sdf_values() const;

private:
  size_t total_size() const;

  const geometry::points3d& points_;
  const geometry::vectors3d& normals_;

  std::vector<size_t> ext_indices_;
  std::vector<size_t> int_indices_;

  std::vector<double> ext_distances_;
  std::vector<double> int_distances_;
};

} // namespace point_cloud
} // namespace polatory
