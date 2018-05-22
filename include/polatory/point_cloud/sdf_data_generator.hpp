// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

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
    double multiplication = 2.0);

  const geometry::points3d& sdf_points() const;
  const common::valuesd& sdf_values() const;

private:
  const geometry::points3d points_;    // Do not hold a reference to a temporary object.
  const geometry::vectors3d normals_;  // Do not hold a reference to a temporary object.

  geometry::points3d sdf_points_;
  common::valuesd sdf_values_;
};

}  // namespace point_cloud
}  // namespace polatory
