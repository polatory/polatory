#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory::point_cloud {

// Generates signed distance function data from given points and normals.
class sdf_data_generator {
 public:
  sdf_data_generator(const geometry::points3d& points, const geometry::vectors3d& normals,
                     double min_distance, double max_distance, double multiplication = 2.0);

  const geometry::points3d& sdf_points() const;
  const vectord& sdf_values() const;

 private:
  geometry::points3d sdf_points_;
  vectord sdf_values_;
};

}  // namespace polatory::point_cloud
