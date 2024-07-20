#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <utility>

namespace polatory::point_cloud {

// Generates signed distance function data from given points and normals.
class sdf_data_generator {
 public:
  sdf_data_generator(const geometry::points3d& points, const geometry::vectors3d& normals,
                     double min_distance, double max_distance);

  sdf_data_generator(const geometry::points3d& points, const geometry::vectors3d& normals,
                     double min_distance, double max_distance, const geometry::matrix3d& aniso);

  const geometry::points3d& sdf_points() const;

  const vectord& sdf_values() const;

 private:
  static std::pair<geometry::points3d, vectord> estimate_impl(const geometry::points3d& points,
                                                              const geometry::vectors3d& normals,
                                                              double min_distance,
                                                              double max_distance);

  geometry::points3d sdf_points_;
  vectord sdf_values_;
};

}  // namespace polatory::point_cloud
