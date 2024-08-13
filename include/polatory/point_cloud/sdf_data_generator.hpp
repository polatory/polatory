#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <utility>

namespace polatory::point_cloud {

// Generates signed distance function data from given points and normals.
class SdfDataGenerator {
 public:
  SdfDataGenerator(const geometry::Points3& points, const geometry::Vectors3& normals,
                   double min_distance, double max_distance);

  SdfDataGenerator(const geometry::Points3& points, const geometry::Vectors3& normals,
                   double min_distance, double max_distance, const Mat3& aniso);

  const geometry::Points3& sdf_points() const;

  const VecX& sdf_values() const;

 private:
  static std::pair<geometry::Points3, VecX> estimate_impl(const geometry::Points3& points,
                                                          const geometry::Vectors3& normals,
                                                          double min_distance, double max_distance);

  geometry::Points3 sdf_points_;
  VecX sdf_values_;
};

}  // namespace polatory::point_cloud
