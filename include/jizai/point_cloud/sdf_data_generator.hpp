#pragma once

#include <jizai/geometry/point3d.hpp>
#include <jizai/types.hpp>
#include <optional>
#include <utility>

namespace jizai::point_cloud {

// Generates signed distance function data from given points and normals.
class SdfDataGenerator {
 public:
  SdfDataGenerator(const geometry::Points3& points, const geometry::Vectors3& normals,
                   std::optional<double> offset = std::nullopt);

  SdfDataGenerator(const geometry::Points3& points, const geometry::Vectors3& normals,
                   const Mat3& aniso);

  SdfDataGenerator(const geometry::Points3& points, const geometry::Vectors3& normals,
                   std::optional<double> offset, const Mat3& aniso);

  const geometry::Points3& sdf_points() const;

  const VecX& sdf_values() const;

 private:
  static std::pair<geometry::Points3, VecX> estimate_impl(const geometry::Points3& points,
                                                          const geometry::Vectors3& normals,
                                                          std::optional<double> offset);

  geometry::Points3 sdf_points_;
  VecX sdf_values_;
};

}  // namespace jizai::point_cloud
