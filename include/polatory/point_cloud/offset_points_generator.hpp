#pragma once

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory::point_cloud {

class offset_points_generator {
 public:
  offset_points_generator(const geometry::points3d& points, const geometry::vectors3d& normals,
                          double offset);

  offset_points_generator(const geometry::points3d& points, const geometry::vectors3d& normals,
                          const common::valuesd& offsets);

  const geometry::points3d& new_points() const;
  const geometry::vectors3d& new_normals() const;

 private:
  geometry::points3d new_points_;
  geometry::vectors3d new_normals_;
};

}  // namespace polatory::point_cloud
