#pragma once

#include <memory>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::point_cloud {

class kdtree {
 public:
  kdtree(const geometry::points3d& points, bool use_exact_search);

  ~kdtree();

  kdtree(const kdtree&) = delete;
  kdtree(kdtree&&) = delete;
  kdtree& operator=(const kdtree&) = delete;
  kdtree& operator=(kdtree&&) = delete;

  void knn_search(const geometry::point3d& point, index_t k, std::vector<index_t>& indices,
                  std::vector<double>& distances) const;

  void radius_search(const geometry::point3d& point, double radius, std::vector<index_t>& indices,
                     std::vector<double>& distances) const;

 private:
  class impl;

  std::unique_ptr<impl> pimpl_;
};

}  // namespace polatory::point_cloud
