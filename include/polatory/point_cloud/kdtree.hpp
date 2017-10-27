// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

namespace polatory {
namespace point_cloud {

class kdtree {
public:
  kdtree(const std::vector<Eigen::Vector3d>& points, bool use_exact_search);

  ~kdtree();

  void knn_search(const Eigen::Vector3d& point, int k,
                  std::vector<size_t>& indices, std::vector<double>& distances) const;

  void radius_search(const Eigen::Vector3d& point, double radius,
                     std::vector<size_t>& indices, std::vector<double>& distances) const;

private:
  class impl;

  std::unique_ptr<impl> pimpl_;
};

} // namespace point_cloud
} // namespace polatory
