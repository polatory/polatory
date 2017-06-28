// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

namespace polatory {
namespace point_cloud {

class kdtree {
public:
   explicit kdtree(const std::vector<Eigen::Vector3d>& points);

   ~kdtree();

   int knn_search(const Eigen::Vector3d& point, int k,
      std::vector<size_t>& indices, std::vector<double>& distances) const;

   int radius_search(const Eigen::Vector3d& point, double radius,
      std::vector<size_t>& indices, std::vector<double>& distances) const;

   void set_exact_search() const;

private:
   class impl;

   std::unique_ptr<impl> pimpl;
};

} // namespace point_cloud
} // namespace polatory
