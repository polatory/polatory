// Copyright (c) 2016, GSI and The Polatory Authors.

#include "polatory/point_cloud/kdtree.hpp"

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "polatory/third_party/flann/flann.hpp"

namespace polatory {
namespace point_cloud {

class kdtree::impl {
  typedef flann::L2<double> Dist;

  flann::SearchParams params_knn;
  flann::SearchParams params_radius;
  std::unique_ptr<flann::Index<Dist>> flann_index;

public:
  explicit impl(const std::vector<Eigen::Vector3d>& points) {
    flann::Matrix<double> points_mat(const_cast<double *>(points[0].data()), points.size(), 3);

    flann_index = std::make_unique<flann::Index<Dist>>(points_mat, flann::KDTreeSingleIndexParams());
    flann_index->buildIndex();
  }

  int knn_search(const Eigen::Vector3d& point, int k,
                 std::vector<size_t>& indices, std::vector<double>& distances) const {
    indices.resize(k);
    distances.resize(k);

    flann::Matrix<double> point_mat(const_cast<double *>(point.data()), 1, 3);
    flann::Matrix<size_t> indices_mat(indices.data(), 1, k);
    flann::Matrix<double> dists_mat(distances.data(), 1, k);

    return flann_index->knnSearch(point_mat, indices_mat, dists_mat, k, params_knn);
  }

  int radius_search(const Eigen::Vector3d& point, double radius,
                    std::vector<size_t>& indices, std::vector<double>& distances) const {
    flann::Matrix<double> point_mat(const_cast<double *>(point.data()), 1, 3);
    std::vector<std::vector<size_t>> indices_v(1);
    std::vector<std::vector<double>> dists_v(1);

    int found = flann_index->radiusSearch(point_mat, indices_v, dists_v, radius, params_radius);

    indices = indices_v[0];
    distances = dists_v[0];

    return found;
  }

  void set_exact_search() {
    params_knn.checks = flann::FLANN_CHECKS_UNLIMITED;
    params_radius.checks = flann::FLANN_CHECKS_UNLIMITED;
  }
};

kdtree::kdtree(const std::vector<Eigen::Vector3d>& points)
  : pimpl(std::make_unique<impl>(points)) {
}

kdtree::~kdtree() = default;

int kdtree::knn_search(const Eigen::Vector3d& point, int k,
                       std::vector<size_t>& indices, std::vector<double>& distances) const {
  return pimpl->knn_search(point, k, indices, distances);
}

int kdtree::radius_search(const Eigen::Vector3d& point, double radius,
                          std::vector<size_t>& indices, std::vector<double>& distances) const {
  return pimpl->radius_search(point, radius, indices, distances);
}

void kdtree::set_exact_search() const {
  pimpl->set_exact_search();
}

} // namespace point_cloud
} // namespace polatory
