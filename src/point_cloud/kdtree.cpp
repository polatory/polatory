// Copyright (c) 2016, GSI and The Polatory Authors.

#include "polatory/point_cloud/kdtree.hpp"

#include <cmath>

#include <flann/flann.hpp>

#include "polatory/common/exception.hpp"

namespace polatory {
namespace point_cloud {

class kdtree::impl {
  using FlannIndex = flann::Index<flann::L2<double>>;

public:
  explicit impl(const std::vector<Eigen::Vector3d>& points) {
    flann::Matrix<double> points_mat(const_cast<double *>(points[0].data()), points.size(), 3);

    flann_index_ = std::make_unique<FlannIndex>(points_mat, flann::KDTreeSingleIndexParams());
    flann_index_->buildIndex();
  }

  void knn_search(const Eigen::Vector3d& point, int k,
                  std::vector<size_t>& indices, std::vector<double>& distances) const {
    flann::Matrix<double> point_mat(const_cast<double *>(point.data()), 1, 3);
    std::vector<std::vector<size_t>> indices_v;
    std::vector<std::vector<double>> dists_v;

    flann_index_->knnSearch(point_mat, indices_v, dists_v, k, params_knn_);

    indices = indices_v[0];
    distances = dists_v[0];

    for (auto& d : distances) {
      d = std::sqrt(d);
    }
  }

  void radius_search(const Eigen::Vector3d& point, double radius,
                     std::vector<size_t>& indices, std::vector<double>& distances) const {
    flann::Matrix<double> point_mat(const_cast<double *>(point.data()), 1, 3);
    std::vector<std::vector<size_t>> indices_v;
    std::vector<std::vector<double>> dists_v;

    flann_index_->radiusSearch(point_mat, indices_v, dists_v, radius * radius, params_radius_);

    indices = indices_v[0];
    distances = dists_v[0];

    for (auto& d : distances) {
      d = std::sqrt(d);
    }
  }

  void set_exact_search() {
    params_knn_.checks = flann::FLANN_CHECKS_UNLIMITED;
    params_radius_.checks = flann::FLANN_CHECKS_UNLIMITED;
  }

private:
  flann::SearchParams params_knn_;
  flann::SearchParams params_radius_;
  std::unique_ptr<FlannIndex> flann_index_;
};

kdtree::kdtree(const std::vector<Eigen::Vector3d>& points)
  : pimpl_(std::make_unique<impl>(points)) {
}

kdtree::~kdtree() = default;

void kdtree::knn_search(const Eigen::Vector3d& point, int k,
                        std::vector<size_t>& indices, std::vector<double>& distances) const {
  if (k <= 0)
    throw common::invalid_argument("k > 0");

  pimpl_->knn_search(point, k, indices, distances);
}

void kdtree::radius_search(const Eigen::Vector3d& point, double radius,
                           std::vector<size_t>& indices, std::vector<double>& distances) const {
  if (radius <= 0.0)
    throw common::invalid_argument("radius > 0.0");

  pimpl_->radius_search(point, radius, indices, distances);
}

void kdtree::set_exact_search() const {
  pimpl_->set_exact_search();
}

} // namespace point_cloud
} // namespace polatory
