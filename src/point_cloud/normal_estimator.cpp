#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/point_cloud/normal_estimator.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>
#include <queue>
#include <stdexcept>

namespace polatory::point_cloud {

normal_estimator::normal_estimator(const geometry::points3d& points)
    : n_points_(points.rows()), points_(points), tree_(points) {}

normal_estimator& normal_estimator::estimate_with_knn(index_t k) {
  return estimate_with_knn(std::vector<index_t>{k});
}

normal_estimator& normal_estimator::estimate_with_knn(const std::vector<index_t>& ks) {
  if (ks.empty()) {
    throw std::runtime_error("ks must not be empty");
  }

  if (std::any_of(ks.begin(), ks.end(), [](auto k) { return k < 3; })) {
    throw std::runtime_error("k must be greater than or equal to 3");
  }

  normals_ = geometry::points3d::Zero(n_points_, 3);
  plane_factors_ = vectord::Zero(n_points_);

  if (n_points_ < 3) {
    estimated_ = true;
    return *this;
  }

  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

  std::vector<index_t> ks_sorted(ks);
  std::sort(ks_sorted.rbegin(), ks_sorted.rend());
  auto k_max = ks_sorted.front();

  std::vector<double> plane_factors;
  std::vector<geometry::vector3d> plane_normals;

#pragma omp parallel for private(nn_indices, nn_distances, plane_factors, plane_normals)
  for (index_t i = 0; i < n_points_; i++) {
    geometry::point3d p = points_.row(i);
    tree_.knn_search(p, k_max, nn_indices, nn_distances);

    plane_factors.clear();
    plane_normals.clear();
    for (auto k : ks_sorted) {
      nn_indices.resize(k);
      plane_estimator est(points_(nn_indices, Eigen::all));
      plane_factors.push_back(est.plane_factor());
      plane_normals.push_back(est.plane_normal());
    }

    auto best = std::distance(plane_factors.begin(),
                              std::max_element(plane_factors.begin(), plane_factors.end()));

    normals_.row(i) = plane_normals.at(best);
    plane_factors_(i) = plane_factors.at(best);
  }

  estimated_ = true;
  return *this;
}

normal_estimator& normal_estimator::estimate_with_radius(double radius) {
  return estimate_with_radius(std::vector<double>{radius});
}

normal_estimator& normal_estimator::estimate_with_radius(const std::vector<double>& radii) {
  if (radii.empty()) {
    throw std::runtime_error("radii must not be empty");
  }

  if (std::any_of(radii.begin(), radii.end(), [](auto radius) { return !(radius > 0.0); })) {
    throw std::runtime_error("radius must be positive");
  }

  normals_ = geometry::points3d::Zero(n_points_, 3);
  plane_factors_ = vectord::Zero(n_points_);

  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

  std::vector<double> radii_sorted(radii);
  std::sort(radii_sorted.rbegin(), radii_sorted.rend());
  auto radius_max = radii_sorted.front();

  std::vector<double> plane_factors;
  std::vector<geometry::vector3d> plane_normals;

#pragma omp parallel for private(nn_indices, nn_distances, plane_factors, plane_normals)
  for (index_t i = 0; i < n_points_; i++) {
    geometry::point3d p = points_.row(i);
    tree_.radius_search(p, radius_max, nn_indices, nn_distances);

    if (nn_indices.size() < 3) {
      continue;
    }

    plane_factors.clear();
    plane_normals.clear();
    for (auto radius : radii_sorted) {
      auto it = std::upper_bound(nn_distances.begin(), nn_distances.end(), radius);
      auto k = std::distance(nn_distances.begin(), it);
      nn_indices.resize(k);
      nn_distances.resize(k);
      plane_estimator est(points_(nn_indices, Eigen::all));
      plane_factors.push_back(est.plane_factor());
      plane_normals.push_back(est.plane_normal());
    }

    auto best = std::distance(plane_factors.begin(),
                              std::max_element(plane_factors.begin(), plane_factors.end()));

    normals_.row(i) = plane_normals.at(best);
    plane_factors_(i) = plane_factors.at(best);
  }

  estimated_ = true;
  return *this;
}

normal_estimator& normal_estimator::filter_by_plane_factor(double threshold) {
  throw_if_not_estimated();

  for (index_t i = 0; i < n_points_; i++) {
    if (plane_factors_(i) < threshold) {
      normals_.row(i).setZero();
    }
  }

  return *this;
}

normal_estimator& normal_estimator::orient_toward_direction(const geometry::vector3d& direction) {
  throw_if_not_estimated();

#pragma omp parallel for
  for (index_t i = 0; i < n_points_; i++) {
    auto n = normals_.row(i);
    if (n.dot(direction) < 0.0) {
      n = -n;
    }
  }

  return *this;
}

normal_estimator& normal_estimator::orient_toward_point(const geometry::point3d& point) {
  throw_if_not_estimated();

#pragma omp parallel for
  for (index_t i = 0; i < n_points_; i++) {
    auto n = normals_.row(i);
    auto p = points_.row(i);
    if (n.dot(point - p) < 0.0) {
      n = -n;
    }
  }

  return *this;
}

class weighted_pair {
 public:
  weighted_pair(index_t first, index_t second, double weight)
      : first_(first), second_(second), weight_(weight) {}

  bool operator<(const weighted_pair& rhs) const { return weight_ < rhs.weight_; }

  index_t first() const { return first_; }

  index_t second() const { return second_; }

  double weight() const { return weight_; }

 private:
  index_t first_;
  index_t second_;
  double weight_;
};

normal_estimator& normal_estimator::orient_closed_surface(index_t k) {
  throw_if_not_estimated();

  auto bbox = geometry::bbox3d::from_points(points_);
  auto center = bbox.center();
  geometry::point3d p_outer{center(0), bbox.min()(1) - 1.0, center(2)};

  std::vector<bool> oriented(n_points_, false);
  for (index_t i = 0; i < n_points_; i++) {
    if (normals_.row(i).isZero()) {
      oriented.at(i) = true;
    }
  }

  std::vector<index_t> indices(n_points_);
  std::iota(indices.begin(), indices.end(), 0);
  {
    std::vector<double> distances(n_points_);
    for (index_t i = 0; i < n_points_; i++) {
      geometry::point3d p = points_.row(i);
      distances.at(i) = (p_outer - p).norm();
    }
    std::sort(indices.begin(), indices.end(),
              [&](auto i, auto j) { return distances.at(i) < distances.at(j); });
  }
  auto indices_it = indices.begin();

  std::priority_queue<weighted_pair> queue;
  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

  index_t n_connected_components{};
  while (std::find(oriented.begin(), oriented.end(), false) != oriented.end()) {
    while (oriented.at(*indices_it)) {
      indices_it++;
    }

    auto i_closest = *indices_it;
    geometry::point3d p_closest = points_.row(i_closest);
    if (normals_.row(i_closest).dot(p_outer - p_closest) < 0.0) {
      normals_.row(i_closest) *= -1.0;
    }
    oriented.at(i_closest) = true;

    tree_.knn_search(p_closest, k, nn_indices, nn_distances);
    for (auto j : nn_indices) {
      if (oriented.at(j)) {
        continue;
      }

      auto weight = std::abs(normals_.row(i_closest).dot(normals_.row(j))) /
                    (p_closest - points_.row(j)).norm();
      queue.emplace(i_closest, j, weight);
    }

    while (!queue.empty()) {
      auto pair = queue.top();
      queue.pop();

      auto i = pair.first();
      auto j = pair.second();
      if (oriented.at(j)) {
        continue;
      }

      if (normals_.row(i).dot(normals_.row(j)) < 0.0) {
        normals_.row(j) *= -1.0;
      }
      oriented.at(j) = true;

      geometry::point3d p = points_.row(j);
      tree_.knn_search(p, k, nn_indices, nn_distances);
      for (auto kk : nn_indices) {
        if (oriented.at(kk)) {
          continue;
        }

        auto weight =
            std::abs(normals_.row(j).dot(normals_.row(kk))) / (p - points_.row(kk)).norm();
        queue.emplace(j, kk, weight);
      }
    }

    n_connected_components++;
  }

  std::cout << "Number of connected components: " << n_connected_components << std::endl;

  return *this;
}

}  // namespace polatory::point_cloud
