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

NormalEstimator::NormalEstimator(const geometry::Points3& points)
    : n_points_(points.rows()), points_(points), tree_(points) {}

NormalEstimator& NormalEstimator::estimate_with_knn(Index k) & {
  return estimate_with_knn(std::vector<Index>{k});
}

NormalEstimator& NormalEstimator::estimate_with_knn(const std::vector<Index>& ks) & {
  if (ks.empty()) {
    throw std::runtime_error("ks must not be empty");
  }

  if (std::any_of(ks.begin(), ks.end(), [](auto k) { return k < 3; })) {
    throw std::runtime_error("k must be greater than or equal to 3");
  }

  normals_ = geometry::Points3::Zero(n_points_, 3);
  plane_factors_ = VecX::Zero(n_points_);

  if (n_points_ < 3) {
    estimated_ = true;
    return *this;
  }

  std::vector<Index> nn_indices;
  std::vector<double> nn_distances;

  std::vector<Index> ks_sorted(ks);
  std::sort(ks_sorted.rbegin(), ks_sorted.rend());
  auto k_max = ks_sorted.front();

  std::vector<double> plane_factors;
  std::vector<geometry::Vector3> plane_normals;

#pragma omp parallel for private(nn_indices, nn_distances, plane_factors, plane_normals)
  for (Index i = 0; i < n_points_; i++) {
    geometry::Point3 p = points_.row(i);
    tree_.knn_search(p, k_max, nn_indices, nn_distances);

    plane_factors.clear();
    plane_normals.clear();
    for (auto k : ks_sorted) {
      nn_indices.resize(k);
      PlaneEstimator est(points_(nn_indices, Eigen::all));
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

NormalEstimator& NormalEstimator::estimate_with_radius(double radius) & {
  return estimate_with_radius(std::vector<double>{radius});
}

NormalEstimator& NormalEstimator::estimate_with_radius(const std::vector<double>& radii) & {
  if (radii.empty()) {
    throw std::runtime_error("radii must not be empty");
  }

  if (std::any_of(radii.begin(), radii.end(), [](auto radius) { return !(radius > 0.0); })) {
    throw std::runtime_error("radius must be positive");
  }

  normals_ = geometry::Points3::Zero(n_points_, 3);
  plane_factors_ = VecX::Zero(n_points_);

  std::vector<Index> nn_indices;
  std::vector<double> nn_distances;

  std::vector<double> radii_sorted(radii);
  std::sort(radii_sorted.rbegin(), radii_sorted.rend());
  auto radius_max = radii_sorted.front();

  std::vector<double> plane_factors;
  std::vector<geometry::Vector3> plane_normals;

#pragma omp parallel for private(nn_indices, nn_distances, plane_factors, plane_normals)
  for (Index i = 0; i < n_points_; i++) {
    geometry::Point3 p = points_.row(i);
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
      PlaneEstimator est(points_(nn_indices, Eigen::all));
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

NormalEstimator& NormalEstimator::filter_by_plane_factor(double threshold) & {
  throw_if_not_estimated();

  for (Index i = 0; i < n_points_; i++) {
    if (plane_factors_(i) < threshold) {
      normals_.row(i).setZero();
    }
  }

  return *this;
}

NormalEstimator& NormalEstimator::orient_toward_direction(const geometry::Vector3& direction) & {
  throw_if_not_estimated();

#pragma omp parallel for
  for (Index i = 0; i < n_points_; i++) {
    auto n = normals_.row(i);
    if (n.dot(direction) < 0.0) {
      n = -n;
    }
  }

  return *this;
}

NormalEstimator& NormalEstimator::orient_toward_point(const geometry::Point3& point) & {
  throw_if_not_estimated();

#pragma omp parallel for
  for (Index i = 0; i < n_points_; i++) {
    auto n = normals_.row(i);
    auto p = points_.row(i);
    if (n.dot(point - p) < 0.0) {
      n = -n;
    }
  }

  return *this;
}

struct WeightedPair {
  bool operator<(const WeightedPair& rhs) const { return weight < rhs.weight; }

  Index first{};
  Index second{};
  double weight{};
};

NormalEstimator& NormalEstimator::orient_closed_surface(Index k) & {
  throw_if_not_estimated();

  geometry::Vector3 seed_point_direction{-geometry::Vector3::UnitY()};
  Index n_connected_components{};

  std::vector<bool> oriented(n_points_, false);
  for (Index i = 0; i < n_points_; i++) {
    if (normals_.row(i).isZero()) {
      oriented.at(i) = true;
    }
  }

  auto it = oriented.begin();
  std::vector<Index> connected_component;
  std::priority_queue<WeightedPair> queue;
  std::vector<Index> nn_indices;
  std::vector<double> nn_distances;
  while (true) {
    it = std::find(it, oriented.end(), false);
    if (it == oriented.end()) {
      break;
    }

    connected_component.clear();

    {
      auto cur = static_cast<Index>(std::distance(oriented.begin(), it));
      oriented.at(cur) = true;
      connected_component.push_back(cur);

      auto p_cur = points_.row(cur);
      auto n_cur = normals_.row(cur);

      tree_.knn_search(p_cur, k, nn_indices, nn_distances);
      for (auto next : nn_indices) {
        if (oriented.at(next)) {
          continue;
        }

        auto p_next = points_.row(next);
        auto n_next = normals_.row(next);

        auto w_next = std::abs(n_next.dot(n_cur)) / (p_next - p_cur).norm();
        queue.emplace(cur, next, w_next);
      }
    }

    while (!queue.empty()) {
      auto [prev, cur, w_cur] = queue.top();
      queue.pop();
      if (oriented.at(cur)) {
        continue;
      }

      auto n_prev = normals_.row(prev);
      auto p_cur = points_.row(cur);
      auto n_cur = normals_.row(cur);

      if (n_cur.dot(n_prev) < 0.0) {
        n_cur *= -1.0;
      }
      oriented.at(cur) = true;
      connected_component.push_back(cur);

      tree_.knn_search(p_cur, k, nn_indices, nn_distances);
      for (auto next : nn_indices) {
        if (oriented.at(next)) {
          continue;
        }

        auto p_next = points_.row(next);
        auto n_next = normals_.row(next);

        auto w_next = std::abs(n_next.dot(n_cur)) / (p_next - p_cur).norm();
        queue.emplace(cur, next, w_next);
      }
    }

    auto seed_it = std::max_element(connected_component.begin(), connected_component.end(),
                                    [&](auto i, auto j) {
                                      return points_.row(i).dot(seed_point_direction) <
                                             points_.row(j).dot(seed_point_direction);
                                    });
    if (normals_.row(*seed_it).dot(seed_point_direction) < 0.0) {
      normals_(connected_component, Eigen::all) *= -1.0;
    }

    n_connected_components++;
  }

  std::cout << "Number of connected components: " << n_connected_components << std::endl;

  return *this;
}

}  // namespace polatory::point_cloud
