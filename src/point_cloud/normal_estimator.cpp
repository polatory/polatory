#include <polatory/point_cloud/normal_estimator.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>
#include <stdexcept>
#include <cmath>
#include <queue>
namespace polatory::point_cloud {

normal_estimator::normal_estimator(const geometry::points3d& points)
    : n_points_(points.rows()), points_(points), tree_(points, true) {}

normal_estimator& normal_estimator::estimate_with_knn(index_t k, double plane_factor_threshold) {
  normals_ = geometry::vectors3d(n_points_, 3);

  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

#pragma omp parallel for private(nn_indices, nn_distances)
  for (index_t i = 0; i < n_points_; i++) {
    geometry::point3d p = points_.row(i);
    tree_.knn_search(p, k, nn_indices, nn_distances);

    normals_.row(i) = estimate_impl(nn_indices, plane_factor_threshold);
  }

  return *this;
}

normal_estimator& normal_estimator::estimate_with_radius(double radius,
                                                         double plane_factor_threshold) {
  normals_ = geometry::vectors3d(n_points_, 3);

  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

#pragma omp parallel for private(nn_indices, nn_distances)
  for (index_t i = 0; i < n_points_; i++) {
    geometry::point3d p = points_.row(i);
    tree_.radius_search(p, radius, nn_indices, nn_distances);

    normals_.row(i) = estimate_impl(nn_indices, plane_factor_threshold);
  }

  return *this;
}

geometry::vectors3d normal_estimator::estimate_with_knn_closed_surface(index_t k, point3d p_outside, double plane_factor_threshold) {
  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;
  normals_ = geometry::vectors3d(n_points_, 3);
  // 1. Estimate per-point normal(not consistently oriented at this point)

  for (index_t i = 0; i < n_points_; i++) {
    normals_.row(i) = geometry::vector3d::Zero();
  }

  for (index_t i = 0; i < n_points_; i++) {
    geometry::point3d p = points_.row(i);
    tree_.knn_search(p, k, nn_indices, nn_distances); 
    
    normals_.row(i) = estimate_impl(nn_indices, plane_factor_threshold);

  }
  

  // 2. For each connected component of the point cloud(in terms of the nearest neighbor search):
  std::vector<bool> visited(n_points_, false);
  std::vector<std::vector<index_t>> components;
  for (index_t i = 0; i < n_points_; i++) {
    if (visited[i]) continue;
    std::vector<index_t> component;
    
   /* geometry::point3d point = points_.row(i);
    tree_.knn_search(point, k, nn_indices, nn_distances);*/

    std::queue<index_t> q;

    // using the bfs to get the component
    q.push(i);
    while (q.size()) {
      index_t t = q.front();
      q.pop();
      if (visited[t]) continue;
      visited[t] = true;
      component.push_back(t);
      geometry::point3d point = points_.row(t);
      tree_.knn_search(point, k, nn_indices, nn_distances);

      for (auto v : nn_indices) {
        if (visited[v]) continue;
        q.push(v);
      }
    }
    components.push_back(component);


    // 3. Find the closest point in the component to the p_outside
    index_t closest_index = i;
    double closest_distance = std::numeric_limits<double>::max();
    for (auto x : component) {
      geometry::point3d p = points_.row(x);
      double distance = std::sqrt((p[0] - p_outside[0]) * (p[0] - p_outside[0]) +
                                  (p[1] - p_outside[1]) * (p[1] - p_outside[1]) +
                                  (p[2] - p_outside[2]) * (p[2] - p_outside[2]));
      if (distance < closest_distance) {
        closest_distance = distance;
        closest_index = x;
      }
    }
  
    // 4. Orient the normal of the closest point so that it aligns with the vector p_outside - p.

    auto n = normals_.row(closest_index);
    geometry::point3d P = points_.row(closest_index);
    point3d t = {P[0] - p_outside[0], P[1] - p_outside[1], P[2] - p_outside[2]};
    if (n.dot(t) > 0.0) {
      n = -n;
    }

    // 5. Propagate the orientation of normals using the nearest neighbor search
    for (auto x : component) {
      if (x == closest_index) continue;
      auto m = normals_.row(x);
      if (m.dot(n) < 0) {
        m = -m;
      }
    }
  }
  return normals_;
}

geometry::vectors3d normal_estimator::orient_by_outward_vector(const geometry::vector3d& v) {
  if (n_points_ > 0 && normals_.rows() == 0) {
    throw std::runtime_error("Normals have not been estimated yet.");
  }

#pragma omp parallel for
  for (index_t i = 0; i < n_points_; i++) {
    auto n = normals_.row(i);
    if (n.dot(v) < 0.0) {
      n = -n;
    }
  }

  return normals_;
}

geometry::vector3d normal_estimator::estimate_impl(const std::vector<index_t>& nn_indices,
                                                   double plane_factor_threshold) const {
  if (nn_indices.size() < 3) {
    return geometry::vector3d::Zero();
  }

  plane_estimator est(points_(nn_indices, Eigen::all));

  if (est.plane_factor() < plane_factor_threshold) {
    return geometry::vector3d::Zero();
  }

  return est.plane_normal();
}

}  // namespace polatory::point_cloud
#include <polatory/point_cloud/normal_estimator.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>
#include <stdexcept>
#include <cmath>
#include <queue>
namespace polatory::point_cloud {

normal_estimator::normal_estimator(const geometry::points3d& points)
    : n_points_(points.rows()), points_(points), tree_(points, true) {}

normal_estimator& normal_estimator::estimate_with_knn(index_t k, double plane_factor_threshold) {
  normals_ = geometry::vectors3d(n_points_, 3);

  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

#pragma omp parallel for private(nn_indices, nn_distances)
  for (index_t i = 0; i < n_points_; i++) {
    geometry::point3d p = points_.row(i);
    tree_.knn_search(p, k, nn_indices, nn_distances);

    normals_.row(i) = estimate_impl(nn_indices, plane_factor_threshold);
  }

  return *this;
}

normal_estimator& normal_estimator::estimate_with_radius(double radius,
                                                         double plane_factor_threshold) {
  normals_ = geometry::vectors3d(n_points_, 3);

  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

#pragma omp parallel for private(nn_indices, nn_distances)
  for (index_t i = 0; i < n_points_; i++) {
    geometry::point3d p = points_.row(i);
    tree_.radius_search(p, radius, nn_indices, nn_distances);

    normals_.row(i) = estimate_impl(nn_indices, plane_factor_threshold);
  }

  return *this;
}

geometry::vectors3d normal_estimator::estimate_with_knn_closed_surface(index_t k, point3d p_outside, double plane_factor_threshold) {
  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;
  normals_ = geometry::vectors3d(n_points_, 3);
  // 1. Estimate per-point normal(not consistently oriented at this point)

  for (index_t i = 0; i < n_points_; i++) {
    normals_.row(i) = geometry::vector3d::Zero();
  }

  for (index_t i = 0; i < n_points_; i++) {
    geometry::point3d p = points_.row(i);
    tree_.knn_search(p, k, nn_indices, nn_distances); 
    
    normals_.row(i) = estimate_impl(nn_indices, plane_factor_threshold);

  }
  

  // 2. For each connected component of the point cloud(in terms of the nearest neighbor search):
  std::vector<bool> visited(n_points_, false);
  std::vector<std::vector<index_t>> components;
  for (index_t i = 0; i < n_points_; i++) {
    if (visited[i]) continue;
    std::vector<index_t> component;
    
   /* geometry::point3d point = points_.row(i);
    tree_.knn_search(point, k, nn_indices, nn_distances);*/

    std::queue<index_t> q;

    // using the bfs to get the component
    q.push(i);
    while (q.size()) {
      index_t t = q.front();
      q.pop();
      if (visited[t]) continue;
      visited[t] = true;
      component.push_back(t);
      geometry::point3d point = points_.row(t);
      tree_.knn_search(point, k, nn_indices, nn_distances);

      for (auto v : nn_indices) {
        if (visited[v]) continue;
        q.push(v);
      }
    }
    components.push_back(component);


    // 3. Find the closest point in the component to the p_outside
    index_t closest_index = i;
    double closest_distance = std::numeric_limits<double>::max();
    for (auto x : component) {
      geometry::point3d p = points_.row(x);
      double distance = std::sqrt((p[0] - p_outside[0]) * (p[0] - p_outside[0]) +
                                  (p[1] - p_outside[1]) * (p[1] - p_outside[1]) +
                                  (p[2] - p_outside[2]) * (p[2] - p_outside[2]));
      if (distance < closest_distance) {
        closest_distance = distance;
        closest_index = x;
      }
    }
  
    // 4. Orient the normal of the closest point so that it aligns with the vector p_outside - p.

    auto n = normals_.row(closest_index);
    geometry::point3d P = points_.row(closest_index);
    point3d t = {P[0] - p_outside[0], P[1] - p_outside[1], P[2] - p_outside[2]};
    if (n.dot(t) > 0.0) {
      n = -n;
    }

    // 5. Propagate the orientation of normals using the nearest neighbor search
    for (auto x : component) {
      if (x == closest_index) continue;
      auto m = normals_.row(x);
      if (m.dot(n) < 0) {
        m = -m;
      }
    }
  }
  return normals_;
}

geometry::vectors3d normal_estimator::orient_by_outward_vector(const geometry::vector3d& v) {
  if (n_points_ > 0 && normals_.rows() == 0) {
    throw std::runtime_error("Normals have not been estimated yet.");
  }

#pragma omp parallel for
  for (index_t i = 0; i < n_points_; i++) {
    auto n = normals_.row(i);
    if (n.dot(v) < 0.0) {
      n = -n;
    }
  }

  return normals_;
}

geometry::vector3d normal_estimator::estimate_impl(const std::vector<index_t>& nn_indices,
                                                   double plane_factor_threshold) const {
  if (nn_indices.size() < 3) {
    return geometry::vector3d::Zero();
  }

  plane_estimator est(points_(nn_indices, Eigen::all));

  if (est.plane_factor() < plane_factor_threshold) {
    return geometry::vector3d::Zero();
  }

  return est.plane_normal();
}

}  // namespace polatory::point_cloud
