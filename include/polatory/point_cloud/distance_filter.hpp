#pragma once

#include <Eigen/Core>
#include <numeric>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace polatory::point_cloud {

template <int Dim>
class distance_filter {
  using Point = geometry::pointNd<Dim>;
  using Points = geometry::pointsNd<Dim>;

 public:
  distance_filter(const Points& points, double distance)
      : distance_filter(points, distance, trivial_indices(points.rows())) {}

  distance_filter(const Points& points, double distance, const std::vector<index_t>& indices)
      : n_points_(points.rows()) {
    if (distance <= 0.0) {
      throw std::invalid_argument("distance must be greater than 0.0.");
    }

    kdtree tree(points);

    std::unordered_set<index_t> indices_to_remove;

    std::vector<index_t> nn_indices;
    std::vector<double> nn_distances;

    for (auto i : indices) {
      if (indices_to_remove.contains(i)) {
        continue;
      }

      Point p = points.row(i);
      tree.radius_search(p, distance, nn_indices, nn_distances);

      for (auto j : nn_indices) {
        if (j != i) {
          indices_to_remove.insert(j);
        }
      }
    }

    for (auto i : indices) {
      if (!indices_to_remove.contains(i)) {
        filtered_indices_.push_back(i);
      }
    }
  }

  template <class Derived>
  auto operator()(const Eigen::MatrixBase<Derived>& m) {
    if (m.rows() != n_points_) {
      throw std::invalid_argument("m.rows() must match with the original points.");
    }

    // Use .eval() to prevent memory corruption caused if the result is being assigned
    // back to the input matrix.
    return m(filtered_indices_, Eigen::all).eval();
  }

  template <class Derived, class... Args>
  auto operator()(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
    return std::make_tuple(operator()(m), operator()(std::forward<Args>(args))...);
  }

  const std::vector<index_t>& filtered_indices() const { return filtered_indices_; }

 private:
  static std::vector<index_t> trivial_indices(index_t n_points) {
    std::vector<index_t> indices(n_points);
    std::iota(indices.begin(), indices.end(), index_t{0});
    return indices;
  }

  const index_t n_points_;

  std::vector<index_t> filtered_indices_;
};

}  // namespace polatory::point_cloud
