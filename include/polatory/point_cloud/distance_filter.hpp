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
class DistanceFilter {
  using Point = geometry::Point<Dim>;
  using Points = geometry::Points<Dim>;

 public:
  explicit DistanceFilter(const Points& points) : points_(points), tree_(points) {}

  DistanceFilter& filter(double distance) {
    return filter(distance, trivial_indices(points_.rows()));
  }

  DistanceFilter& filter(double distance, const std::vector<Index>& indices) {
    if (!(distance >= 0.0)) {
      throw std::invalid_argument("distance must be non-negative");
    }

    std::unordered_set<Index> indices_to_remove;

    std::vector<Index> nn_indices;
    std::vector<double> nn_distances;
    for (auto i : indices) {
      if (indices_to_remove.contains(i)) {
        continue;
      }

      auto p = points_.row(i);
      tree_.radius_search(p, distance, nn_indices, nn_distances);

      for (auto j : nn_indices) {
        if (j != i) {
          indices_to_remove.insert(j);
        }
      }
    }

    filtered_indices_.clear();
    for (auto i : indices) {
      if (!indices_to_remove.contains(i)) {
        filtered_indices_.push_back(i);
      }
    }

    filtered_ = true;
    return *this;
  }

  template <class Derived>
  auto operator()(const Eigen::MatrixBase<Derived>& m) {
    throw_if_not_filtered();

    if (m.rows() != points_.rows()) {
      throw std::invalid_argument("m.rows() must match with the original points");
    }

    // Use .eval() to prevent memory corruption caused if the result is being assigned
    // back to the input matrix.
    return m(filtered_indices_, Eigen::all).eval();
  }

  template <class Derived, class... Args>
  auto operator()(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
    return std::make_tuple(operator()(m), operator()(std::forward<Args>(args))...);
  }

  const std::vector<Index>& filtered_indices() const {
    throw_if_not_filtered();

    return filtered_indices_;
  }

 private:
  void throw_if_not_filtered() const {
    if (!filtered_) {
      throw std::runtime_error("points have not been filtered");
    }
  }

  static std::vector<Index> trivial_indices(Index n_points) {
    std::vector<Index> indices(n_points);
    std::iota(indices.begin(), indices.end(), Index{0});
    return indices;
  }

  const Points& points_;
  const KdTree<Dim> tree_;
  bool filtered_{};
  std::vector<Index> filtered_indices_;
};

}  // namespace polatory::point_cloud
