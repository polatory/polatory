// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "../interpolation/rbf_direct_solver.hpp"
#include "../rbf/rbf_base.hpp"

namespace polatory {
namespace preconditioner {

template <typename Floating>
class fine_grid {
  const std::vector<size_t> point_indices;
  const std::vector<bool> inner_point;

  const size_t n_points;
  mutable size_t n_points_full;

  std::unique_ptr<interpolation::rbf_direct_solver<Floating>> solver;

  Eigen::VectorXd solution;

public:
  fine_grid(const rbf::rbf_base& rbf,
            const std::vector<size_t>& point_indices,
            const std::vector<bool>& inner_point)
    : point_indices(point_indices)
    , inner_point(inner_point)
    , n_points(point_indices.size())
    , n_points_full(0) {
    solver = std::make_unique<interpolation::rbf_direct_solver<Floating>>(rbf, -1, point_indices.size());
  }

  fine_grid(const rbf::rbf_base& rbf,
            const std::vector<Eigen::Vector3d>& points_full,
            const std::vector<size_t>& point_indices,
            const std::vector<bool>& inner_point)
    : point_indices(point_indices)
    , inner_point(inner_point)
    , n_points(point_indices.size())
    , n_points_full(points_full.size()) {
    std::vector<Eigen::Vector3d> points;
    for (auto idx : point_indices) {
      points.push_back(points_full[idx]);
    }

    solver = std::make_unique<interpolation::rbf_direct_solver<Floating>>(rbf, -1, points);
  }

  void clear() const {
    solver->clear();
  }

  const std::vector<size_t>& indices() const {
    return point_indices;
  }

  void setup(const std::vector<Eigen::Vector3d>& points_full) const {
    n_points_full = points_full.size();

    std::vector<Eigen::Vector3d> points;
    for (auto idx : point_indices) {
      points.push_back(points_full[idx]);
    }

    solver->setup(points);
  }

  template <typename Derived>
  void set_solution_to(Eigen::MatrixBase<Derived>& weights_full) const {
    assert(weights_full.size() == n_points_full);

    for (size_t i = 0; i < n_points; i++) {
      if (inner_point[i])
        weights_full(point_indices[i]) = solution(i);
    }
  }

  size_t size() const {
    return n_points;
  }

  template <typename Derived>
  void solve(const Eigen::MatrixBase<Derived>& values_full) {
    assert(values_full.size() == n_points_full);

    Eigen::VectorXd values(n_points);
    for (size_t i = 0; i < n_points; i++) {
      values(i) = values_full(point_indices[i]);
    }

    solution = solver->solve(values);
  }
};

} // namespace preconditioner
} // namespace polatory
