// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "polatory/interpolation/rbf_direct_solver.hpp"
#include "polatory/rbf/rbf_base.hpp"
#include "polatory/polynomial/basis_base.hpp"

namespace polatory {
namespace preconditioner {

template <typename Floating>
class coarse_grid {
  const rbf::rbf_base& rbf;
  const int poly_degree;
  const std::vector<size_t> point_indices;

  const size_t n_points;
  const size_t n_points_full;
  const size_t n_polynomials;

  std::vector<Eigen::Vector3d> points;

  std::unique_ptr<interpolation::rbf_direct_solver<Floating>> solver;

  Eigen::VectorXd solution;

public:
  coarse_grid(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
              const std::vector<Eigen::Vector3d>& points_full,
              const std::vector<size_t>& point_indices)
    : rbf(rbf)
    , poly_degree(poly_degree)
    , point_indices(point_indices)
    , n_points(point_indices.size())
    , n_points_full(points_full.size())
    , n_polynomials(polynomial::basis_base::basis_size(poly_dimension, poly_degree)) {
    for (auto idx : point_indices) {
      points.push_back(points_full[idx]);
    }

    solver = std::make_unique<interpolation::rbf_direct_solver<Floating>>(rbf, poly_dimension, poly_degree, points);
  }

  template <typename Derived>
  void set_solution_to(Eigen::MatrixBase<Derived>& weights_full) const {
    assert(weights_full.size() == n_points_full + n_polynomials);

    for (size_t i = 0; i < n_points; i++) {
      weights_full(point_indices[i]) = solution(i);
    }

    weights_full.tail(n_polynomials) = solution.tail(n_polynomials);
  }

  const std::vector<size_t>& indices() const {
    return point_indices;
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
