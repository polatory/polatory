// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "../common/vector_range_view.hpp"
#include "../geometry/bbox3.hpp"
#include "../polynomial/basis_base.hpp"
#include "../rbf/rbf_base.hpp"
#include "rbf_evaluator.hpp"

namespace polatory {
namespace interpolation {

class rbf_residual_evaluator {
  const int chunk_size = 1024;

  const double nugget;
  const size_t n_polynomials;

  std::vector<Eigen::Vector3d> points;
  size_t n_points;

  std::unique_ptr<rbf_evaluator<>> evaluator;

public:
  template <typename Container>
  rbf_residual_evaluator(const rbf::rbf_base& rbf, int poly_degree,
                         const Container& in_points)
    : nugget(rbf.nugget())
    , n_polynomials(polynomial::basis_base::dimension(poly_degree))
    , points(in_points.begin(), in_points.end())
    , n_points(in_points.size()) {
    evaluator = std::make_unique<rbf_evaluator<>>(rbf, poly_degree, points);
  }

  rbf_residual_evaluator(const rbf::rbf_base& rbf, int poly_degree,
                         int tree_height, const geometry::bbox3d& bbox)
    : nugget(rbf.nugget())
    , n_polynomials(polynomial::basis_base::dimension(poly_degree))
    , n_points(0) {
    evaluator = std::make_unique<rbf_evaluator<>>(rbf, poly_degree, tree_height, bbox);
  }

  template <typename Derived, typename Derived2>
  bool converged(const Eigen::MatrixBase<Derived>& values, const Eigen::MatrixBase<Derived2>& weights,
                 double absolute_tolerance) const {
    assert(values.size() == n_points);
    assert(weights.size() == n_points + n_polynomials);

    evaluator->set_weights(weights);

    double max_residual = 0.0;
    for (size_t i = 0; i < n_points / chunk_size + 1; i++) {
      auto begin = i * chunk_size;
      auto end = std::min(n_points, begin + chunk_size);
      if (begin == end) break;

      evaluator->set_field_points(common::make_range_view(points, begin, end));
      auto fit = evaluator->evaluate();

      for (size_t j = 0; j < end - begin; j++) {
        auto res = std::abs(values(begin + j) - fit(j));
        if (res >= absolute_tolerance + std::abs(nugget * weights(begin + j)))
          return false;

        max_residual = std::max(max_residual, res);
      }
    }

    std::cout << "Maximum residual: " << max_residual << std::endl;

    return true;
  }

  template <typename Container>
  void set_points(const Container& in_points) {
    n_points = in_points.size();
    points = std::vector<Eigen::Vector3d>(in_points.begin(), in_points.end());

    evaluator->set_source_points(in_points);
  }
};

} // namespace interpolation
} // namespace polatory
