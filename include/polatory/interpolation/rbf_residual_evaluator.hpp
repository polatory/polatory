// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "polatory/common/vector_range_view.hpp"
#include "polatory/geometry/bbox3d.hpp"
#include "polatory/polynomial/basis_base.hpp"
#include "polatory/rbf/rbf_base.hpp"
#include "rbf_evaluator.hpp"

namespace polatory {
namespace interpolation {

class rbf_residual_evaluator {
  const int chunk_size = 1024;

public:
  template <class Container>
  rbf_residual_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                         const Container& points)
    : rbf_(rbf)
    , n_polynomials_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , points_(points.begin(), points.end())
    , n_points_(points.size()) {
    evaluator_ = std::make_unique<rbf_evaluator<>>(rbf, poly_dimension, poly_degree, points_);
  }

  rbf_residual_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                         int tree_height, const geometry::bbox3d& bbox)
    : rbf_(rbf)
    , n_polynomials_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , n_points_(0) {
    evaluator_ = std::make_unique<rbf_evaluator<>>(rbf, poly_dimension, poly_degree, tree_height, bbox);
  }

  template <class Derived, class Derived2>
  bool converged(const Eigen::MatrixBase<Derived>& values, const Eigen::MatrixBase<Derived2>& weights,
                 double absolute_tolerance) const {
    assert(values.size() == n_points_);
    assert(weights.size() == n_points_ + n_polynomials_);

    evaluator_->set_weights(weights);

    double max_residual = 0.0;
    for (size_t i = 0; i < n_points_ / chunk_size + 1; i++) {
      auto begin = i * chunk_size;
      auto end = std::min(n_points_, begin + chunk_size);
      if (begin == end) break;

      evaluator_->set_field_points(common::make_range_view(points_, begin, end));
      auto fit = evaluator_->evaluate();

      for (size_t j = 0; j < end - begin; j++) {
        auto res = std::abs(values(begin + j) - fit(j));
        if (res >= absolute_tolerance + std::abs(rbf_.nugget() * weights(begin + j)))
          return false;

        max_residual = std::max(max_residual, res);
      }
    }

    std::cout << "Maximum residual: " << max_residual << std::endl;

    return true;
  }

  template <class Container>
  void set_points(const Container& points) {
    points_ = std::vector<Eigen::Vector3d>(points.begin(), points.end());
    n_points_ = points.size();

    evaluator_->set_source_points(points);
  }

private:
  const rbf::rbf_base& rbf_;
  const size_t n_polynomials_;

  std::vector<Eigen::Vector3d> points_;
  size_t n_points_;

  std::unique_ptr<rbf_evaluator<>> evaluator_;
};

} // namespace interpolation
} // namespace polatory
