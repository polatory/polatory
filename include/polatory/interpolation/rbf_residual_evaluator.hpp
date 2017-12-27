// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>
#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace interpolation {

class rbf_residual_evaluator {
  static constexpr int chunk_size = 1024;

public:
  rbf_residual_evaluator(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
                         const geometry::points3d& points)
    : rbf_(rbf)
    , n_poly_basis_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , n_points_(points.rows())
    , points_(points) {
    evaluator_ = std::make_unique<rbf_evaluator<>>(rbf, poly_dimension, poly_degree, points_);
  }

  rbf_residual_evaluator(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
                         int tree_height, const geometry::bbox3d& bbox)
    : rbf_(rbf)
    , n_poly_basis_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , n_points_(0) {
    evaluator_ = std::make_unique<rbf_evaluator<>>(rbf, poly_dimension, poly_degree, tree_height, bbox);
  }

  template <class Derived, class Derived2>
  std::pair<bool, double> converged(const Eigen::MatrixBase<Derived>& values,
                                    const Eigen::MatrixBase<Derived2>& weights,
                                    double absolute_tolerance) const {
    assert(values.rows() == n_points_);
    assert(weights.rows() == n_points_ + n_poly_basis_);

    evaluator_->set_weights(weights);

    auto& rbf_kern = rbf_.get();
    auto nugget = rbf_kern.nugget();

    double max_residual = 0.0;
    for (size_t i = 0; i < n_points_ / chunk_size + 1; i++) {
      auto begin = i * chunk_size;
      auto end = std::min(n_points_, begin + chunk_size);
      if (begin == end) break;

      evaluator_->set_field_points(points_.middleRows(begin, end - begin));
      auto fit = evaluator_->evaluate();

      for (size_t j = 0; j < end - begin; j++) {
        auto res = std::abs(values(begin + j) - fit(j));
        if (res >= absolute_tolerance + std::abs(nugget * weights(begin + j)))
          return std::make_pair(false, 0.0);

        max_residual = std::max(max_residual, res);
      }
    }

    return std::make_pair(true, max_residual);
  }

  void set_points(const geometry::points3d& points) {
    n_points_ = points.rows();
    points_ = points;

    evaluator_->set_source_points(points);
  }

private:
  const rbf::rbf rbf_;
  const size_t n_poly_basis_;

  size_t n_points_;
  geometry::points3d points_;

  std::unique_ptr<rbf_evaluator<>> evaluator_;
};

}  // namespace interpolation
}  // namespace polatory
