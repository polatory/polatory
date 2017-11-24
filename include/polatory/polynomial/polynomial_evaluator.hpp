// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <cassert>
#include <iterator>
#include <vector>

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace polynomial {

template <class Basis>
class polynomial_evaluator {
  const Basis basis;

  geometry::points3d points;
  Eigen::VectorXd weights;

public:
  explicit polynomial_evaluator(int dimension, int degree)
    : basis(dimension, degree)
    , weights(Eigen::VectorXd::Zero(basis.basis_size())) {
  }

  Eigen::VectorXd evaluate() const {
    Eigen::MatrixXd pt = basis.evaluate_points(points);

    return pt.transpose() * weights;
  }

  void set_field_points(const geometry::points3d& points) {
    this->points = points;
  }

  void set_weights(const Eigen::VectorXd& weights) {
    assert(weights.size() == basis.basis_size());

    this->weights = weights;
  }

  size_t size() const {
    return basis.basis_size();
  }
};

} // namespace polynomial
} // namespace polatory
