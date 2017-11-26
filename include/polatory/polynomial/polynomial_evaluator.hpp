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

public:
  explicit polynomial_evaluator(int dimension, int degree)
    : basis_(dimension, degree)
    , weights_(Eigen::VectorXd::Zero(basis_.basis_size())) {
  }

  Eigen::VectorXd evaluate() const {
    Eigen::MatrixXd pt = basis_.evaluate_points(points_);

    return pt.transpose() * weights_;
  }

  void set_field_points(const geometry::points3d& points) {
    points_ = points;
  }

  void set_weights(const Eigen::VectorXd& weights) {
    assert(weights.size() == basis_.basis_size());

    weights_ = weights;
  }

  size_t size() const {
    return basis_.basis_size();
  }

private:
  const Basis basis_;

  geometry::points3d points_;
  Eigen::VectorXd weights_;
};

} // namespace polynomial
} // namespace polatory
