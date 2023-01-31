#pragma once

#include <Eigen/Core>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory::polynomial {

template <class Basis>
class polynomial_evaluator {
 public:
  explicit polynomial_evaluator(int dimension, int degree)
      : basis_(dimension, degree), weights_(common::valuesd::Zero(basis_.basis_size())) {}

  common::valuesd evaluate() const {
    Eigen::MatrixXd pt = basis_.evaluate(points_);

    return pt.transpose() * weights_;
  }

  void set_field_points(const geometry::points3d& points) { points_ = points; }

  void set_weights(const common::valuesd& weights) {
    POLATORY_ASSERT(weights.rows() == basis_.basis_size());

    weights_ = weights;
  }

 private:
  const Basis basis_;

  geometry::points3d points_;
  common::valuesd weights_;
};

}  // namespace polatory::polynomial
