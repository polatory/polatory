#pragma once

#include <Eigen/Core>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory::polynomial {

template <class Basis>
class polynomial_evaluator {
 public:
  explicit polynomial_evaluator(int degree)
      : basis_(degree), weights_(common::valuesd::Zero(basis_.basis_size())) {}

  common::valuesd evaluate() const {
    Eigen::MatrixXd pt = basis_.evaluate(points_, grad_points_);

    return pt.transpose() * weights_;
  }

  void set_field_points(const geometry::points3d& points, const geometry::points3d& grad_points) {
    points_ = points;
    grad_points_ = grad_points;
  }

  void set_weights(const common::valuesd& weights) {
    POLATORY_ASSERT(weights.rows() == basis_.basis_size());

    weights_ = weights;
  }

 private:
  const Basis basis_;

  geometry::points3d points_;
  geometry::points3d grad_points_;
  common::valuesd weights_;
};

}  // namespace polatory::polynomial
