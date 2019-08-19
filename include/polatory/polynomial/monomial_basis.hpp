// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace polynomial {

class monomial_basis : public polynomial_basis_base {
public:
  explicit monomial_basis(int dimension, int degree)
    : polynomial_basis_base(dimension, degree) {
    POLATORY_ASSERT(degree >= 0 && degree <= 2);
  }

  Eigen::MatrixXd evaluate_points(const geometry::points3d& points) const {
    auto n_points = static_cast<index_t>(points.rows());

    Eigen::MatrixXd result(basis_size(), n_points);

    switch (dimension()) {
    case 1:
      switch (degree()) {
      case 0:
        // 1
        for (index_t i = 0; i < n_points; i++) {
          result(0, i) = 1.0;
        }
        break;
      case 1:
        // 1, x
        for (index_t i = 0; i < n_points; i++) {
          auto p = points.row(i);
          result(0, i) = 1.0;
          result(1, i) = p(0);
        }
        break;
      case 2:
        // 1, x, x^2
        for (index_t i = 0; i < n_points; i++) {
          auto p = points.row(i);
          result(0, i) = 1.0;
          result(1, i) = p(0);
          result(2, i) = p(0) * p(0);
        }
        break;
      default:
        POLATORY_NEVER_REACH();
        break;
      }
      break;

    case 2:
      switch (degree()) {
      case 0:
        // 1
        for (index_t i = 0; i < n_points; i++) {
          result(0, i) = 1.0;
        }
        break;
      case 1:
        // 1, x, y
        for (index_t i = 0; i < n_points; i++) {
          auto p = points.row(i);
          result(0, i) = 1.0;
          result(1, i) = p(0);
          result(2, i) = p(1);
        }
        break;
      case 2:
        // 1, x, y, x^2, xy, y^2
        for (index_t i = 0; i < n_points; i++) {
          auto p = points.row(i);
          result(0, i) = 1.0;
          result(1, i) = p(0);
          result(2, i) = p(1);
          result(3, i) = p(0) * p(0);
          result(4, i) = p(0) * p(1);
          result(5, i) = p(1) * p(1);
        }
        break;
      default:
        POLATORY_NEVER_REACH();
        break;
      }
      break;

    case 3:
      switch (degree()) {
      case 0:
        // 1
        for (index_t i = 0; i < n_points; i++) {
          result(0, i) = 1.0;
        }
        break;
      case 1:
        // 1, x, y, z
        for (index_t i = 0; i < n_points; i++) {
          auto p = points.row(i);
          result(0, i) = 1.0;
          result(1, i) = p(0);
          result(2, i) = p(1);
          result(3, i) = p(2);
        }
        break;
      case 2:
        // 1, x, y, z, x^2, xy, xz, y^2, yz, z^2
        for (index_t i = 0; i < n_points; i++) {
          auto p = points.row(i);
          result(0, i) = 1.0;
          result(1, i) = p(0);
          result(2, i) = p(1);
          result(3, i) = p(2);
          result(4, i) = p(0) * p(0);
          result(5, i) = p(0) * p(1);
          result(6, i) = p(0) * p(2);
          result(7, i) = p(1) * p(1);
          result(8, i) = p(1) * p(2);
          result(9, i) = p(2) * p(2);
        }
        break;
      default:
        POLATORY_NEVER_REACH();
        break;
      }
      break;

    default:
      POLATORY_NEVER_REACH();
      break;
    }

    return result;
  }
};

}  // namespace polynomial
}  // namespace polatory
