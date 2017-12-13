// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/basis_base.hpp>

namespace polatory {
namespace polynomial {

class monomial_basis : public basis_base {
public:
  explicit monomial_basis(int dimension, int degree)
    : basis_base(dimension, degree) {
    assert(degree >= 0 && degree <= 2);
  }

  Eigen::MatrixXd evaluate_points(const geometry::points3d& points) const {
    size_t n_points = points.rows();

    Eigen::MatrixXd result(basis_size(), n_points);

    switch (dimension()) {
    case 1:
      switch (degree()) {
      case 0:
        // 1
        for (size_t i = 0; i < n_points; i++) {
          result(0, i) = 1.0;
        }
        break;
      case 1:
        // 1, x
        for (size_t i = 0; i < n_points; i++) {
          auto p = points.row(i);
          result(0, i) = 1.0;
          result(1, i) = p(0);
        }
        break;
      case 2:
        // 1, x, x^2
        for (size_t i = 0; i < n_points; i++) {
          auto p = points.row(i);
          result(0, i) = 1.0;
          result(1, i) = p(0);
          result(2, i) = p(0) * p(0);
        }
        break;
      default:
        assert(false);
        break;
      }
      break;

    case 2:
      switch (degree()) {
      case 0:
        // 1
        for (size_t i = 0; i < n_points; i++) {
          result(0, i) = 1.0;
        }
        break;
      case 1:
        // 1, x, y
        for (size_t i = 0; i < n_points; i++) {
          auto p = points.row(i);
          result(0, i) = 1.0;
          result(1, i) = p(0);
          result(2, i) = p(1);
        }
        break;
      case 2:
        // 1, x, y, x^2, xy, y^2
        for (size_t i = 0; i < n_points; i++) {
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
        assert(false);
        break;
      }
      break;

    case 3:
      switch (degree()) {
      case 0:
        // 1
        for (size_t i = 0; i < n_points; i++) {
          result(0, i) = 1.0;
        }
        break;
      case 1:
        // 1, x, y, z
        for (size_t i = 0; i < n_points; i++) {
          auto p = points.row(i);
          result(0, i) = 1.0;
          result(1, i) = p(0);
          result(2, i) = p(1);
          result(3, i) = p(2);
        }
        break;
      case 2:
        // 1, x, y, z, x^2, xy, xz, y^2, yz, z^2
        for (size_t i = 0; i < n_points; i++) {
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
        assert(false);
        break;
      }
      break;

    default:
      assert(false);
      break;
    }

    return result;
  }
};

} // namespace polynomial
} // namespace polatory
