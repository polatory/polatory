// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include "polatory/polynomial/basis_base.hpp"
#include "polatory/polynomial/lagrange_basis.hpp"
#include "polatory/polynomial/monomial_basis.hpp"
#include "polatory/rbf/rbf_base.hpp"

namespace polatory {
namespace interpolation {

// Solves rbf interpolation problems for
// small- to mid-sized (up to 1k) point sets.
// Computational complexity: O(N^2) in space and O(N^3) in time,
// where N ~ the number of points.
template <typename Floating>
class rbf_direct_solver {
  using Vector3F = Eigen::Matrix<Floating, 3, 1>;
  using VectorXF = Eigen::Matrix<Floating, Eigen::Dynamic, 1>;
  using MatrixXF = Eigen::Matrix<Floating, Eigen::Dynamic, Eigen::Dynamic>;

  const rbf::rbf_base& rbf;

  const int poly_dimension;
  const int poly_degree;

  std::vector<size_t> point_idcs;
  std::vector<Vector3F> poly_points;

  // Matrix -E.
  MatrixXF me;

  // First l rows of matrix A.
  MatrixXF a_top;

  // Decomposition of martix Q^T A Q, where Q^T = ( -E^T  I ).
  // This version is used when the system is positive definite.
  Eigen::LDLT<MatrixXF> ldlt_of_qtaq;

  // Decomposition of martix A.
  // This version is used when the system is indefinite.
  Eigen::PartialPivLU<MatrixXF> lu_of_a;

  const size_t l;
  const size_t m;

public:
  rbf_direct_solver(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                    size_t n_points)
    : rbf(rbf)
    , poly_dimension(poly_dimension)
    , poly_degree(poly_degree)
    , l(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , m(n_points) {
    assert(m > l);
  }

  rbf_direct_solver(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                    const std::vector<Eigen::Vector3d>& points)
    : rbf(rbf)
    , poly_dimension(poly_dimension)
    , poly_degree(poly_degree)
    , l(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , m(points.size()) {
    assert(m > l);

    setup(points);
  }

  void clear() {
    a_top = MatrixXF();

    ldlt_of_qtaq = Eigen::LDLT<MatrixXF>();

    lu_of_a = Eigen::PartialPivLU<MatrixXF>();

    me = MatrixXF();
  }

  void setup(const std::vector<Eigen::Vector3d>& points) {
    point_idcs.resize(m);
    std::iota(point_idcs.begin(), point_idcs.end(), 0);

    if (poly_degree >= 0) {
      std::random_device rd;
      std::mt19937 gen(rd());

      std::shuffle(point_idcs.begin(), point_idcs.end(), gen);
    }

    // Compute A.
    MatrixXF a(m, m);
    auto diagonal = rbf.evaluate(0.0) + rbf.nugget();
    for (size_t i = 0; i < m; i++) {
      a(i, i) = diagonal;
    }
    for (size_t i = 0; i < m - 1; i++) {
      for (size_t j = i + 1; j < m; j++) {
        a(i, j) = rbf.evaluate(points[point_idcs[i]], points[point_idcs[j]]);
        a(j, i) = a(i, j);
      }
    }

    if (poly_degree >= 0) {
      a_top = a.topRows(l);

      // Compute -E.
      for (size_t i = 0; i < l; i++) {
        poly_points.push_back(points[point_idcs[i]].template cast<Floating>());
      }

      std::vector<Vector3F> other_points;
      for (size_t i = l; i < m; i++) {
        other_points.push_back(points[point_idcs[i]].template cast<Floating>());
      }

      polynomial::lagrange_basis<Floating> lagr_basis(poly_dimension, poly_degree, poly_points);
      me = -lagr_basis.evaluate_points(other_points);

      // Compute decomposition of Q^T A Q.
      ldlt_of_qtaq = (me.transpose() * a.topLeftCorner(l, l) * me
                      + me.transpose() * a.topRightCorner(l, m - l)
                      + a.bottomLeftCorner(m - l, l) * me
                      + a.bottomRightCorner(m - l, m - l)).ldlt();
    } else {
      lu_of_a = a.partialPivLu();
    }
  }

  template <typename Derived>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Derived>& values) const {
    assert(values.size() == m);

    VectorXF values_permuted = VectorXF(values.size());
    for (size_t i = 0; i < m; i++) {
      values_permuted(i) = values(point_idcs[i]);
    }

    Eigen::VectorXd lambda_c;
    if (poly_degree >= 0) {
      // Compute Q^T d.
      VectorXF qtd = me.transpose() * values_permuted.head(l)
                     + values_permuted.tail(m - l);

      // Solve (Q^T A Q) gamma = Q^T d for gamma.
      VectorXF gamma = ldlt_of_qtaq.solve(qtd);

      // Compute lambda = Q gamma.
      lambda_c = Eigen::VectorXd(m + l);
      lambda_c.head(l) = (me * gamma).template cast<double>();
      lambda_c.segment(l, m - l) = gamma.template cast<double>();

      // Solve P c = d - A lambda for c at poly_points.

      VectorXF d_at_poly_points = VectorXF(l);
      for (size_t i = 0; i < l; i++) {
        d_at_poly_points(i) = values_permuted(i);
      }

      VectorXF a_lambda_at_poly_points = VectorXF::Zero(l);
      for (size_t i = 0; i < l; i++) {
        for (size_t j = 0; j < m; j++) {
          a_lambda_at_poly_points(i) += lambda_c(j) * a_top(i, j);
        }
      }

      polynomial::monomial_basis<Floating> mono_basis(poly_dimension, poly_degree);
      auto pt = mono_basis.evaluate_points(poly_points);
      lambda_c.tail(l) = pt.transpose().fullPivLu().solve(d_at_poly_points - a_lambda_at_poly_points).template cast<double>();
    } else {
      lambda_c = lu_of_a.solve(values_permuted).template cast<double>();
    }

    Eigen::VectorXd lambda_permuted = lambda_c.head(m);
    for (size_t i = 0; i < m; i++) {
      lambda_c(point_idcs[i]) = lambda_permuted(i);
    }

    return lambda_c;
  }
};

} // namespace interpolation
} // namespace polatory
