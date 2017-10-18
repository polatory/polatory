// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "polatory/polynomial/lagrange_basis.hpp"
#include "polatory/polynomial/monomial_basis.hpp"
#include "polatory/rbf/rbf_base.hpp"

namespace polatory {
namespace preconditioner {

template <typename Floating>
class coarse_grid {
  using Vector3F = Eigen::Matrix<Floating, 3, 1>;
  using VectorXF = Eigen::Matrix<Floating, Eigen::Dynamic, 1>;
  using MatrixXF = Eigen::Matrix<Floating, Eigen::Dynamic, Eigen::Dynamic>;
  using LagrangeBasis = polynomial::lagrange_basis<Floating>;
  using MonomialBasis = polynomial::monomial_basis<Floating>;

  const rbf::rbf_base& rbf_;
  const std::shared_ptr<LagrangeBasis> lagrange_basis_;
  const std::vector<size_t> point_idcs_;

  const size_t l_;
  const size_t m_;

  // First l rows of matrix A.
  MatrixXF a_top_;

  // Matrix -E.
  MatrixXF me_;

  // Decomposition of martix Q^T A Q, where Q^T = ( -E^T  I ).
  // This version is used when the system is positive definite.
  Eigen::LDLT<MatrixXF> ldlt_of_qtaq_;

  MatrixXF pt_;

  // Current solution.
  VectorXF lambda_c_;

public:
  coarse_grid(const rbf::rbf_base& rbf,
              std::shared_ptr<LagrangeBasis> lagrange_basis,
              const std::vector<size_t>& point_indices)
    : rbf_(rbf)
    , lagrange_basis_(lagrange_basis)
    , point_idcs_(point_indices)
    , l_(lagrange_basis ? lagrange_basis->basis_size() : 0)
    , m_(point_indices.size()) {
    assert(m_ > l_);
  }

  coarse_grid(const rbf::rbf_base& rbf,
              std::shared_ptr<LagrangeBasis> lagrange_basis,
              const std::vector<size_t>& point_indices,
              const std::vector<Eigen::Vector3d>& points_full)
    : coarse_grid(rbf, lagrange_basis, point_indices) {
    setup(points_full);
  }

  void clear() {
    a_top_ = MatrixXF();
    me_ = MatrixXF();
    ldlt_of_qtaq_ = Eigen::LDLT<MatrixXF>();
    pt_ = MatrixXF();
  }

  void setup(const std::vector<Eigen::Vector3d>& points_full) {
    // Compute A.
    MatrixXF a(m_, m_);
    auto diagonal = rbf_.evaluate(0.0) + rbf_.nugget();
    for (size_t i = 0; i < m_; i++) {
      a(i, i) = diagonal;
    }
    for (size_t i = 0; i < m_ - 1; i++) {
      for (size_t j = i + 1; j < m_; j++) {
        a(i, j) = rbf_.evaluate(points_full[point_idcs_[i]], points_full[point_idcs_[j]]);
        a(j, i) = a(i, j);
      }
    }

    if (l_ > 0) {
      a_top_ = a.topRows(l_);

      // Compute -E.
      std::vector<Vector3F> tail_points;
      tail_points.reserve(m_ - l_);
      for (size_t i = l_; i < m_; i++) {
        tail_points.push_back(points_full[point_idcs_[i]].template cast<Floating>());
      }

      me_ = -lagrange_basis_->evaluate_points(tail_points);

      // Compute decomposition of Q^T A Q.
      ldlt_of_qtaq_ = (me_.transpose() * a.topLeftCorner(l_, l_) * me_
                       + me_.transpose() * a.topRightCorner(l_, m_ - l_)
                       + a.bottomLeftCorner(m_ - l_, l_) * me_
                       + a.bottomRightCorner(m_ - l_, m_ - l_)).ldlt();

      // Compute P^T.
      std::vector<Vector3F> head_points;
      head_points.reserve(l_);
      for (size_t i = 0; i < l_; i++) {
        head_points.push_back(points_full[point_idcs_[i]].template cast<Floating>());
      }

      MonomialBasis mono_basis(lagrange_basis_->dimension(), lagrange_basis_->degree());
      pt_ = mono_basis.evaluate_points(head_points);
    } else {
      ldlt_of_qtaq_ = a.ldlt();
    }
  }

  template <typename Derived>
  void set_solution_to(Eigen::MatrixBase<Derived>& weights_full) const {
    for (size_t i = 0; i < m_; i++) {
      weights_full(point_idcs_[i]) = lambda_c_(i);
    }

    weights_full.tail(l_) = lambda_c_.tail(l_).template cast<double>();
  }

  template <typename Derived>
  void solve(const Eigen::MatrixBase<Derived>& values_full) {
    VectorXF values = VectorXF(m_);
    for (size_t i = 0; i < m_; i++) {
      values(i) = values_full(point_idcs_[i]);
    }

    if (l_ > 0) {
      // Compute Q^T d.
      VectorXF qtd = me_.transpose() * values.head(l_)
                     + values.tail(m_ - l_);

      // Solve (Q^T A Q) gamma = Q^T d for gamma.
      VectorXF gamma = ldlt_of_qtaq_.solve(qtd);

      // Compute lambda = Q gamma.
      lambda_c_ = VectorXF(m_ + l_);
      lambda_c_.head(l_) = me_ * gamma;
      lambda_c_.segment(l_, m_ - l_) = gamma;

      // Solve P c = d - A lambda for c at poly_points.
      VectorXF a_lambda_top = a_top_ * lambda_c_.head(m_);
      lambda_c_.tail(l_) = pt_.transpose().fullPivLu().solve(values.head(l_) - a_lambda_top);
    } else {
      lambda_c_ = ldlt_of_qtaq_.solve(values);
    }
  }
};

} // namespace preconditioner
} // namespace polatory
