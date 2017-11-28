// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace preconditioner {

template <class Floating>
class fine_grid {
  using Vector3F = Eigen::Matrix<Floating, 3, 1>;
  using VectorXF = Eigen::Matrix<Floating, Eigen::Dynamic, 1>;
  using MatrixXF = Eigen::Matrix<Floating, Eigen::Dynamic, Eigen::Dynamic>;
  using LagrangeBasis = polynomial::lagrange_basis<Floating>;

public:
  fine_grid(const rbf::rbf& rbf,
            std::shared_ptr<LagrangeBasis> lagrange_basis,
            const std::vector<size_t>& point_indices,
            const std::vector<bool>& inner_point)
    : rbf_(rbf)
    , lagrange_basis_(lagrange_basis)
    , point_idcs_(point_indices)
    , inner_point_(inner_point)
    , l_(lagrange_basis ? lagrange_basis->basis_size() : 0)
    , m_(point_indices.size()) {
    assert(m_ > l_);
  }

  fine_grid(const rbf::rbf& rbf,
            std::shared_ptr<LagrangeBasis> lagrange_basis,
            const std::vector<size_t>& point_indices,
            const std::vector<bool>& inner_point,
            const geometry::points3d& points_full)
    : fine_grid(rbf, lagrange_basis, point_indices, inner_point) {
    setup(points_full);
  }

  void clear() {
    me_ = MatrixXF();
    ldlt_of_qtaq_ = Eigen::LDLT<MatrixXF>();
  }

  void setup(const geometry::points3d& points_full) {
    // Compute A.
    MatrixXF a(m_, m_);
    auto& rbf_kern = rbf_.get();
    auto diagonal = rbf_kern.evaluate(0.0) + rbf_kern.nugget();
    for (size_t i = 0; i < m_; i++) {
      a(i, i) = diagonal;
    }
    for (size_t i = 0; i < m_ - 1; i++) {
      for (size_t j = i + 1; j < m_; j++) {
        a(i, j) = rbf_kern.evaluate(points_full.row(point_idcs_[i]), points_full.row(point_idcs_[j]));
        a(j, i) = a(i, j);
      }
    }

    if (l_ > 0) {
      // Compute -E.
      geometry::points3d tail_points(m_ - l_);
      for (size_t i = 0; i < m_ - l_; i++) {
        tail_points.row(i) = points_full.row(point_idcs_[l_ + i]);
      }

      me_ = -lagrange_basis_->evaluate_points(tail_points);

      // Compute decomposition of Q^T A Q.
      ldlt_of_qtaq_ = (me_.transpose() * a.topLeftCorner(l_, l_) * me_
                       + me_.transpose() * a.topRightCorner(l_, m_ - l_)
                       + a.bottomLeftCorner(m_ - l_, l_) * me_
                       + a.bottomRightCorner(m_ - l_, m_ - l_)).ldlt();
    } else {
      ldlt_of_qtaq_ = a.ldlt();
    }
  }

  template <class Derived>
  void set_solution_to(Eigen::MatrixBase<Derived>& weights_full) const {
    for (size_t i = 0; i < m_; i++) {
      if (inner_point_[i])
        weights_full(point_idcs_[i]) = lambda_(i);
    }
  }

  template <class Derived>
  void solve(const Eigen::MatrixBase<Derived>& values_full) {
    VectorXF values = VectorXF(m_);
    for (size_t i = 0; i < m_; i++) {
      values(i) = values_full(point_idcs_[i]);
    }

    if (l_ > 0) {
      // Compute Q^T d.
      VectorXF qtd = me_.transpose() * values.head(l_)
                     + values.tail(m_ - l_);

      // Solve Q^T A Q gamma = Q^T d for gamma.
      VectorXF gamma = ldlt_of_qtaq_.solve(qtd);

      // Compute lambda = Q gamma.
      lambda_ = VectorXF(m_);
      lambda_.head(l_) = me_ * gamma;
      lambda_.tail(m_ - l_) = gamma;
    } else {
      lambda_ = ldlt_of_qtaq_.solve(values);
    }
  }

private:
  const rbf::rbf rbf_;
  const std::shared_ptr<LagrangeBasis> lagrange_basis_;
  const std::vector<size_t> point_idcs_;
  const std::vector<bool> inner_point_;

  const size_t l_;
  const size_t m_;

  // Matrix -E.
  MatrixXF me_;

  // Cholesky decomposition of matrix Q^T A Q.
  Eigen::LDLT<MatrixXF> ldlt_of_qtaq_;

  // Current solution.
  VectorXF lambda_;
};

} // namespace preconditioner
} // namespace polatory
