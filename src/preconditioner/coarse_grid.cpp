// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/preconditioner/coarse_grid.hpp>

#include <cassert>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/iterator_range.hpp>
#include <polatory/polynomial/monomial_basis.hpp>

namespace polatory {
namespace preconditioner {

coarse_grid::coarse_grid(const model& model,
                         std::shared_ptr<polynomial::lagrange_basis> lagrange_basis,
                         const std::vector<size_t>& point_indices)
  : model_(model)
  , lagrange_basis_(lagrange_basis)
  , point_idcs_(point_indices)
  , l_(lagrange_basis ? lagrange_basis->basis_size() : 0)
  , m_(point_indices.size()) {
  assert(m_ > l_);
}

coarse_grid::coarse_grid(const model& model,
                         std::shared_ptr<polynomial::lagrange_basis> lagrange_basis,
                         const std::vector<size_t>& point_indices,
                         const geometry::points3d& points_full)
  : coarse_grid(model, lagrange_basis, point_indices) {
  setup(points_full);
}

void coarse_grid::clear() {
  me_ = Eigen::MatrixXd();
  ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();

  a_top_ = Eigen::MatrixXd();
  lu_of_p_top_ = Eigen::FullPivLU<Eigen::MatrixXd>();
}

void coarse_grid::setup(const geometry::points3d& points_full) {
  // Compute A.
  Eigen::MatrixXd a(m_, m_);
  auto& rbf = model_.rbf();
  auto diagonal = rbf.evaluate_transformed(0.0) + rbf.nugget();
  for (size_t i = 0; i < m_; i++) {
    a(i, i) = diagonal;
  }
  for (size_t i = 0; i < m_ - 1; i++) {
    for (size_t j = i + 1; j < m_; j++) {
      a(i, j) = rbf.evaluate(points_full.row(point_idcs_[i]) - points_full.row(point_idcs_[j]));
      a(j, i) = a(i, j);
    }
  }

  if (l_ > 0) {
    // Compute -E.
    auto tail_points = common::take_rows(points_full, common::make_range(point_idcs_.begin() + l_, point_idcs_.end()));
    me_ = -lagrange_basis_->evaluate_points(tail_points);

    // Compute decomposition of Q^T A Q.
    ldlt_of_qtaq_ = (me_.transpose() * a.topLeftCorner(l_, l_) * me_
                     + me_.transpose() * a.topRightCorner(l_, m_ - l_)
                     + a.bottomLeftCorner(m_ - l_, l_) * me_
                     + a.bottomRightCorner(m_ - l_, m_ - l_)).ldlt();

    // Compute matrices used for solving polynomial part.
    a_top_ = a.topRows(l_);

    polynomial::monomial_basis mono_basis(lagrange_basis_->dimension(), lagrange_basis_->degree());
    auto head_points = common::take_rows(points_full, common::make_range(point_idcs_.begin(), point_idcs_.begin() + l_));
    Eigen::MatrixXd p_top = mono_basis.evaluate_points(head_points).transpose();
    lu_of_p_top_ = p_top.fullPivLu();
  } else {
    ldlt_of_qtaq_ = a.ldlt();
  }
}

}  // namespace preconditioner
}  // namespace polatory
