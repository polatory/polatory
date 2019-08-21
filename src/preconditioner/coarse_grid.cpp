// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/preconditioner/coarse_grid.hpp>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/iterator_range.hpp>
#include <polatory/common/macros.hpp>
#include <polatory/polynomial/monomial_basis.hpp>

namespace polatory {
namespace preconditioner {

coarse_grid::coarse_grid(const model& model,
                         const std::unique_ptr<polynomial::lagrange_basis>& lagrange_basis,
                         const std::vector<index_t>& point_indices)
  : model_(model)
  , lagrange_basis_(lagrange_basis)
  , point_idcs_(point_indices)
  , l_(lagrange_basis ? lagrange_basis->basis_size() : 0)
  , m_(static_cast<index_t>(point_indices.size())) {
  POLATORY_ASSERT(m_ > l_);
}

coarse_grid::coarse_grid(const model& model,
                         const std::unique_ptr<polynomial::lagrange_basis>& lagrange_basis,
                         const std::vector<index_t>& point_indices,
                         const geometry::points3d& points_full)
  : coarse_grid(model, lagrange_basis, point_indices) {
  setup(points_full);
}

void coarse_grid::clear() {
  me_ = Eigen::MatrixXd();
  ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();  // NOLINT(clang-analyzer-core.uninitialized.Assign)

  a_top_ = Eigen::MatrixXd();
  lu_of_p_top_ = Eigen::FullPivLU<Eigen::MatrixXd>();  // NOLINT(clang-analyzer-core.uninitialized.Assign)
}

void coarse_grid::setup(const geometry::points3d& points_full) {
  // Compute A.
  Eigen::MatrixXd a(m_, m_);
  auto& rbf = model_.rbf();
  auto diagonal = rbf.evaluate_untransformed(0.0) + model_.nugget();
  for (index_t i = 0; i < m_; i++) {
    a(i, i) = diagonal;
  }
  for (index_t i = 0; i < m_ - 1; i++) {
    for (index_t j = i + 1; j < m_; j++) {
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

void coarse_grid::set_solution_to(Eigen::Ref<common::valuesd> weights_full) const {
  for (index_t i = 0; i < m_; i++) {
    weights_full(point_idcs_[i]) = lambda_c_(i);
  }

  weights_full.tail(l_) = lambda_c_.tail(l_).cast<double>();
}

void coarse_grid::solve(const Eigen::Ref<const common::valuesd>& values_full) {
  common::valuesd values(m_);
  for (index_t i = 0; i < m_; i++) {
    values(i) = values_full(point_idcs_[i]);
  }

  if (l_ > 0) {
    // Compute Q^T d.
    common::valuesd qtd = me_.transpose() * values.head(l_)
                          + values.tail(m_ - l_);

    // Solve Q^T A Q gamma = Q^T d for gamma.
    common::valuesd gamma = ldlt_of_qtaq_.solve(qtd);

    // Compute lambda = Q gamma.
    lambda_c_ = common::valuesd(m_ + l_);
    lambda_c_.head(l_) = me_ * gamma;
    lambda_c_.segment(l_, m_ - l_) = gamma;

    // Solve P c = d - A lambda for c at poly_points.
    common::valuesd a_top_lambda = a_top_ * lambda_c_.head(m_);
    lambda_c_.tail(l_) = lu_of_p_top_.solve(values.head(l_) - a_top_lambda);
  } else {
    lambda_c_ = ldlt_of_qtaq_.solve(values);
  }
}

}  // namespace preconditioner
}  // namespace polatory
