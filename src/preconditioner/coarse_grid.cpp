#include <polatory/common/macros.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <utility>

#include "mat_a.hpp"

namespace polatory::preconditioner {

coarse_grid::coarse_grid(const model& model, domain&& domain)
    : model_(model),
      point_idcs_(std::move(domain.point_indices)),
      grad_point_idcs_(std::move(domain.grad_point_indices)),
      l_(model.poly_basis_size()),
      mu_(static_cast<index_t>(point_idcs_.size())),
      sigma_(static_cast<index_t>(grad_point_idcs_.size())),
      m_(mu_ + model.poly_dimension() * sigma_) {
  POLATORY_ASSERT(mu_ > l_);
}

void coarse_grid::clear() {
  me_ = Eigen::MatrixXd();
  ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();

  a_top_ = Eigen::MatrixXd();
  lu_of_p_top_ = Eigen::FullPivLU<Eigen::MatrixXd>();
}

void coarse_grid::setup(const geometry::points3d& points_full,
                        const Eigen::MatrixXd& lagrange_pt_full) {
  setup(points_full, geometry::points3d(0, 3), lagrange_pt_full);
}

void coarse_grid::setup(const geometry::points3d& points_full,
                        const geometry::points3d& grad_points_full,
                        const Eigen::MatrixXd& lagrange_pt_full) {
  auto points = points_full(point_idcs_, Eigen::all);
  auto grad_points = grad_points_full(grad_point_idcs_, Eigen::all);

  // Compute A.
  auto a = mat_a(model_, points, grad_points);

  if (l_ > 0) {
    // Compute -E.
    auto lagrange_pt = lagrange_pt_full(Eigen::all, point_idcs_);
    me_ = -lagrange_pt.rightCols(m_ - l_);

    // Compute decomposition of Q^T A Q.
    ldlt_of_qtaq_ = (me_.transpose() * a.topLeftCorner(l_, l_) * me_ +
                     me_.transpose() * a.topRightCorner(l_, m_ - l_) +
                     a.bottomLeftCorner(m_ - l_, l_) * me_ + a.bottomRightCorner(m_ - l_, m_ - l_))
                        .ldlt();

    // Compute matrices used for solving polynomial part.
    a_top_ = a.topRows(l_);

    polynomial::monomial_basis mono_basis(model_.poly_dimension(), model_.poly_degree());
    auto head_points = points.topRows(l_);
    Eigen::MatrixXd p_top = mono_basis.evaluate(head_points).transpose();
    lu_of_p_top_ = p_top.fullPivLu();
  } else {
    ldlt_of_qtaq_ = a.ldlt();
  }

  mu_full_ = points_full.rows();
}

void coarse_grid::set_solution_to(Eigen::Ref<common::valuesd> weights_full) const {
  auto dim = model_.poly_dimension();

  weights_full(point_idcs_) = lambda_c_.head(mu_);

  for (index_t i = 0; i < sigma_; i++) {
    weights_full.segment(mu_full_ + dim * grad_point_idcs_.at(i), dim) =
        lambda_c_.segment(mu_ + dim * i, dim);
  }

  weights_full.tail(l_) = lambda_c_.tail(l_);
}

void coarse_grid::solve(const Eigen::Ref<const common::valuesd>& values_full) {
  auto values = values_full(point_idcs_);

  if (l_ > 0) {
    // Compute Q^T d.
    common::valuesd qtd = me_.transpose() * values.head(l_) + values.tail(m_ - l_);

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

}  // namespace polatory::preconditioner
