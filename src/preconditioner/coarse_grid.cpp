#include <polatory/common/macros.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>

namespace polatory::preconditioner {

coarse_grid::coarse_grid(const model& model,
                         const std::unique_ptr<polynomial::lagrange_basis>& lagrange_basis,
                         const std::vector<index_t>& point_indices)
    : model_(model),
      lagrange_basis_(lagrange_basis),
      point_idcs_(point_indices),
      l_(lagrange_basis ? lagrange_basis->basis_size() : 0),
      m_(static_cast<index_t>(point_indices.size())) {
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
  ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();

  a_top_ = Eigen::MatrixXd();
  lu_of_p_top_ = Eigen::FullPivLU<Eigen::MatrixXd>();
}

void coarse_grid::setup(const geometry::points3d& points_full) {
  auto points = points_full(point_idcs_, Eigen::all);

  // Compute A.
  Eigen::MatrixXd a(m_, m_);
  const auto& rbf = model_.rbf();
  a.diagonal().array() = rbf.evaluate(geometry::vector3d::Zero()) + model_.nugget();
  for (index_t i = 0; i < m_ - 1; i++) {
    for (index_t j = i + 1; j < m_; j++) {
      a(i, j) = rbf.evaluate(points.row(i) - points.row(j));
      a(j, i) = a(i, j);
    }
  }

  if (l_ > 0) {
    // Compute -E.
    auto tail_points = points.bottomRows(m_ - l_);
    me_ = -lagrange_basis_->evaluate(tail_points);

    // Compute decomposition of Q^T A Q.
    ldlt_of_qtaq_ = (me_.transpose() * a.topLeftCorner(l_, l_) * me_ +
                     me_.transpose() * a.topRightCorner(l_, m_ - l_) +
                     a.bottomLeftCorner(m_ - l_, l_) * me_ + a.bottomRightCorner(m_ - l_, m_ - l_))
                        .ldlt();

    // Compute matrices used for solving polynomial part.
    a_top_ = a.topRows(l_);

    polynomial::monomial_basis mono_basis(lagrange_basis_->dimension(), lagrange_basis_->degree());
    auto head_points = points.topRows(l_);
    Eigen::MatrixXd p_top = mono_basis.evaluate(head_points).transpose();
    lu_of_p_top_ = p_top.fullPivLu();
  } else {
    ldlt_of_qtaq_ = a.ldlt();
  }
}

void coarse_grid::set_solution_to(Eigen::Ref<common::valuesd> weights_full) const {
  weights_full(point_idcs_) = lambda_c_.head(m_);
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
