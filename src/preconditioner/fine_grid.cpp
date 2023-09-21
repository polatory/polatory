#include <polatory/common/macros.hpp>
#include <polatory/preconditioner/fine_grid.hpp>

#include "mat_a.hpp"

namespace polatory::preconditioner {

fine_grid::fine_grid(const model& model, const std::vector<index_t>& point_indices,
                     const std::vector<bool>& inner_point)
    : fine_grid(model, point_indices, {}, inner_point, {}) {}

fine_grid::fine_grid(const model& model, const std::vector<index_t>& point_indices,
                     const std::vector<index_t>& grad_point_indices,
                     const std::vector<bool>& inner_point,
                     const std::vector<bool>& inner_grad_point)
    : model_(model),
      point_idcs_(point_indices),
      grad_point_idcs_(grad_point_indices),
      inner_point_(inner_point),
      inner_grad_point_(inner_grad_point),
      l_(model.poly_basis_size()),
      mu_(static_cast<index_t>(point_indices.size())),
      sigma_(static_cast<index_t>(grad_point_indices.size())),
      m_(mu_ + model.poly_dimension() * sigma_) {
  POLATORY_ASSERT(mu_ > l_);
}

void fine_grid::clear() {
  me_ = Eigen::MatrixXd();
  ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();
}

void fine_grid::setup(const geometry::points3d& points_full,
                      const Eigen::MatrixXd& lagrange_pt_full) {
  setup(points_full, geometry::points3d(0, 3), lagrange_pt_full);
}

void fine_grid::setup(const geometry::points3d& points_full,
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
  } else {
    ldlt_of_qtaq_ = a.ldlt();
  }

  mu_full_ = points_full.rows();
}

void fine_grid::set_solution_to(Eigen::Ref<common::valuesd> weights_full) const {
  auto dim = model_.poly_dimension();

  for (index_t i = 0; i < mu_; i++) {
    if (inner_point_.at(i)) {
      weights_full(point_idcs_.at(i)) = lambda_(i);
    }
  }

  for (index_t i = 0; i < sigma_; i++) {
    if (inner_grad_point_.at(i)) {
      weights_full.segment(mu_full_ + dim * grad_point_idcs_.at(i), dim) =
          lambda_.segment(mu_ + dim * i, dim);
    }
  }
}

void fine_grid::solve(const Eigen::Ref<const common::valuesd>& values_full) {
  auto values = values_full(point_idcs_);

  if (l_ > 0) {
    // Compute Q^T d.
    common::valuesd qtd = me_.transpose() * values.head(l_) + values.tail(m_ - l_);

    // Solve Q^T A Q gamma = Q^T d for gamma.
    common::valuesd gamma = ldlt_of_qtaq_.solve(qtd);

    // Compute lambda = Q gamma.
    lambda_ = common::valuesd(m_);
    lambda_.head(l_) = me_ * gamma;
    lambda_.tail(m_ - l_) = gamma;
  } else {
    lambda_ = ldlt_of_qtaq_.solve(values);
  }
}

}  // namespace polatory::preconditioner
