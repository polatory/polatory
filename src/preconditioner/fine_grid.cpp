#include <polatory/common/macros.hpp>
#include <polatory/preconditioner/fine_grid.hpp>
#include <utility>

#include "mat_a.hpp"

namespace polatory::preconditioner {

fine_grid::fine_grid(const model& model, domain&& domain)
    : model_(model),
      point_idcs_(std::move(domain.point_indices)),
      grad_point_idcs_(std::move(domain.grad_point_indices)),
      inner_point_(std::move(domain.inner_point)),
      inner_grad_point_(std::move(domain.inner_grad_point)),
      dim_(model.poly_dimension()),
      l_(model.poly_basis_size()),
      mu_(static_cast<index_t>(point_idcs_.size())),
      sigma_(static_cast<index_t>(grad_point_idcs_.size())),
      m_(mu_ + dim_ * sigma_) {
  POLATORY_ASSERT(mu_ > l_);
}

void fine_grid::clear() {
  me_ = Eigen::MatrixXd();
  ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();
}

void fine_grid::setup(const geometry::points3d& points_full,
                      const geometry::points3d& grad_points_full,
                      const Eigen::MatrixXd& lagrange_pt_full) {
  auto points = points_full(point_idcs_, Eigen::all);
  auto grad_points = grad_points_full(grad_point_idcs_, Eigen::all);

  // Compute A.
  auto a = mat_a(model_, points, grad_points);

  if (l_ > 0) {
    std::vector<index_t> flat_indices(point_idcs_);
    flat_indices.reserve(mu_ + dim_ * sigma_);
    for (auto i : grad_point_idcs_) {
      for (index_t j = 0; j < dim_; j++) {
        flat_indices.push_back(mu_ + dim_ * i + j);
      }
    }

    // Compute -E.
    auto lagrange_pt = lagrange_pt_full(Eigen::all, flat_indices);
    me_ = -lagrange_pt.rightCols(m_ - l_);

    Eigen::MatrixXd met = me_.transpose();

    // Compute decomposition of Q^T A Q.
    ldlt_of_qtaq_ = (met * a.topLeftCorner(l_, l_) * me_ + met * a.topRightCorner(l_, m_ - l_) +
                     a.bottomLeftCorner(m_ - l_, l_) * me_ + a.bottomRightCorner(m_ - l_, m_ - l_))
                        .ldlt();
  } else {
    ldlt_of_qtaq_ = a.ldlt();
  }

  mu_full_ = points_full.rows();
  sigma_full_ = grad_points_full.rows();
}

void fine_grid::set_solution_to(common::valuesd& weights_full) const {
  for (index_t i = 0; i < mu_; i++) {
    if (inner_point_.at(i)) {
      weights_full(point_idcs_.at(i)) = lambda_(i);
    }
  }

  for (index_t i = 0; i < sigma_; i++) {
    if (inner_grad_point_.at(i)) {
      weights_full.segment(mu_full_ + dim_ * grad_point_idcs_.at(i), dim_) =
          lambda_.segment(mu_ + dim_ * i, dim_);
    }
  }
}

void fine_grid::solve(const common::valuesd& values_full) {
  common::valuesd values(m_);
  values.head(mu_) = values_full(point_idcs_);
  values.tail(dim_ * sigma_).reshaped<Eigen::RowMajor>(sigma_, dim_) =
      values_full.tail(dim_ * sigma_full_)
          .reshaped<Eigen::RowMajor>(sigma_full_, dim_)(grad_point_idcs_, Eigen::all);

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
