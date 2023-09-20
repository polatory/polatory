#include <polatory/common/macros.hpp>
#include <polatory/preconditioner/fine_grid.hpp>

namespace polatory::preconditioner {

fine_grid::fine_grid(const model& model, const std::vector<index_t>& point_indices,
                     const std::vector<bool>& inner_point)
    : model_(model),
      point_idcs_(point_indices),
      inner_point_(inner_point),
      l_(model.poly_basis_size()),
      m_(static_cast<index_t>(point_indices.size())) {
  POLATORY_ASSERT(m_ > l_);
}

void fine_grid::clear() {
  me_ = Eigen::MatrixXd();
  ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();
}

void fine_grid::setup(const geometry::points3d& points_full,
                      const Eigen::MatrixXd& lagrange_pt_full) {
  auto points = points_full(point_idcs_, Eigen::all);
  auto lagrange_pt = lagrange_pt_full(Eigen::all, point_idcs_);

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
    me_ = -lagrange_pt.rightCols(m_ - l_);

    // Compute decomposition of Q^T A Q.
    ldlt_of_qtaq_ = (me_.transpose() * a.topLeftCorner(l_, l_) * me_ +
                     me_.transpose() * a.topRightCorner(l_, m_ - l_) +
                     a.bottomLeftCorner(m_ - l_, l_) * me_ + a.bottomRightCorner(m_ - l_, m_ - l_))
                        .ldlt();
  } else {
    ldlt_of_qtaq_ = a.ldlt();
  }
}

void fine_grid::set_solution_to(Eigen::Ref<common::valuesd> weights_full) const {
  for (index_t i = 0; i < m_; i++) {
    if (inner_point_.at(i)) {
      weights_full(point_idcs_.at(i)) = lambda_(i);
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
