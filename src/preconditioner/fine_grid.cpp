#include <polatory/preconditioner/fine_grid.hpp>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/iterator_range.hpp>
#include <polatory/common/macros.hpp>

namespace polatory {
namespace preconditioner {

fine_grid::fine_grid(const model& model,
                     const std::unique_ptr<polynomial::lagrange_basis>& lagrange_basis,
                     const std::vector<index_t>& point_indices,
                     const std::vector<bool>& inner_point)
  : model_(model)
  , lagrange_basis_(lagrange_basis)
  , point_idcs_(point_indices)
  , inner_point_(inner_point)
  , l_(lagrange_basis ? lagrange_basis->basis_size() : 0)
  , m_(static_cast<index_t>(point_indices.size())) {
  POLATORY_ASSERT(m_ > l_);
}

fine_grid::fine_grid(const model& model,
                     const std::unique_ptr<polynomial::lagrange_basis>& lagrange_basis,
                     const std::vector<index_t>& point_indices,
                     const std::vector<bool>& inner_point,
                     const geometry::points3d& points_full)
  : fine_grid(model, lagrange_basis, point_indices, inner_point) {
  setup(points_full);
}

void fine_grid::clear() {
  me_ = Eigen::MatrixXd();
  ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();  // NOLINT(clang-analyzer-core.uninitialized.Assign)
}

void fine_grid::setup(const geometry::points3d& points_full) {
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
    me_ = -lagrange_basis_->evaluate(tail_points);

    // Compute decomposition of Q^T A Q.
    ldlt_of_qtaq_ = (me_.transpose() * a.topLeftCorner(l_, l_) * me_
                     + me_.transpose() * a.topRightCorner(l_, m_ - l_)
                     + a.bottomLeftCorner(m_ - l_, l_) * me_
                     + a.bottomRightCorner(m_ - l_, m_ - l_)).ldlt();
  } else {
    ldlt_of_qtaq_ = a.ldlt();
  }
}

void fine_grid::set_solution_to(Eigen::Ref<common::valuesd> weights_full) const {
  for (index_t i = 0; i < m_; i++) {
    if (inner_point_[i])
      weights_full(point_idcs_[i]) = lambda_(i);
  }
}

void fine_grid::solve(const Eigen::Ref<const common::valuesd>& values_full) {
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
    lambda_ = common::valuesd(m_);
    lambda_.head(l_) = me_ * gamma;
    lambda_.tail(m_ - l_) = gamma;
  } else {
    lambda_ = ldlt_of_qtaq_.solve(values);
  }
}

}  // namespace preconditioner
}  // namespace polatory
