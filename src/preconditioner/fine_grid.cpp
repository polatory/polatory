// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/preconditioner/fine_grid.hpp>

#include <cassert>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/iterator_range.hpp>

namespace polatory {
namespace preconditioner {

fine_grid::fine_grid(const rbf::rbf& rbf,
                     std::shared_ptr<polynomial::lagrange_basis> lagrange_basis,
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

fine_grid::fine_grid(const rbf::rbf& rbf,
                     std::shared_ptr<polynomial::lagrange_basis> lagrange_basis,
                     const std::vector<size_t>& point_indices,
                     const std::vector<bool>& inner_point,
                     const geometry::points3d& points_full)
  : fine_grid(rbf, lagrange_basis, point_indices, inner_point) {
  setup(points_full);
}

void fine_grid::clear() {
  me_ = Eigen::MatrixXd();
  ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();
}

void fine_grid::setup(const geometry::points3d& points_full) {
  // Compute A.
  Eigen::MatrixXd a(m_, m_);
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
    auto tail_points = common::take_rows(points_full, common::make_range(point_idcs_.begin() + l_, point_idcs_.end()));
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

} // namespace preconditioner
} // namespace polatory
