#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/types.hpp>
#include <utility>
#include <vector>

#include "mat_a.hpp"

namespace polatory::preconditioner {

template <class Model>
class fine_grid {
  static constexpr int kDim = Model::kDim;
  using Domain = domain<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  fine_grid(const Model& model, Domain&& domain)
      : model_(model),
        point_idcs_(std::move(domain.point_indices)),
        grad_point_idcs_(std::move(domain.grad_point_indices)),
        inner_point_(std::move(domain.inner_point)),
        inner_grad_point_(std::move(domain.inner_grad_point)),
        l_(model.poly_basis_size()),
        mu_(static_cast<index_t>(point_idcs_.size())),
        sigma_(static_cast<index_t>(grad_point_idcs_.size())),
        m_(mu_ + kDim * sigma_) {
    POLATORY_ASSERT(mu_ > l_);
  }

  void clear() {
    me_ = Eigen::MatrixXd();
    ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();
  }

  void setup(const Points& points_full, const Points& grad_points_full,
             const Eigen::MatrixXd& lagrange_pt_full) {
    auto points = points_full(point_idcs_, Eigen::all);
    auto grad_points = grad_points_full(grad_point_idcs_, Eigen::all);

    // Compute A.
    auto a = mat_a(model_, points, grad_points);

    if (l_ > 0) {
      std::vector<index_t> flat_indices(point_idcs_);
      flat_indices.reserve(mu_ + kDim * sigma_);
      for (auto i : grad_point_idcs_) {
        for (index_t j = 0; j < kDim; j++) {
          flat_indices.push_back(mu_ + kDim * i + j);
        }
      }

      // Compute -E.
      auto lagrange_pt = lagrange_pt_full(Eigen::all, flat_indices);
      me_ = -lagrange_pt.rightCols(m_ - l_);

      Eigen::MatrixXd met = me_.transpose();

      // Compute decomposition of Q^T A Q.
      ldlt_of_qtaq_ =
          (met * a.topLeftCorner(l_, l_) * me_ + met * a.topRightCorner(l_, m_ - l_) +
           a.bottomLeftCorner(m_ - l_, l_) * me_ + a.bottomRightCorner(m_ - l_, m_ - l_))
              .ldlt();
    } else {
      ldlt_of_qtaq_ = a.ldlt();
    }

    mu_full_ = points_full.rows();
    sigma_full_ = grad_points_full.rows();
  }

  void set_solution_to(common::valuesd& weights_full) const {
    for (index_t i = 0; i < mu_; i++) {
      if (inner_point_.at(i)) {
        weights_full(point_idcs_.at(i)) = lambda_(i);
      }
    }

    for (index_t i = 0; i < sigma_; i++) {
      if (inner_grad_point_.at(i)) {
        weights_full.segment(mu_full_ + kDim * grad_point_idcs_.at(i), kDim) =
            lambda_.segment(mu_ + kDim * i, kDim);
      }
    }
  }

  void solve(const common::valuesd& values_full) {
    common::valuesd values(m_);
    values.head(mu_) = values_full(point_idcs_);
    values.tail(kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim) =
        values_full.tail(kDim * sigma_full_)
            .reshaped<Eigen::RowMajor>(sigma_full_, kDim)(grad_point_idcs_, Eigen::all);

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

 private:
  const Model& model_;
  const std::vector<index_t> point_idcs_;
  const std::vector<index_t> grad_point_idcs_;
  const std::vector<bool> inner_point_;
  const std::vector<bool> inner_grad_point_;

  const index_t l_;
  const index_t mu_;
  const index_t sigma_;
  const index_t m_;
  index_t mu_full_;
  index_t sigma_full_;

  // Matrix -E.
  Eigen::MatrixXd me_;

  // Cholesky decomposition of matrix Q^T A Q.
  Eigen::LDLT<Eigen::MatrixXd> ldlt_of_qtaq_;

  // Current solution.
  common::valuesd lambda_;
};

}  // namespace polatory::preconditioner
