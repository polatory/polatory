#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/types.hpp>
#include <utility>
#include <vector>

#include "mat_a.hpp"

namespace polatory::preconditioner {

template <class Model>
class coarse_grid {
 public:
  coarse_grid(const Model& model, domain&& domain)
      : model_(model),
        point_idcs_(std::move(domain.point_indices)),
        grad_point_idcs_(std::move(domain.grad_point_indices)),
        dim_(model.poly_dimension()),
        l_(model.poly_basis_size()),
        mu_(static_cast<index_t>(point_idcs_.size())),
        sigma_(static_cast<index_t>(grad_point_idcs_.size())),
        m_(mu_ + dim_ * sigma_) {
    POLATORY_ASSERT(mu_ > l_);
  }

  void clear() {
    me_ = Eigen::MatrixXd();
    ldlt_of_qtaq_ = Eigen::LDLT<Eigen::MatrixXd>();

    a_top_ = Eigen::MatrixXd();
    lu_of_p_top_ = Eigen::FullPivLU<Eigen::MatrixXd>();
  }

  void setup(const geometry::points3d& points_full, const geometry::points3d& grad_points_full,
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

      // Compute decomposition of Q^T A Q.
      ldlt_of_qtaq_ =
          (me_.transpose() * a.topLeftCorner(l_, l_) * me_ +
           me_.transpose() * a.topRightCorner(l_, m_ - l_) + a.bottomLeftCorner(m_ - l_, l_) * me_ +
           a.bottomRightCorner(m_ - l_, m_ - l_))
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
    sigma_full_ = grad_points_full.rows();
  }

  void set_solution_to(Eigen::Ref<common::valuesd> weights_full) const {
    weights_full(point_idcs_) = lambda_c_.head(mu_);

    weights_full.segment(mu_full_, dim_ * sigma_full_)
        .reshaped<Eigen::RowMajor>(sigma_full_, dim_)(grad_point_idcs_, Eigen::all) =
        lambda_c_.segment(mu_, dim_ * sigma_).reshaped<Eigen::RowMajor>(sigma_, dim_);

    weights_full.tail(l_) = lambda_c_.tail(l_);
  }

  void solve(const Eigen::Ref<const common::valuesd>& values_full) {
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

 private:
  const Model& model_;
  const std::vector<index_t> point_idcs_;
  const std::vector<index_t> grad_point_idcs_;

  const int dim_;
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

  // First l rows of matrix A.
  Eigen::MatrixXd a_top_;

  // LU decomposition of first l rows of matrix P.
  Eigen::FullPivLU<Eigen::MatrixXd> lu_of_p_top_;

  // Current solution.
  common::valuesd lambda_c_;
};

}  // namespace polatory::preconditioner
