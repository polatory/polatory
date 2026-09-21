#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/condition_number.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/preconditioner/mat_a.hpp>
#include <polatory/preconditioner/mat_q.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::preconditioner {

template <int Dim>
class CoarseGrid {
  static constexpr int kDim = Dim;
  using Domain = Domain<kDim>;
  using MatQ = MatQ<kDim>;
  using Model = Model<kDim>;
  using MonomialBasis = polynomial::MonomialBasis<kDim>;
  using Points = geometry::Points<kDim>;

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  CoarseGrid(const Model& model, Domain&& domain)
      : model_(model),
        point_idcs_(std::move(domain.point_indices)),
        grad_point_idcs_(std::move(domain.grad_point_indices)),
        l_(model.poly_basis_size()),
        mu_(static_cast<Index>(point_idcs_.size())),
        sigma_(static_cast<Index>(grad_point_idcs_.size())),
        m_(mu_ + kDim * sigma_) {}

  double condition_number() const { return cond_; }

  void setup(const Points& points_full, const Points& grad_points_full,
             bool compute_condition_number = false) {
    Points points = points_full(point_idcs_, kAll);
    Points grad_points = grad_points_full(grad_point_idcs_, kAll);

    MatQ mat_q(model_.poly_degree(), points, grad_points);
    if (mat_q.rank() != l_) {
      throw std::runtime_error("the coarse points are not unisolvent");
    }
    indices_ = mat_q.indices();
    q_top_ = mat_q.top();

    MatX a = mat_a(model_, points, grad_points)(indices_, indices_);

    if (m_ > l_) {
      MatX qtaq = a.bottomRightCorner(m_ - l_, m_ - l_);
      qtaq.noalias() += q_top_.transpose() * (a.topLeftCorner(l_, l_) * q_top_);
      qtaq.noalias() += q_top_.transpose() * a.topRightCorner(l_, m_ - l_);
      qtaq.noalias() += a.bottomLeftCorner(m_ - l_, l_) * q_top_;
      if (compute_condition_number) {
        cond_ = numeric::condition_number(qtaq);
      }
      ldlt_of_qtaq_ = qtaq.ldlt();
    }

    if (l_ > 0) {
      // Compute matrices used for solving the polynomial part.
      a_top_ = a.topRows(l_);

      std::vector<Index> special(indices_.begin(), indices_.begin() + l_);
      MatX p_top = MonomialBasis(model_.poly_degree()).evaluate(points, grad_points)(special, kAll);
      lu_of_p_top_ = p_top.fullPivLu();
    }

    mu_full_ = points_full.rows();
    sigma_full_ = grad_points_full.rows();
  }

  void set_solution_to(Eigen::Ref<VecX> weights_full) const {
    weights_full(point_idcs_) = lambda_c_.head(mu_);

    weights_full.segment(mu_full_, kDim * sigma_full_)
        .reshaped<Eigen::RowMajor>(sigma_full_, kDim)(grad_point_idcs_, kAll) =
        lambda_c_.segment(mu_, kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim);

    weights_full.tail(l_) = lambda_c_.tail(l_);
  }

  void solve(const Eigen::Ref<const VecX>& values_full) {
    VecX values(m_);
    values.head(mu_) = values_full(point_idcs_);
    values.tail(kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim) =
        values_full.tail(kDim * sigma_full_)
            .reshaped<Eigen::RowMajor>(sigma_full_, kDim)(grad_point_idcs_, kAll);
    VecX ordered_values = values(indices_);

    VecX ordered_lambda = VecX::Zero(m_);

    if (m_ > l_) {
      // Compute Q^T d.
      VecX qtd = q_top_.transpose() * ordered_values.head(l_) + ordered_values.tail(m_ - l_);

      // Solve Q^T A Q gamma = Q^T d for gamma.
      VecX gamma = ldlt_of_qtaq_.solve(qtd);

      // Compute lambda = Q gamma.
      ordered_lambda.head(l_) = q_top_ * gamma;
      ordered_lambda.tail(m_ - l_) = gamma;
    }

    VecX lambda(m_);
    lambda(indices_) = ordered_lambda;
    lambda_c_ = VecX(m_ + l_);
    lambda_c_.head(m_) = lambda;

    if (l_ > 0) {
      // Solve P c = d - A lambda for c at the special functionals.
      VecX a_top_lambda = a_top_ * ordered_lambda;
      lambda_c_.tail(l_) = lu_of_p_top_.solve(ordered_values.head(l_) - a_top_lambda);
    }
  }

 private:
  const Model& model_;
  const std::vector<Index> point_idcs_;
  const std::vector<Index> grad_point_idcs_;

  const Index l_;
  const Index mu_;
  const Index sigma_;
  const Index m_;
  Index mu_full_{};
  Index sigma_full_{};
  double cond_{};

  // Local row indices with the special functionals first.
  std::vector<Index> indices_;

  // Matrix l rows of matrix Q.
  MatX q_top_;

  // LDLT decomposition of matrix Q^T A Q.
  Eigen::LDLT<MatX> ldlt_of_qtaq_;

  // First l rows of matrix A.
  MatX a_top_;

  // LU decomposition of the top part of matrix P.
  Eigen::FullPivLU<MatX> lu_of_p_top_;

  // Current solution.
  VecX lambda_c_;
};

}  // namespace polatory::preconditioner
