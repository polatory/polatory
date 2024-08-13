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

template <int Dim>
class CoarseGrid {
  static constexpr int kDim = Dim;
  using Domain = Domain<kDim>;
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
        m_(mu_ + kDim * sigma_) {
    POLATORY_ASSERT(mu_ >= l_ || (model_.poly_degree() == 1 && mu_ == 1 && sigma_ >= 1));
  }

  void setup(const Points& points_full, const Points& grad_points_full,
             const MatX& lagrange_p_full) {
    Points points = points_full(point_idcs_, Eigen::all);
    Points grad_points = grad_points_full(grad_point_idcs_, Eigen::all);

    // Compute A.
    auto a = mat_a(model_, points, grad_points);

    if (l_ > 0) {
      if (m_ > l_) {
        std::vector<Index> flat_indices(point_idcs_);
        flat_indices.reserve(mu_ + kDim * sigma_);
        for (auto i : grad_point_idcs_) {
          for (Index j = 0; j < kDim; j++) {
            flat_indices.push_back(mu_ + kDim * i + j);
          }
        }

        // Compute matrix Q.
        auto lagrange_p = lagrange_p_full(flat_indices, Eigen::all);
        q_top_ = -lagrange_p.bottomRows(m_ - l_).transpose();

        // Compute decomposition of Q^T A Q.
        ldlt_of_qtaq_ =
            (q_top_.transpose() * a.topLeftCorner(l_, l_) * q_top_ +
             q_top_.transpose() * a.topRightCorner(l_, m_ - l_) +
             a.bottomLeftCorner(m_ - l_, l_) * q_top_ + a.bottomRightCorner(m_ - l_, m_ - l_))
                .ldlt();
      }

      // Compute matrices used for solving the polynomial part.
      a_top_ = a.topRows(l_);

      MonomialBasis mono_basis(model_.poly_degree());
      MatX p_top;
      if (model_.poly_degree() == 1 && mu_ == 1 && sigma_ >= 1) {
        // The special case.
        p_top = mono_basis.evaluate(points, grad_points.topRows(1));
      } else {
        // The ordinary case.
        p_top = mono_basis.evaluate(points.topRows(l_));
      }
      lu_of_p_top_ = p_top.fullPivLu();
    } else {
      ldlt_of_qtaq_ = a.ldlt();
    }

    mu_full_ = points_full.rows();
    sigma_full_ = grad_points_full.rows();
  }

  void set_solution_to(Eigen::Ref<VecX> weights_full) const {
    weights_full(point_idcs_) = lambda_c_.head(mu_);

    weights_full.segment(mu_full_, kDim * sigma_full_)
        .reshaped<Eigen::RowMajor>(sigma_full_, kDim)(grad_point_idcs_, Eigen::all) =
        lambda_c_.segment(mu_, kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim);

    weights_full.tail(l_) = lambda_c_.tail(l_);
  }

  void solve(const Eigen::Ref<const VecX>& values_full) {
    VecX values(m_);
    values.head(mu_) = values_full(point_idcs_);
    values.tail(kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim) =
        values_full.tail(kDim * sigma_full_)
            .reshaped<Eigen::RowMajor>(sigma_full_, kDim)(grad_point_idcs_, Eigen::all);

    if (l_ > 0) {
      lambda_c_ = VecX(m_ + l_);

      if (m_ > l_) {
        // Compute Q^T d.
        VecX qtd = q_top_.transpose() * values.head(l_) + values.tail(m_ - l_);

        // Solve Q^T A Q gamma = Q^T d for gamma.
        VecX gamma = ldlt_of_qtaq_.solve(qtd);

        // Compute lambda = Q gamma.
        lambda_c_.head(l_) = q_top_ * gamma;
        lambda_c_.segment(l_, m_ - l_) = gamma;
      } else {
        lambda_c_.head(m_) = VecX::Zero(m_);
      }

      // Solve P c = d - A lambda for c at poly_points.
      VecX a_top_lambda = a_top_ * lambda_c_.head(m_);
      lambda_c_.tail(l_) = lu_of_p_top_.solve(values.head(l_) - a_top_lambda);
    } else {
      lambda_c_ = ldlt_of_qtaq_.solve(values);
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

  // First l rows of matrix Q.
  MatX q_top_;

  // Cholesky decomposition of matrix Q^T A Q.
  Eigen::LDLT<MatX> ldlt_of_qtaq_;

  // First l rows of matrix A.
  MatX a_top_;

  // LU decomposition of first l rows of matrix P.
  Eigen::FullPivLU<MatX> lu_of_p_top_;

  // Current solution.
  VecX lambda_c_;
};

}  // namespace polatory::preconditioner
