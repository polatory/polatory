#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/preconditioner/binary_cache.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/types.hpp>
#include <utility>
#include <vector>

#include "mat_a.hpp"

namespace Eigen {

template <typename MatrixType_, int UpLo_ = Eigen::Lower>
class LDLT2 : public LDLT<MatrixType_, UpLo_> {
 public:
  using Base = LDLT<MatrixType_, UpLo_>;
  using MatrixType = typename Base::MatrixType;
  using Base::Base;

  inline MatrixType& matrixLDLT() {
    eigen_assert(Base::m_isInitialized && "LDLT is not initialized.");
    return this->m_matrix;
  }
};

}  // namespace Eigen

namespace polatory::preconditioner {

template <int Dim>
class fine_grid {
  static constexpr int kDim = Dim;
  using Domain = domain<kDim>;
  using Model = model<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  fine_grid(const Model& model, Domain&& domain, binary_cache& cache)
      : model_(model),
        point_idcs_(std::move(domain.point_indices)),
        grad_point_idcs_(std::move(domain.grad_point_indices)),
        inner_point_(std::move(domain.inner_point)),
        inner_grad_point_(std::move(domain.inner_grad_point)),
        cache_(cache),
        l_(model.poly_basis_size()),
        mu_(static_cast<index_t>(point_idcs_.size())),
        sigma_(static_cast<index_t>(grad_point_idcs_.size())),
        m_(mu_ + kDim * sigma_) {
    POLATORY_ASSERT(mu_ >= l_ || (model_.poly_degree() == 1 && mu_ == 1 && sigma_ >= 1));
  }

  void setup(const Points& points_full, const Points& grad_points_full,
             const matrixd& lagrange_p_full) {
    Points points = points_full(point_idcs_, Eigen::all);
    Points grad_points = grad_points_full(grad_point_idcs_, Eigen::all);

    // Compute A.
    auto a = mat_a(model_, points, grad_points);

    if (l_ > 0) {
      if (m_ > l_) {
        std::vector<index_t> flat_indices(point_idcs_);
        flat_indices.reserve(mu_ + kDim * sigma_);
        for (auto i : grad_point_idcs_) {
          for (index_t j = 0; j < kDim; j++) {
            flat_indices.push_back(mu_ + kDim * i + j);
          }
        }

        // Compute matrix Q.
        auto lagrange_p = lagrange_p_full(flat_indices, Eigen::all);
        q_top_ = -lagrange_p.bottomRows(m_ - l_).transpose();

        // Compute decomposition of Q^T A Q.
        ldlt_of_qtaq_ = Eigen::LDLT2<matrixd>(
            q_top_.transpose() * a.topLeftCorner(l_, l_) * q_top_ +
            q_top_.transpose() * a.topRightCorner(l_, m_ - l_) +
            a.bottomLeftCorner(m_ - l_, l_) * q_top_ + a.bottomRightCorner(m_ - l_, m_ - l_));
        save_ldlt_of_qtaq();
      }
    } else {
      ldlt_of_qtaq_ = Eigen::LDLT2<matrixd>(a);
      save_ldlt_of_qtaq();
    }

    mu_full_ = points_full.rows();
    sigma_full_ = grad_points_full.rows();
  }

  void set_solution_to(vectord& weights_full) const {
    for (index_t i = 0; i < mu_; i++) {
      if (inner_point_.at(i)) {
        weights_full(point_idcs_.at(i)) = lambda_(i);
      }
    }

    for (index_t i = 0; i < sigma_; i++) {
      if (inner_grad_point_.at(i)) {
        weights_full.segment<kDim>(mu_full_ + kDim * grad_point_idcs_.at(i)) =
            lambda_.segment<kDim>(mu_ + kDim * i);
      }
    }
  }

  void solve(const vectord& values_full) {
    vectord values(m_);
    values.head(mu_) = values_full(point_idcs_);
    values.tail(kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim) =
        values_full.tail(kDim * sigma_full_)
            .reshaped<Eigen::RowMajor>(sigma_full_, kDim)(grad_point_idcs_, Eigen::all);

    if (l_ > 0) {
      lambda_ = vectord(m_);

      if (m_ > l_) {
        // Compute Q^T d.
        vectord qtd = q_top_.transpose() * values.head(l_) + values.tail(m_ - l_);

        // Solve Q^T A Q gamma = Q^T d for gamma.
        load_ldlt_of_qtaq();
        vectord gamma = ldlt_of_qtaq_.solve(qtd);
        ldlt_of_qtaq_.matrixLDLT().resize(0, 0);

        // Compute lambda = Q gamma.
        lambda_.head(l_) = q_top_ * gamma;
        lambda_.tail(m_ - l_) = gamma;
      } else {
        lambda_ = vectord::Zero(m_);
      }
    } else {
      load_ldlt_of_qtaq();
      lambda_ = ldlt_of_qtaq_.solve(values);
      ldlt_of_qtaq_.matrixLDLT().resize(0, 0);
    }
  }

 private:
  void load_ldlt_of_qtaq() {
    auto& ldlt = ldlt_of_qtaq_.matrixLDLT();
    ldlt.resize(m_ - l_, m_ - l_);
    cache_.get(cache_id_, ldlt.data());
  }

  void save_ldlt_of_qtaq() {
    auto& ldlt = ldlt_of_qtaq_.matrixLDLT();
    cache_id_ = cache_.put(ldlt.data(), ldlt.size() * sizeof(double));
    ldlt.resize(0, 0);
  }

  const Model& model_;
  const std::vector<index_t> point_idcs_;
  const std::vector<index_t> grad_point_idcs_;
  const std::vector<bool> inner_point_;
  const std::vector<bool> inner_grad_point_;
  binary_cache& cache_;
  std::size_t cache_id_{};

  const index_t l_;
  const index_t mu_;
  const index_t sigma_;
  const index_t m_;
  index_t mu_full_{};
  index_t sigma_full_{};

  // Matrix l rows of matrix Q.
  matrixd q_top_;

  // Cholesky decomposition of matrix Q^T A Q.
  Eigen::LDLT2<matrixd> ldlt_of_qtaq_;

  // Current solution.
  vectord lambda_;
};

}  // namespace polatory::preconditioner
