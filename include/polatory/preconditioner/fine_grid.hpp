#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <cstring>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/condition_number.hpp>
#include <polatory/preconditioner/binary_cache.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/preconditioner/mat_a.hpp>
#include <polatory/preconditioner/mat_q.hpp>
#include <polatory/types.hpp>
#include <utility>
#include <vector>

namespace Eigen {

template <typename MatrixType_, int UpLo_ = Eigen::Lower>
class LDLT2 : public LDLT<MatrixType_, UpLo_> {
 public:
  using Base = LDLT<MatrixType_, UpLo_>;
  using MatrixType = Base::MatrixType;
  using Base::Base;

  MatrixType& matrixLDLT() {
    eigen_assert(Base::m_isInitialized && "LDLT is not initialized.");
    return this->m_matrix;
  }
};

}  // namespace Eigen

namespace polatory::preconditioner {

template <int Dim>
class FineGrid {
  static constexpr int kDim = Dim;
  using Domain = Domain<kDim>;
  using MatQ = MatQ<kDim>;
  using Model = Model<kDim>;
  using Points = geometry::Points<kDim>;

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  FineGrid(const Model& model, Domain&& domain, BinaryCache& cache)
      : model_(model),
        point_idcs_(std::move(domain.point_indices)),
        grad_point_idcs_(std::move(domain.grad_point_indices)),
        inner_point_(std::move(domain.inner_point)),
        inner_grad_point_(std::move(domain.inner_grad_point)),
        cache_(cache),
        mu_(static_cast<Index>(point_idcs_.size())),
        sigma_(static_cast<Index>(grad_point_idcs_.size())),
        m_(mu_ + kDim * sigma_) {}

  double condition_number() const { return cond_; }

  void setup(const Points& points_full, const Points& grad_points_full,
             bool compute_condition_number = false) {
    Points points = points_full(point_idcs_, kAll);
    Points grad_points = grad_points_full(grad_point_idcs_, kAll);

    MatQ mat_q(model_.poly_degree(), points, grad_points);
    l_ = mat_q.rank();
    indices_ = mat_q.indices();
    q_top_ = mat_q.top();

    MatX a = mat_a(model_, points, grad_points)(indices_, indices_);

    if (m_ > l_) {
      MatX qtaq = q_top_.transpose() * a.topLeftCorner(l_, l_) * q_top_ +
                  q_top_.transpose() * a.topRightCorner(l_, m_ - l_) +
                  a.bottomLeftCorner(m_ - l_, l_) * q_top_ + a.bottomRightCorner(m_ - l_, m_ - l_);
      if (compute_condition_number) {
        cond_ = numeric::condition_number(qtaq);
      }
      ldlt_of_qtaq_ = Eigen::LDLT2<MatX>(qtaq);
      save_ldlt_of_qtaq();
    }

    mu_full_ = points_full.rows();
    sigma_full_ = grad_points_full.rows();
  }

  void set_solution_to(VecX& weights_full) const {
    for (Index i = 0; i < mu_; i++) {
      if (inner_point_.at(i)) {
        weights_full(point_idcs_.at(i)) = lambda_(i);
      }
    }

    for (Index i = 0; i < sigma_; i++) {
      if (inner_grad_point_.at(i)) {
        weights_full.segment<kDim>(mu_full_ + kDim * grad_point_idcs_.at(i)) =
            lambda_.segment<kDim>(mu_ + kDim * i);
      }
    }
  }

  void solve(const VecX& values_full) {
    VecX values(m_);
    values.head(mu_) = values_full(point_idcs_);
    values.tail(kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim) =
        values_full.tail(kDim * sigma_full_)
            .reshaped<Eigen::RowMajor>(sigma_full_, kDim)(grad_point_idcs_, kAll);
    VecX ordered_values = values(indices_);

    lambda_ = VecX::Zero(m_);

    if (m_ > l_) {
      // Compute Q^T d.
      VecX qtd = q_top_.transpose() * ordered_values.head(l_) + ordered_values.tail(m_ - l_);

      // Solve Q^T A Q gamma = Q^T d for gamma.
      load_ldlt_of_qtaq();
      VecX gamma = ldlt_of_qtaq_.solve(qtd);
      ldlt_of_qtaq_.matrixLDLT().resize(0, 0);

      // Compute lambda = Q gamma.
      VecX ordered_lambda(m_);
      ordered_lambda.head(l_) = q_top_ * gamma;
      ordered_lambda.tail(m_ - l_) = gamma;
      lambda_(indices_) = ordered_lambda;
    }
  }

 private:
  void load_ldlt_of_qtaq() {
    auto& ldlt = ldlt_of_qtaq_.matrixLDLT();
    ldlt.resize(m_ - l_, m_ - l_);
    cache_.get(cache_id_, ldlt.data());
    // Unpack the lower triangular part.
    for (auto row = ldlt.rows() - 1; row >= 1; row--) {
      const auto* src = ldlt.data() + row * (row + 1) / 2;
      auto* dst = ldlt.data() + row * ldlt.cols();
      auto bytes = (row + 1) * sizeof(double);
      std::memcpy(dst, src, bytes);
    }
  }

  void save_ldlt_of_qtaq() {
    auto& ldlt = ldlt_of_qtaq_.matrixLDLT();
    // Pack the lower triangular part.
    auto rows = ldlt.rows();
    for (Index row = 1; row < rows; row++) {
      const auto* src = ldlt.data() + row * ldlt.cols();
      auto* dst = ldlt.data() + row * (row + 1) / 2;
      auto bytes = (row + 1) * sizeof(double);
      std::memcpy(dst, src, bytes);
    }
    cache_id_ = cache_.put(ldlt.data(), (rows * (rows + 1) / 2) * sizeof(double));
    ldlt.resize(0, 0);
  }

  const Model& model_;
  const std::vector<Index> point_idcs_;
  const std::vector<Index> grad_point_idcs_;
  const std::vector<bool> inner_point_;
  const std::vector<bool> inner_grad_point_;
  BinaryCache& cache_;
  std::size_t cache_id_{};

  const Index mu_;
  const Index sigma_;
  const Index m_;
  Index l_{};
  Index mu_full_{};
  Index sigma_full_{};
  double cond_{};

  // Local row indices with the special functionals first.
  std::vector<Index> indices_;

  // First l rows of matrix Q.
  MatX q_top_;

  // Cholesky decomposition of matrix Q^T A Q.
  Eigen::LDLT2<MatX> ldlt_of_qtaq_;

  // Current solution.
  VecX lambda_;
};

}  // namespace polatory::preconditioner
