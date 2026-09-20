#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <limits>
#include <polatory/preconditioner/binary_cache.hpp>
#include <polatory/types.hpp>

namespace polatory::preconditioner {

class CachedLdlt {
  using RowVecX = Mat<1, Eigen::Dynamic>;

 public:
  explicit CachedLdlt(BinaryCache& cache) : cache_(cache) {}

  void compute(const MatX& a) {
    Eigen::LDLT<MatX> ldlt(a);
    n_ = a.rows();
    p_ = ldlt.transpositionsP();

    const auto& coeffs = ldlt.matrixLDLT();
    reserve_packed();
    for (Index i = 0; i < n_; i++) {
      packed_.segment(row_offset(i), i + 1) = coeffs.row(i).head(i + 1);
    }

    cache_id_ = cache_.put(packed_.data(), packed_size() * sizeof(double));
  }

  VecX solve(const VecX& b) const {
    reserve_packed();
    cache_.get(cache_id_, packed_.data());

    VecX x = p_ * b;

    for (Index i = 1; i < n_; i++) {
      auto row = packed_.segment(row_offset(i), i);
      x(i) -= row.dot(x.head(i));
    }

    for (Index i = 0; i < n_; i++) {
      auto d = packed_(row_offset(i) + i);
      x(i) = std::abs(d) > std::numeric_limits<double>::min() ? x(i) / d : 0.0;
    }

    for (Index i = n_ - 1; i >= 1; i--) {
      auto row = packed_.segment(row_offset(i), i);
      x.head(i) -= x(i) * row.transpose();
    }

    return p_.transpose() * x;
  }

 private:
  Index packed_size() const { return row_offset(n_); }

  void reserve_packed() const {
    if (packed_.size() < packed_size()) {
      packed_.resize(packed_size());
    }
  }

  static Index row_offset(Index i) { return i * (i + 1) / 2; }

  static inline thread_local RowVecX packed_;
  BinaryCache& cache_;
  std::size_t cache_id_{};
  Index n_{};
  Eigen::Transpositions<Eigen::Dynamic> p_;
};

}  // namespace polatory::preconditioner
