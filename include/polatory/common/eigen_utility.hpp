#pragma once

#include <Eigen/Core>
#include <iterator>
#include <polatory/common/iterator_range.hpp>
#include <polatory/common/macros.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::common {

namespace detail {

template <class Derived>
Eigen::Index common_cols(const Eigen::MatrixBase<Derived>& m) {
  return m.cols();
}

template <class Derived, class... Args>
Eigen::Index common_cols(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
  if (m.cols() != common_cols(std::forward<Args>(args)...)) {
    throw std::invalid_argument("All inputs must have the same number of columns.");
  }

  return m.cols();
}

template <class Derived>
Eigen::Index common_rows(const Eigen::MatrixBase<Derived>& m) {
  return m.rows();
}

template <class Derived, class... Args>
Eigen::Index common_rows(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
  if (m.rows() != common_rows(std::forward<Args>(args)...)) {
    throw std::invalid_argument("All inputs must have the same number of rows.");
  }

  return m.rows();
}

template <class ResultDerived, class Derived>
void concatenate_cols_impl(Eigen::MatrixBase<ResultDerived>& result,
                           const Eigen::MatrixBase<Derived>& m) {
  result = m;
}

template <class ResultDerived, class Derived, class... Args>
void concatenate_cols_impl(Eigen::MatrixBase<ResultDerived>& result,
                           const Eigen::MatrixBase<Derived>& m, Args&&... args) {
  result.leftCols(m.cols()) = m;

  auto result_tail = result.rightCols(result.cols() - m.cols());
  concatenate_cols_impl(result_tail, std::forward<Args>(args)...);
}

template <class ResultDerived, class Derived>
void concatenate_rows_impl(Eigen::MatrixBase<ResultDerived>& result,
                           const Eigen::MatrixBase<Derived>& m) {
  result = m;
}

template <class ResultDerived, class Derived, class... Args>
void concatenate_rows_impl(Eigen::MatrixBase<ResultDerived>& result,
                           const Eigen::MatrixBase<Derived>& m, Args&&... args) {
  result.topRows(m.rows()) = m;

  auto result_tail = result.bottomRows(result.rows() - m.rows());
  concatenate_rows_impl(result_tail, std::forward<Args>(args)...);
}

template <class ResultDerived, class Derived>
void take_cols_impl(Eigen::MatrixBase<ResultDerived>& result, const Eigen::MatrixBase<Derived>& m,
                    Eigen::Index index) {
  result.col(0) = m.col(index);
}

template <class ResultDerived, class Derived, class... Ts>
void take_cols_impl(Eigen::MatrixBase<ResultDerived>& result, const Eigen::MatrixBase<Derived>& m,
                    Eigen::Index index, Ts... indices) {
  result.col(0) = m.col(index);

  auto result_tail = result.rightCols(result.cols() - 1);
  take_cols_impl(result_tail, m, indices...);
}

template <class ResultDerived, class Derived>
void take_rows_impl(Eigen::MatrixBase<ResultDerived>& result, const Eigen::MatrixBase<Derived>& m,
                    Eigen::Index index) {
  result.row(0) = m.row(index);
}

template <class ResultDerived, class Derived, class... Ts>
void take_rows_impl(Eigen::MatrixBase<ResultDerived>& result, const Eigen::MatrixBase<Derived>& m,
                    Eigen::Index index, Ts... indices) {
  result.row(0) = m.row(index);

  auto result_tail = result.bottomRows(result.rows() - 1);
  take_rows_impl(result_tail, m, indices...);
}

}  // namespace detail

template <class... Args>
auto concatenate_cols(Args&&... args) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      detail::common_rows(std::forward<Args>(args)...), (args.cols() + ...));

  detail::concatenate_cols_impl(result, std::forward<Args>(args)...);

  return result;
}

template <class... Args>
auto concatenate_rows(Args&&... args) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      (args.rows() + ...), detail::common_cols(std::forward<Args>(args)...));

  detail::concatenate_rows_impl(result, std::forward<Args>(args)...);

  return result;
}

template <class Derived, class... Ts>
auto take_cols(const Eigen::MatrixBase<Derived>& m, Ts... indices) {
  Eigen::Matrix<typename Eigen::MatrixBase<Derived>::Scalar,
                Eigen::MatrixBase<Derived>::RowsAtCompileTime, sizeof...(indices),
                Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor>
  result(m.rows(), sizeof...(indices));

  detail::take_cols_impl(result, m, indices...);

  return result;
}

template <class Derived, class ForwardRange>
auto take_cols(const Eigen::MatrixBase<Derived>& m, ForwardRange indices) {
  Eigen::Index n_cols = std::distance(indices.begin(), indices.end());

  Eigen::Matrix<typename Eigen::MatrixBase<Derived>::Scalar,
                Eigen::MatrixBase<Derived>::RowsAtCompileTime, Eigen::Dynamic,
                Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor>
      result(m.rows(), n_cols);

  auto it = indices.begin();
  for (Eigen::Index i = 0; i < n_cols; i++) {
    result.col(i) = m.col(*it++);
  }

  return result;
}

template <class Derived>
auto take_cols(const Eigen::MatrixBase<Derived>& m, const std::vector<Eigen::Index>& indices) {
  return take_cols(m, make_range(indices.begin(), indices.end()));
}

template <class Derived, class... Ts>
auto take_rows(const Eigen::MatrixBase<Derived>& m, Ts... indices) {
  Eigen::Matrix<typename Eigen::MatrixBase<Derived>::Scalar, sizeof...(indices),
                Eigen::MatrixBase<Derived>::ColsAtCompileTime,
                Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor>
  result(sizeof...(indices), m.cols());

  detail::take_rows_impl(result, m, indices...);

  return result;
}

template <class Derived, class ForwardRange>
auto take_rows(const Eigen::MatrixBase<Derived>& m, ForwardRange indices) {
  Eigen::Index n_rows = std::distance(indices.begin(), indices.end());

  Eigen::Matrix<typename Eigen::MatrixBase<Derived>::Scalar, Eigen::Dynamic,
                Eigen::MatrixBase<Derived>::ColsAtCompileTime,
                Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor>
      result(n_rows, m.cols());

  auto it = indices.begin();
  for (Eigen::Index i = 0; i < n_rows; i++) {
    result.row(i) = m.row(*it++);
  }

  return result;
}

template <class Derived>
auto take_rows(const Eigen::MatrixBase<Derived>& m, const std::vector<Eigen::Index>& indices) {
  return take_rows(m, make_range(indices.begin(), indices.end()));
}

}  // namespace polatory::common
