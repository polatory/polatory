#pragma once

#include <Eigen/Core>
#include <iterator>
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
    throw std::invalid_argument("all matrices must have the same number of columns");
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
    throw std::invalid_argument("all matrices must have the same number of rows");
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

}  // namespace polatory::common
