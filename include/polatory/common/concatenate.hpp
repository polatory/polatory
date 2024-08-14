#pragma once

#include <Eigen/Core>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>

namespace polatory::common {

namespace internal {

template <class Derived>
Index common_cols(const Eigen::MatrixBase<Derived>& m) {
  return m.cols();
}

template <class Derived, class... Args>
Index common_cols(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
  if (m.cols() != common_cols(std::forward<Args>(args)...)) {
    throw std::invalid_argument("the matrices must have the same number of columns");
  }

  return m.cols();
}

template <class Derived>
Index common_rows(const Eigen::MatrixBase<Derived>& m) {
  return m.rows();
}

template <class Derived, class... Args>
Index common_rows(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
  if (m.rows() != common_rows(std::forward<Args>(args)...)) {
    throw std::invalid_argument("the matrices must have the same number of rows");
  }

  return m.rows();
}

template <class DerivedResult, class DerivedM>
void concatenate_cols_impl(Eigen::MatrixBase<DerivedResult>& result,
                           const Eigen::MatrixBase<DerivedM>& m) {
  result = m;
}

template <class DerivedResult, class DerivedM, class... Args>
void concatenate_cols_impl(Eigen::MatrixBase<DerivedResult>& result,
                           const Eigen::MatrixBase<DerivedM>& m, Args&&... args) {
  result.leftCols(m.cols()) = m;

  auto result_rest = result.rightCols(result.cols() - m.cols());
  concatenate_cols_impl(result_rest, std::forward<Args>(args)...);
}

template <class DerivedResult, class DerivedM>
void concatenate_rows_impl(Eigen::MatrixBase<DerivedResult>& result,
                           const Eigen::MatrixBase<DerivedM>& m) {
  result = m;
}

template <class DerivedResult, class DerivedM, class... Args>
void concatenate_rows_impl(Eigen::MatrixBase<DerivedResult>& result,
                           const Eigen::MatrixBase<DerivedM>& m, Args&&... args) {
  result.topRows(m.rows()) = m;

  auto result_rest = result.bottomRows(result.rows() - m.rows());
  concatenate_rows_impl(result_rest, std::forward<Args>(args)...);
}

}  // namespace internal

template <class Result, class... Args>
Result concatenate_cols(Args&&... args) {
  Result result(internal::common_rows(std::forward<Args>(args)...), (args.cols() + ...));

  internal::concatenate_cols_impl(result, std::forward<Args>(args)...);

  return result;
}

template <class Result, class... Args>
Result concatenate_rows(Args&&... args) {
  Result result((args.rows() + ...), internal::common_cols(std::forward<Args>(args)...));

  internal::concatenate_rows_impl(result, std::forward<Args>(args)...);

  return result;
}

}  // namespace polatory::common
