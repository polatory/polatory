// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include <boost/operators.hpp>
#include <Eigen/Core>

#include <polatory/common/exception.hpp>
#include <polatory/common/fold.hpp>
#include <polatory/common/iterator_range.hpp>
#include <polatory/common/macros.hpp>

namespace polatory {
namespace common {

namespace detail {

// NB: These iterator types are NOT models of InputIterator as they do not implement operator->().

template <class Derived>
class col_iterator
  : public boost::random_access_iterator_helper<col_iterator<Derived>, typename Eigen::MatrixBase<Derived>::ColXpr, Eigen::Index, void, typename Eigen::MatrixBase<Derived>::ColXpr> {
  using self_type = col_iterator;

public:
  using iterator_category = std::input_iterator_tag;

  col_iterator(Eigen::MatrixBase<Derived>& m, Eigen::Index index)  // NOLINT(runtime/references)
    : m_ (std::addressof(m))
    , index_(index) {
  }

  bool operator==(const self_type& other) const {
    POLATORY_ASSERT(m_ == other.m_);
    return index_ == other.index_;
  }

  self_type& operator++() {
    POLATORY_ASSERT(index_ < m_->cols());
    index_++;
    return *this;
  }

  self_type& operator--() {
    POLATORY_ASSERT(index_ > 0);
    index_--;
    return *this;
  }

  typename self_type::reference operator*() const {
    return m_->col(index_);
  }

  typename self_type::pointer operator->() const = delete;

  bool operator<(const self_type& other) const {
    POLATORY_ASSERT(m_ == other.m_);
    return index_ < other.index_;
  }

  self_type& operator+=(typename self_type::difference_type n) {
    index_ += n;
    return *this;
  }

  self_type& operator-=(typename self_type::difference_type n) {
    index_ -= n;
    return *this;
  }

  friend typename self_type::difference_type
  operator-(const self_type& lhs, const self_type& rhs) {
    POLATORY_ASSERT(lhs.m_ == rhs.m_);
    return lhs.index_ - rhs.index_;
  }

private:
  Eigen::MatrixBase<Derived> *m_;
  Eigen::Index index_;
};

template <class Derived>
class const_col_iterator
  : public boost::random_access_iterator_helper<const_col_iterator<Derived>, typename Eigen::MatrixBase<Derived>::ConstColXpr, Eigen::Index, void, typename Eigen::MatrixBase<Derived>::ConstColXpr> {
  using self_type = const_col_iterator;

public:
  using iterator_category = std::input_iterator_tag;

  const_col_iterator(const Eigen::MatrixBase<Derived>& m, Eigen::Index index)
    : m_(std::addressof(m))
    , index_(index) {
  }

  bool operator==(const self_type& other) const {
    POLATORY_ASSERT(m_ == other.m_);
    return index_ == other.index_;
  }

  self_type& operator++() {
    POLATORY_ASSERT(index_ < m_->cols());
    index_++;
    return *this;
  }

  self_type& operator--() {
    POLATORY_ASSERT(index_ > 0);
    index_--;
    return *this;
  }

  typename self_type::reference operator*() const {  // NOLINT(readability-const-return-type)
    return m_->col(index_);
  }

  typename self_type::pointer operator->() const = delete;

  bool operator<(const self_type& other) const {
    POLATORY_ASSERT(m_ == other.m_);
    return index_ < other.index_;
  }

  self_type& operator+=(typename self_type::difference_type n) {
    index_ += n;
    return *this;
  }

  self_type& operator-=(typename self_type::difference_type n) {
    index_ -= n;
    return *this;
  }

  friend typename self_type::difference_type
  operator-(const self_type& lhs, const self_type& rhs) {
    POLATORY_ASSERT(lhs.m_ == rhs.m_);
    return lhs.index_ - rhs.index_;
  }

private:
  const Eigen::MatrixBase<Derived> *m_;
  Eigen::Index index_;
};

template <class Derived>
class row_iterator
  : public boost::random_access_iterator_helper<row_iterator<Derived>, typename Eigen::MatrixBase<Derived>::RowXpr, Eigen::Index, void, typename Eigen::MatrixBase<Derived>::RowXpr> {
  using self_type = row_iterator;

public:
  using iterator_category = std::input_iterator_tag;

  row_iterator(Eigen::MatrixBase<Derived>& m, Eigen::Index index)  // NOLINT(runtime/references)
    : m_ (std::addressof(m))
    , index_(index) {
  }

  bool operator==(const self_type& other) const {
    POLATORY_ASSERT(m_ == other.m_);
    return index_ == other.index_;
  }

  self_type& operator++() {
    POLATORY_ASSERT(index_ < m_->rows());
    index_++;
    return *this;
  }

  self_type& operator--() {
    POLATORY_ASSERT(index_ > 0);
    index_--;
    return *this;
  }

  typename self_type::reference operator*() const {
    return m_->row(index_);
  }

  typename self_type::pointer operator->() const = delete;

  bool operator<(const self_type& other) const {
    POLATORY_ASSERT(m_ == other.m_);
    return index_ < other.index_;
  }

  self_type& operator+=(typename self_type::difference_type n) {
    index_ += n;
    return *this;
  }

  self_type& operator-=(typename self_type::difference_type n) {
    index_ -= n;
    return *this;
  }

  friend typename self_type::difference_type
  operator-(const self_type& lhs, const self_type& rhs) {
    POLATORY_ASSERT(lhs.m_ == rhs.m_);
    return lhs.index_ - rhs.index_;
  }

private:
  Eigen::MatrixBase<Derived> *m_;
  Eigen::Index index_;
};

template <class Derived>
class const_row_iterator
  : public boost::random_access_iterator_helper<const_row_iterator<Derived>, typename Eigen::MatrixBase<Derived>::ConstRowXpr, Eigen::Index, void, typename Eigen::MatrixBase<Derived>::ConstRowXpr> {
  using self_type = const_row_iterator;

public:
  using iterator_category = std::input_iterator_tag;

  const_row_iterator(const Eigen::MatrixBase<Derived>& m, Eigen::Index index)
    : m_(std::addressof(m))
    , index_(index) {
  }

  bool operator==(const self_type& other) const {
    POLATORY_ASSERT(m_ == other.m_);
    return index_ == other.index_;
  }

  self_type& operator++() {
    POLATORY_ASSERT(index_ < m_->rows());
    index_++;
    return *this;
  }

  self_type& operator--() {
    POLATORY_ASSERT(index_ > 0);
    index_--;
    return *this;
  }

  typename self_type::reference operator*() const {  // NOLINT(readability-const-return-type)
    return m_->row(index_);
  }

  typename self_type::pointer operator->() const = delete;

  bool operator<(const self_type& other) const {
    POLATORY_ASSERT(m_ == other.m_);
    return index_ < other.index_;
  }

  self_type& operator+=(typename self_type::difference_type n) {
    index_ += n;
    return *this;
  }

  self_type& operator-=(typename self_type::difference_type n) {
    index_ -= n;
    return *this;
  }

  friend typename self_type::difference_type
  operator-(const self_type& lhs, const self_type& rhs) {
    POLATORY_ASSERT(lhs.m_ == rhs.m_);
    return lhs.index_ - rhs.index_;
  }

private:
  const Eigen::MatrixBase<Derived> *m_;
  Eigen::Index index_;
};

}  // namespace detail

template <class Derived>
auto col_begin(Eigen::MatrixBase<Derived>& m) {  // NOLINT(runtime/references)
  return detail::col_iterator<Derived>(m, 0);
}

template <class Derived>
auto col_end(Eigen::MatrixBase<Derived>& m) {  // NOLINT(runtime/references)
  return detail::col_iterator<Derived>(m, m.cols());
}

template <class Derived>
auto col_begin(const Eigen::MatrixBase<Derived>& m) {
  return detail::const_col_iterator<Derived>(m, 0);
}

template <class Derived>
auto col_end(const Eigen::MatrixBase<Derived>& m) {
  return detail::const_col_iterator<Derived>(m, m.cols());
}

template <class Derived>
auto row_begin(Eigen::MatrixBase<Derived>& m) {  // NOLINT(runtime/references)
  return detail::row_iterator<Derived>(m, 0);
}

template <class Derived>
auto row_end(Eigen::MatrixBase<Derived>& m) {  // NOLINT(runtime/references)
  return detail::row_iterator<Derived>(m, m.rows());
}

template <class Derived>
auto row_begin(const Eigen::MatrixBase<Derived>& m) {
  return detail::const_row_iterator<Derived>(m, 0);
}

template <class Derived>
auto row_end(const Eigen::MatrixBase<Derived>& m) {
  return detail::const_row_iterator<Derived>(m, m.rows());
}

namespace detail {

template <class Derived>
class col_range_wrapper {
public:
  explicit col_range_wrapper(Eigen::MatrixBase<Derived>& m)  // NOLINT(runtime/references)
    : m_(m) {
  }

  auto begin() { return col_begin(m_); }

  auto end() { return col_end(m_); }

private:
  Eigen::MatrixBase<Derived>& m_;
};

template <class Derived>
class const_col_range_wrapper {
public:
  explicit const_col_range_wrapper(const Eigen::MatrixBase<Derived>& m)
    : m_(m) {
  }

  auto begin() { return col_begin(m_); }

  auto end() { return col_end(m_); }

private:
  const Eigen::MatrixBase<Derived>& m_;
};

template <class Derived>
class row_range_wrapper {
public:
  explicit row_range_wrapper(Eigen::MatrixBase<Derived>& m)  // NOLINT(runtime/references)
    : m_(m) {
  }

  auto begin() { return row_begin(m_); }

  auto end() { return row_end(m_); }

private:
  Eigen::MatrixBase<Derived>& m_;
};

template <class Derived>
class const_row_range_wrapper {
public:
  explicit const_row_range_wrapper(const Eigen::MatrixBase<Derived>& m)
    : m_(m) {
  }

  auto begin() { return row_begin(m_); }

  auto end() { return row_end(m_); }

private:
  const Eigen::MatrixBase<Derived>& m_;
};

}  // namespace detail

template <class Derived>
auto col_range(Eigen::MatrixBase<Derived>& m) {  // NOLINT(runtime/references)
  return detail::col_range_wrapper<Derived>(m);
}

template <class Derived>
auto col_range(const Eigen::MatrixBase<Derived>& m) {
  return detail::const_col_range_wrapper<Derived>(m);
}

template <class Derived>
auto row_range(Eigen::MatrixBase<Derived>& m) {  // NOLINT(runtime/references)
  return detail::row_range_wrapper<Derived>(m);
}

template <class Derived>
auto row_range(const Eigen::MatrixBase<Derived>& m) {
  return detail::const_row_range_wrapper<Derived>(m);
}

namespace detail {

template <class Derived>
Eigen::Index common_cols(const Eigen::MatrixBase<Derived>& m) {
  return m.cols();
}

template <class Derived, class ...Args>
Eigen::Index common_cols(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
  if (m.cols() != common_cols(std::forward<Args>(args)...))
    throw invalid_argument("All matrices must have the same number of columns.");

  return m.cols();
}

template <class Derived>
Eigen::Index common_rows(const Eigen::MatrixBase<Derived>& m) {
  return m.rows();
}

template <class Derived, class ...Args>
Eigen::Index common_rows(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
  if (m.rows() != common_rows(std::forward<Args>(args)...))
    throw invalid_argument("All matrices must have the same number of rows.");

  return m.rows();
}

template <class ResultDerived, class Derived>
void concatenate_cols_impl(Eigen::MatrixBase<ResultDerived>& result,  // NOLINT(runtime/references)
                           const Eigen::MatrixBase<Derived>& m) {
  result = m;
}

template <class ResultDerived, class Derived, class ...Args>
void concatenate_cols_impl(Eigen::MatrixBase<ResultDerived>& result,  // NOLINT(runtime/references)
                           const Eigen::MatrixBase<Derived>& m, Args&&... args) {
  result.leftCols(m.cols()) = m;

  auto result_tail = result.rightCols(result.cols() - m.cols());
  concatenate_cols_impl(result_tail, std::forward<Args>(args)...);
}

template <class ResultDerived, class Derived>
void concatenate_rows_impl(Eigen::MatrixBase<ResultDerived>& result,  // NOLINT(runtime/references)
                           const Eigen::MatrixBase<Derived>& m) {
  result = m;
}

template <class ResultDerived, class Derived, class ...Args>
void concatenate_rows_impl(Eigen::MatrixBase<ResultDerived>& result,  // NOLINT(runtime/references)
                           const Eigen::MatrixBase<Derived>& m, Args&&... args) {
  result.topRows(m.rows()) = m;

  auto result_tail = result.bottomRows(result.rows() - m.rows());
  concatenate_rows_impl(result_tail, std::forward<Args>(args)...);
}

template <class ResultDerived, class Derived>
void take_cols_impl(Eigen::MatrixBase<ResultDerived>& result,  // NOLINT(runtime/references)
                    const Eigen::MatrixBase<Derived>& m, Eigen::Index index) {
  result.col(0) = m.col(index);
}

template <class ResultDerived, class Derived, class... Ts>
void take_cols_impl(Eigen::MatrixBase<ResultDerived>& result,  // NOLINT(runtime/references)
                    const Eigen::MatrixBase<Derived>& m, Eigen::Index index, Ts... indices) {
  result.col(0) = m.col(index);

  auto result_tail = result.rightCols(result.cols() - 1);
  take_cols_impl(result_tail, m, indices...);
}

template <class ResultDerived, class Derived>
void take_rows_impl(Eigen::MatrixBase<ResultDerived>& result,  // NOLINT(runtime/references)
                    const Eigen::MatrixBase<Derived>& m, Eigen::Index index) {
  result.row(0) = m.row(index);
}

template <class ResultDerived, class Derived, class... Ts>
void take_rows_impl(Eigen::MatrixBase<ResultDerived>& result,  // NOLINT(runtime/references)
                    const Eigen::MatrixBase<Derived>& m, Eigen::Index index, Ts... indices) {
  result.row(0) = m.row(index);

  auto result_tail = result.bottomRows(result.rows() - 1);
  take_rows_impl(result_tail, m, indices...);
}

}  // namespace detail

template <class ...Args>
auto concatenate_cols(Args&&... args) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
    detail::common_rows(std::forward<Args>(args)...),
    common::fold_left(std::plus<>(), args.cols()...)
  );

  detail::concatenate_cols_impl(result, std::forward<Args>(args)...);

  return result;
}

template <class ...Args>
auto concatenate_rows(Args&&... args) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
    common::fold_left(std::plus<>(), args.rows()...),
    detail::common_cols(std::forward<Args>(args)...)
  );

  detail::concatenate_rows_impl(result, std::forward<Args>(args)...);

  return result;
}

template <class Derived, class... Ts>
auto take_cols(const Eigen::MatrixBase<Derived>& m, Ts... indices) {
  Eigen::Matrix<
    typename Eigen::MatrixBase<Derived>::Scalar,
    Eigen::MatrixBase<Derived>::RowsAtCompileTime,
    sizeof...(indices),
    Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor
  > result(m.rows(), sizeof...(indices));

  detail::take_cols_impl(result, m, indices...);

  return result;
}

template <class Derived, class ForwardRange>
auto take_cols(const Eigen::MatrixBase<Derived>& m, ForwardRange indices) {
  Eigen::Index n_cols = std::distance(indices.begin(), indices.end());

  Eigen::Matrix<
    typename Eigen::MatrixBase<Derived>::Scalar,
    Eigen::MatrixBase<Derived>::RowsAtCompileTime,
    Eigen::Dynamic,
    Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor
  > result(m.rows(), n_cols);

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
  Eigen::Matrix<
    typename Eigen::MatrixBase<Derived>::Scalar,
    sizeof...(indices),
    Eigen::MatrixBase<Derived>::ColsAtCompileTime,
    Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor
  > result(sizeof...(indices), m.cols());

  detail::take_rows_impl(result, m, indices...);

  return result;
}

template <class Derived, class ForwardRange>
auto take_rows(const Eigen::MatrixBase<Derived>& m, ForwardRange indices) {
  Eigen::Index n_rows = std::distance(indices.begin(), indices.end());

  Eigen::Matrix<
    typename Eigen::MatrixBase<Derived>::Scalar,
    Eigen::Dynamic,
    Eigen::MatrixBase<Derived>::ColsAtCompileTime,
    Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor
  > result(n_rows, m.cols());

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

}  // namespace common
}  // namespace polatory
