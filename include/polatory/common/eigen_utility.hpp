// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include <boost/operators.hpp>
#include <Eigen/Core>

namespace polatory {
namespace common {

namespace detail {

template <class Derived>
class col_iterator
  : public boost::random_access_iterator_helper<col_iterator<Derived>, typename Eigen::MatrixBase<Derived>::ColXpr> {
  using self_type = col_iterator;

public:
  col_iterator(Eigen::MatrixBase<Derived>& m, size_t index)
    : m_ (m)
    , index_(index) {
  }

  bool operator==(const self_type& other) const {
    assert(std::addressof(m_) == std::addressof(other.m_));
    return index_ == other.index_;
  }

  self_type& operator++() {
    assert(index_ < m_.cols());
    index_++;
    return *this;
  }

  self_type& operator--() {
    assert(index_ > 0);
    index_--;
    return *this;
  }

  typename self_type::value_type operator*() {
    return m_.col(index_);
  }

  bool operator<(const self_type& other) const {
    assert(std::addressof(m_) == std::addressof(other.m_));
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
    assert(std::addressof(lhs.m_) == std::addressof(rhs.m_));
    return lhs.index_ - rhs.index_;
  }

private:
  Eigen::MatrixBase<Derived>& m_;
  size_t index_;
};

template <class Derived>
class const_col_iterator
  : public boost::random_access_iterator_helper<const_col_iterator<Derived>, typename Eigen::MatrixBase<Derived>::ConstColXpr> {
  using self_type = const_col_iterator;

public:
  const_col_iterator(const Eigen::MatrixBase<Derived>& m, size_t index)
    : m_(m)
    , index_(index) {
  }

  bool operator==(const self_type& other) const {
    assert(std::addressof(m_) == std::addressof(other.m_));
    return index_ == other.index_;
  }

  self_type& operator++() {
    assert(index_ < m_.cols());
    index_++;
    return *this;
  }

  self_type& operator--() {
    assert(index_ > 0);
    index_--;
    return *this;
  }

  typename self_type::value_type operator*() const {
    return m_.col(index_);
  }

  bool operator<(const self_type& other) const {
    assert(std::addressof(m_) == std::addressof(other.m_));
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
    assert(std::addressof(lhs.m_) == std::addressof(rhs.m_));
    return lhs.index_ - rhs.index_;
  }

private:
  const Eigen::MatrixBase<Derived>& m_;
  size_t index_;
};

template <class Derived>
class row_iterator
  : public boost::random_access_iterator_helper<row_iterator<Derived>, typename Eigen::MatrixBase<Derived>::RowXpr> {
  using self_type = row_iterator;

public:
  row_iterator(Eigen::MatrixBase<Derived>& m, size_t index)
    : m_ (m)
    , index_(index) {
  }

  bool operator==(const self_type& other) const {
    assert(std::addressof(m_) == std::addressof(other.m_));
    return index_ == other.index_;
  }

  self_type& operator++() {
    assert(index_ < m_.rows());
    index_++;
    return *this;
  }

  self_type& operator--() {
    assert(index_ > 0);
    index_--;
    return *this;
  }

  typename self_type::value_type operator*() {
    return m_.row(index_);
  }

  bool operator<(const self_type& other) const {
    assert(std::addressof(m_) == std::addressof(other.m_));
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
    assert(std::addressof(lhs.m_) == std::addressof(rhs.m_));
    return lhs.index_ - rhs.index_;
  }

private:
  Eigen::MatrixBase<Derived>& m_;
  size_t index_;
};

template <class Derived>
class const_row_iterator
  : public boost::random_access_iterator_helper<const_row_iterator<Derived>, typename Eigen::MatrixBase<Derived>::ConstRowXpr> {
  using self_type = const_row_iterator;

public:
  const_row_iterator(const Eigen::MatrixBase<Derived>& m, size_t index)
    : m_(m)
    , index_(index) {
  }

  bool operator==(const self_type& other) const {
    assert(std::addressof(m_) == std::addressof(other.m_));
    return index_ == other.index_;
  }

  self_type& operator++() {
    assert(index_ < m_.rows());
    index_++;
    return *this;
  }

  self_type& operator--() {
    assert(index_ > 0);
    index_--;
    return *this;
  }

  typename self_type::value_type operator*() const {
    return m_.row(index_);
  }

  bool operator<(const self_type& other) const {
    assert(std::addressof(m_) == std::addressof(other.m_));
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
    assert(std::addressof(lhs.m_) == std::addressof(rhs.m_));
    return lhs.index_ - rhs.index_;
  }

private:
  const Eigen::MatrixBase<Derived>& m_;
  size_t index_;
};

} // namespace detail

template <class Derived>
auto col_begin(Eigen::MatrixBase<Derived>& m) {
  return detail::col_iterator<Derived>(m, 0);
}

template <class Derived>
auto col_end(Eigen::MatrixBase<Derived>& m) {
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
auto row_begin(Eigen::MatrixBase<Derived>& m) {
  return detail::row_iterator<Derived>(m, 0);
}

template <class Derived>
auto row_end(Eigen::MatrixBase<Derived>& m) {
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
  col_range_wrapper(Eigen::MatrixBase <Derived>& m)
    : m_(m) {
  }

  auto begin() { return col_begin(m_); }

  auto end() { return col_end(m_); }

private:
  Eigen::MatrixBase <Derived>& m_;
};

template <class Derived>
class const_col_range_wrapper {
public:
  const_col_range_wrapper(const Eigen::MatrixBase <Derived>& m)
    : m_(m) {
  }

  auto begin() { return col_begin(m_); }

  auto end() { return col_end(m_); }

private:
  const Eigen::MatrixBase <Derived>& m_;
};

template <class Derived>
class row_range_wrapper {
public:
  row_range_wrapper(Eigen::MatrixBase <Derived>& m)
    : m_(m) {
  }

  auto begin() { return row_begin(m_); }

  auto end() { return row_end(m_); }

private:
  Eigen::MatrixBase <Derived>& m_;
};

template <class Derived>
class const_row_range_wrapper {
public:
  const_row_range_wrapper(const Eigen::MatrixBase <Derived>& m)
    : m_(m) {
  }

  auto begin() { return row_begin(m_); }

  auto end() { return row_end(m_); }

private:
  const Eigen::MatrixBase <Derived>& m_;
};

} // namespace detail

template <class Derived>
auto col_range(Eigen::MatrixBase<Derived>& m) {
  return detail::col_range_wrapper<Derived>(m);
}

template <class Derived>
auto col_range(const Eigen::MatrixBase<Derived>& m) {
  return detail::const_col_range_wrapper<Derived>(m);
}

template <class Derived>
auto row_range(Eigen::MatrixBase<Derived>& m) {
  return detail::row_range_wrapper<Derived>(m);
}

template <class Derived>
auto row_range(const Eigen::MatrixBase<Derived>& m) {
  return detail::const_row_range_wrapper<Derived>(m);
}

template <class Derived, class T, size_t N>
auto take_cols(const Eigen::MatrixBase <Derived>& m, const T (& indices)[N]) {
  Eigen::Matrix<
    typename Eigen::MatrixBase<Derived>::Scalar,
    Eigen::MatrixBase<Derived>::RowsAtCompileTime,
    N,
    Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor
  > result(m.rows(), N);

  for (size_t i = 0; i < N; i++) {
    result.col(i) = m.col(indices[i]);
  }

  return result;
}

template <class Derived>
auto take_cols(const Eigen::MatrixBase<Derived>& m, const std::vector<size_t>& indices) {
  Eigen::Matrix<
    typename Eigen::MatrixBase<Derived>::Scalar,
    Eigen::MatrixBase<Derived>::RowsAtCompileTime,
    Eigen::Dynamic,
    Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor
  > result(m.rows(), indices.size());

  for (size_t i = 0; i < indices.size(); i++) {
    result.col(i) = m.col(indices[i]);
  }

  return result;
}

template <class Derived, class T, size_t N>
auto take_rows(const Eigen::MatrixBase <Derived>& m, const T (& indices)[N]) {
  Eigen::Matrix<
    typename Eigen::MatrixBase<Derived>::Scalar,
    N,
    Eigen::MatrixBase<Derived>::ColsAtCompileTime,
    Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor
  > result(N, m.cols());

  for (size_t i = 0; i < N; i++) {
    result.row(i) = m.row(indices[i]);
  }

  return result;
}

template <class Derived>
auto take_rows(const Eigen::MatrixBase<Derived>& m, const std::vector<size_t>& indices) {
  Eigen::Matrix<
    typename Eigen::MatrixBase<Derived>::Scalar,
    Eigen::Dynamic,
    Eigen::MatrixBase<Derived>::ColsAtCompileTime,
    Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor
  > result(indices.size(), m.cols());

  for (size_t i = 0; i < indices.size(); i++) {
    result.row(i) = m.row(indices[i]);
  }

  return result;
}

} // namespace common
} // namespace polatory
