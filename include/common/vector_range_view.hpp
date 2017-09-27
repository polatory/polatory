// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

namespace polatory {
namespace common {

template <class T>
class vector_range_view {
public:
  using iterator = typename std::vector<T>::const_iterator;

  vector_range_view(const std::vector<T>& vector, size_t begin_index, size_t end_index)
    : v_(vector)
    , begin_idx_(begin_index)
    , end_idx_(end_index) {
  }

  iterator begin() const {
    return v_.begin() + begin_idx_;
  }

  bool empty() const {
    return begin_idx_ == end_idx_;
  }

  iterator end() const {
    return v_.begin() + end_idx_;
  }

  const T& operator[](size_t index) const {
    return v_[begin_idx_ + index];
  }

  size_t size() const {
    return end_idx_ - begin_idx_;
  }

private:
  const std::vector<T>& v_;
  const size_t begin_idx_;
  const size_t end_idx_;
};

template <class T>
auto make_range_view(const std::vector<T>& vector, size_t begin_index, size_t end_index) {
  return vector_range_view<T>(vector, begin_index, end_index);
}

} // namespace common
} // namespace polatory
