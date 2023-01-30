#pragma once

#include <cstddef>

namespace polatory::common {

template <class RandomAccessIterator>
class iterator_range {
 public:
  iterator_range(RandomAccessIterator begin, RandomAccessIterator end) : begin_(begin), end_(end) {}

  auto begin() const { return begin_; }

  auto end() const { return end_; }

  auto size() const { return end_ - begin_; }

  auto operator[](std::size_t i) const { return begin_[i]; }

 private:
  RandomAccessIterator begin_;
  RandomAccessIterator end_;
};

template <class RandomAccessIterator>
iterator_range<RandomAccessIterator> make_range(RandomAccessIterator begin,
                                                RandomAccessIterator end) {
  return iterator_range<RandomAccessIterator>(begin, end);
}

}  // namespace polatory::common
