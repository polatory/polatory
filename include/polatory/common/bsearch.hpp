// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <functional>
#include <iterator>

namespace polatory {
namespace common {

template <class BidirectionalIterator, class T, class Compare = std::less<T>>
auto bsearch_lt(BidirectionalIterator begin, BidirectionalIterator end,
                const T& val, Compare comp = Compare()) {
  auto it = std::lower_bound(begin, end, val, comp);
  return it == begin ? end : --it;
}

template <class BidirectionalIterator, class T, class Compare2 = std::less<T>>
auto bsearch_le(BidirectionalIterator begin, BidirectionalIterator end,
                const T& val, Compare2 comp = Compare2()) {
  auto it = std::upper_bound(begin, end, val, comp);
  return it == begin ? end : --it;
}

template <class BidirectionalIterator, class T, class Compare2 = std::less<T>>
auto bsearch_gt(BidirectionalIterator begin, BidirectionalIterator end,
                const T& val, Compare2 comp = Compare2()) {
  return std::upper_bound(begin, end, val, comp);
}

template <class BidirectionalIterator, class T, class Compare = std::less<T>>
auto bsearch_ge(BidirectionalIterator begin, BidirectionalIterator end,
                const T& val, Compare comp = Compare()) {
  return std::lower_bound(begin, end, val, comp);
}

template <class BidirectionalIterator, class T, class Compare = std::less<T>, class Compare2 = std::less<T>>
auto bsearch_eq(BidirectionalIterator begin, BidirectionalIterator end,
                const T& val, Compare comp = Compare(), Compare2 comp2 = Compare2()) {
  BidirectionalIterator it;
  typename std::iterator_traits<BidirectionalIterator>::difference_type count, step;
  count = std::distance(begin, end);

  while (count > 0) {
    it = begin;
    step = count / 2;
    std::advance(it, step);
    if (comp(*it, val)) {
      begin = ++it;
      count -= step + 1;
    } else if (comp2(val, *it)) {
      count = step;
    } else {
      return it;
    }
  }
  return end;
}

}  // namespace common
}  // namespace polatory
