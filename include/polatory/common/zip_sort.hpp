// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

// TODO(mizuno): Should be replaced by ranges.

namespace polatory {
namespace common {

namespace detail {

template <class RandomAccessIterator,
          class D = typename std::iterator_traits<RandomAccessIterator>::difference_type>
static void inverse_permute(RandomAccessIterator begin, RandomAccessIterator end, const std::vector<D>& p) {
  using std::swap;

  auto size = std::distance(begin, end);

  std::vector<bool> done(size);
  for (D i = 0; i < size; ++i) {
    if (done[i])
      continue;

    auto prev_j = i;
    auto j = p[i];
    while (i != j) {
      swap(begin[prev_j], begin[j]);
      done[j] = true;
      prev_j = j;
      j = p[j];
    }
    done[i] = true;
  }
}

}  // namespace detail

template <class RandomAccessIterator1, class RandomAccessIterator2, class Compare>
void zip_sort(RandomAccessIterator1 begin1, RandomAccessIterator1 end1,
              RandomAccessIterator2 begin2, RandomAccessIterator2 end2,
              Compare compare) {
  using D1 = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
  using D2 = typename std::iterator_traits<RandomAccessIterator2>::difference_type;

  static_assert(std::is_same<D1, D2>::value,
                "RandomAccessIterator1 and RandomAccessIterator2 must have the same difference_type.");

  assert(std::distance(begin1, end1) == std::distance(begin2, end2));

  auto size = std::distance(begin1, end1);

  std::vector<D1> permutation(size);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&](auto i, auto j) {
              return compare(std::make_pair(begin1[i], begin2[i]), std::make_pair(begin1[j], begin2[j]));
            });

  detail::inverse_permute(begin1, end1, permutation);
  detail::inverse_permute(begin2, end2, permutation);
}

}  // namespace common
}  // namespace polatory
