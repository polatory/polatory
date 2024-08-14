#pragma once

#include <algorithm>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

namespace polatory::common {

template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
void zip_sort(RandomAccessIterator1 first1, RandomAccessIterator1 last1,
              RandomAccessIterator2 first2, Compare comp) {
  using ValueType1 = typename std::iterator_traits<RandomAccessIterator1>::value_type;
  using ValueType2 = typename std::iterator_traits<RandomAccessIterator2>::value_type;

  std::vector<std::pair<ValueType1, ValueType2>> zipped;
  zipped.reserve(std::distance(first1, last1));

  for (auto [it1, it2] = std::make_tuple(first1, first2); it1 != last1; ++it1, ++it2) {
    zipped.emplace_back(std::move(*it1), std::move(*it2));
  }

  std::sort(zipped.begin(), zipped.end(), comp);

  for (auto [it1, it2, zip_it] = std::make_tuple(first1, first2, zipped.begin()); it1 != last1;
       ++it1, ++it2, ++zip_it) {
    *it1 = std::move(zip_it->first);
    *it2 = std::move(zip_it->second);
  }
}

}  // namespace polatory::common
