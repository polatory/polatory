#pragma once

#include <algorithm>
#include <boost/range/irange.hpp>
#include <jizai/types.hpp>
#include <vector>

namespace jizai::common {

inline std::vector<Index> complementary_indices(const std::vector<Index>& indices, Index n_points) {
  std::vector<Index> c_idcs(n_points - indices.size());

  auto universe = boost::irange<Index>(Index{0}, n_points);
  auto idcs = indices;
  std::ranges::sort(idcs);
  std::ranges::set_difference(universe, idcs, c_idcs.begin());

  return c_idcs;
}

}  // namespace jizai::common
