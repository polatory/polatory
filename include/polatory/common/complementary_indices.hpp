#pragma once

#include <algorithm>
#include <boost/range/irange.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::common {

inline std::vector<Index> complementary_indices(const std::vector<Index>& indices, Index n_points) {
  std::vector<Index> c_idcs(n_points - indices.size());

  auto universe = boost::irange<Index>(Index{0}, n_points);
  auto idcs = indices;
  std::sort(idcs.begin(), idcs.end());
  std::set_difference(universe.begin(), universe.end(), idcs.begin(), idcs.end(), c_idcs.begin());

  return c_idcs;
}

}  // namespace polatory::common
