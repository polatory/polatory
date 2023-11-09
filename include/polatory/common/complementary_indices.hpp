#pragma once

#include <algorithm>
#include <boost/range/irange.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::common {

inline std::vector<index_t> complementary_indices(const std::vector<index_t>& indices,
                                                  index_t n_points) {
  std::vector<index_t> c_idcs(n_points - indices.size());

  auto universe = boost::irange<index_t>(index_t{0}, n_points);
  auto idcs = indices;
  std::sort(idcs.begin(), idcs.end());
  std::set_difference(universe.begin(), universe.end(), idcs.begin(), idcs.end(), c_idcs.begin());

  return c_idcs;
}

}  // namespace polatory::common
