#pragma once

#include <boost/unordered/unordered_flat_map.hpp>
#include <numeric>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::isosurface {

// A disjoint-set (union-find) forest with path halving over indices [0, n).
class DisjointSets {
 public:
  explicit DisjointSets(Index n) : parent_(n) {
    std::iota(parent_.begin(), parent_.end(), Index{0});
  }

  // The members of each set.
  std::vector<std::vector<Index>> groups() {
    boost::unordered_flat_map<Index, Index> group_of;  // a set's root -> its position in result
    std::vector<std::vector<Index>> result;
    for (Index i = 0; i < static_cast<Index>(parent_.size()); i++) {
      auto [it, inserted] = group_of.try_emplace(find(i), static_cast<Index>(result.size()));
      if (inserted) {
        result.emplace_back();
      }
      result.at(it->second).push_back(i);
    }
    return result;
  }

  // Merges the sets containing i and j.
  void unite(Index i, Index j) { parent_.at(find(i)) = find(j); }

 private:
  // The representative of i's set, compressing i's path on the way.
  Index find(Index i) {
    while (parent_.at(i) != i) {
      parent_.at(i) = parent_.at(parent_.at(i));
      i = parent_.at(i);
    }
    return i;
  }

  std::vector<Index> parent_;
};

}  // namespace polatory::isosurface
