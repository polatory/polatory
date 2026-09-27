#pragma once

#include <boost/unordered/unordered_flat_map.hpp>
#include <numeric>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::isosurface {

class DisjointSets {
 public:
  explicit DisjointSets(Index n) : parent_(n) {
    std::iota(parent_.begin(), parent_.end(), Index{0});
  }

  std::vector<std::vector<Index>> groups() {
    boost::unordered_flat_map<Index, Index> root_to_group;
    std::vector<std::vector<Index>> result;
    for (Index i = 0; i < static_cast<Index>(parent_.size()); i++) {
      auto [it, inserted] = root_to_group.try_emplace(find(i), static_cast<Index>(result.size()));
      if (inserted) {
        result.emplace_back();
      }
      result.at(it->second).push_back(i);
    }
    return result;
  }

  void unite(Index i, Index j) { parent_.at(find(i)) = find(j); }

 private:
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
