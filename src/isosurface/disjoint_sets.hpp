#pragma once

#include <numeric>
#include <polatory/types.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace polatory::isosurface {

// A disjoint-set (union-find) forest with path halving.
template <class Indexer>
class DisjointSets {
  using Value = typename Indexer::value_type;

 public:
  explicit DisjointSets(Indexer indexer) : indexer_(std::move(indexer)), parent_(indexer_.size()) {
    std::iota(parent_.begin(), parent_.end(), Index{0});
  }

  // The members of each set.
  std::vector<std::vector<Value>> groups() {
    std::unordered_map<Index, Index> group_of;  // a set's root index -> its position in result
    std::vector<std::vector<Value>> result;
    for (Index i = 0; i < static_cast<Index>(parent_.size()); i++) {
      auto [it, inserted] = group_of.try_emplace(find(i), static_cast<Index>(result.size()));
      if (inserted) {
        result.emplace_back();
      }
      result.at(it->second).push_back(indexer_.to_value(i));
    }
    return result;
  }

  // Merges the sets containing values a and b.
  void unite(const Value& a, const Value& b) {
    parent_.at(find(indexer_.to_index(a))) = find(indexer_.to_index(b));
  }

 private:
  // The representative index of i's set, compressing i's path on the way.
  Index find(Index i) {
    while (parent_.at(i) != i) {
      parent_.at(i) = parent_.at(parent_.at(i));
      i = parent_.at(i);
    }
    return i;
  }

  Indexer indexer_;
  std::vector<Index> parent_;
};

}  // namespace polatory::isosurface
