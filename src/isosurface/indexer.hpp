#pragma once

#include <polatory/types.hpp>
#include <unordered_map>
#include <vector>

namespace polatory::isosurface {

// Maps values that are already a dense [0, n) range to themselves; compiles away.
class IdentityIndexer {
 public:
  using value_type = Index;

  explicit IdentityIndexer(Index n) : n_(n) {}

  Index size() const { return n_; }
  Index to_index(Index value) const { return value; }
  Index to_value(Index index) const { return index; }

 private:
  Index n_;
};

// Assigns a dense [0, n) index to each distinct value of an arbitrary type, and maps back.
template <class T>
class ValueIndexer {
 public:
  using value_type = T;

  ValueIndexer() = default;

  template <class Range>
  explicit ValueIndexer(const Range& values) {
    for (const auto& value : values) {
      insert(value);
    }
  }

  // The index of value, assigning the next free one if it is new.
  Index insert(const T& value) {
    auto [it, inserted] = to_index_.try_emplace(value, static_cast<Index>(values_.size()));
    if (inserted) {
      values_.push_back(value);
    }
    return it->second;
  }

  Index size() const { return static_cast<Index>(values_.size()); }
  Index to_index(const T& value) const { return to_index_.at(value); }
  const T& to_value(Index index) const { return values_.at(index); }

 private:
  std::unordered_map<T, Index> to_index_;
  std::vector<T> values_;
};

}  // namespace polatory::isosurface
