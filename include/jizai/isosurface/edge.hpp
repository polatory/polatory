#pragma once

#include <boost/container_hash/hash.hpp>
#include <compare>
#include <cstddef>
#include <jizai/types.hpp>

namespace jizai::isosurface {

struct Edge {
  Index a;
  Index b;

  Edge(Index u, Index w) : a(u < w ? u : w), b(u < w ? w : u) {}

  auto operator<=>(const Edge&) const = default;
};

struct EdgeHash {
  std::size_t operator()(const Edge& e) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, e.a);
    boost::hash_combine(seed, e.b);
    return seed;
  }
};

}  // namespace jizai::isosurface
