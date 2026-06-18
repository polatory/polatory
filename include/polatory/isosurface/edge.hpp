#pragma once

#include <boost/container_hash/hash.hpp>
#include <compare>
#include <cstddef>
#include <polatory/types.hpp>

namespace polatory::isosurface {

// An undirected edge between two vertices. The constructor orders the endpoints, so Edge{u, w}
// and Edge{w, u} are the same edge and a direction-bearing edge cannot be represented. Build
// it braced: {u, w} as an argument or key, Edge{u, w} where a type is needed.
struct Edge {
  Index a;
  Index b;

  Edge(Index u, Index w) : a(u < w ? u : w), b(u < w ? w : u) {}

  auto operator<=>(const Edge&) const = default;
};

// Hashes an Edge, matching boost::hash<std::pair<Index, Index>> so a map keyed on Edge keeps
// the layout it had with a pair key.
struct EdgeHash {
  std::size_t operator()(const Edge& e) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, e.a);
    boost::hash_combine(seed, e.b);
    return seed;
  }
};

}  // namespace polatory::isosurface
