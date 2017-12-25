// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/isosurface/rmt_surface.hpp>

namespace polatory {
namespace isosurface {

namespace detail {

constexpr std::array<std::array<edge_index, 3>, 6> rmt_tetrahedron::EdgeIndices;

constexpr std::array<std::array<edge_index, 3>, 6> rmt_tetrahedron::OuterEdgeIndices;

}  // namespace detail

}  // namespace isosurface
}  // namespace polatory
