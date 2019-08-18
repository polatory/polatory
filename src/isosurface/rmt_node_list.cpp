// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/isosurface/rmt_node_list.hpp>

#include <exception>

namespace polatory {
namespace isosurface {

namespace detail {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4297)  // 'function' : function assumed not to throw an exception but does
#endif
neighbor_cell_vectors::neighbor_cell_vectors() noexcept try : base{ {
  cell_vector(+1, +1, +1),
  cell_vector(+1, +1, +0),
  cell_vector(+0, +0, -1),
  cell_vector(+1, +0, +1),
  cell_vector(+1, +0, +0),
  cell_vector(+0, -1, -1),
  cell_vector(+0, -1, +0),
  cell_vector(-1, -1, -1),
  cell_vector(-1, -1, +0),
  cell_vector(+0, +0, +1),
  cell_vector(-1, +0, -1),
  cell_vector(-1, +0, +0),
  cell_vector(+0, +1, +1),
  cell_vector(+0, +1, +0)
} } {
} catch (const std::exception&) {
  std::terminate();
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

}  // namespace detail

const detail::neighbor_cell_vectors NeighborCellVectors;

}  // namespace isosurface
}  // namespace polatory
