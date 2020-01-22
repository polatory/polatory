// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/isosurface/rmt_primitive_lattice.hpp>

#include <exception>

namespace polatory {
namespace isosurface {

namespace detail {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4297)  // 'function' : function assumed not to throw an exception but does
#endif
lattice_vectors::lattice_vectors() noexcept try : base{ {
  rotation() * geometry::point3d({ 1.0, 1.0, -1.0 }).transpose() / std::sqrt(2.0),
  rotation() * geometry::point3d({ 1.0, -1.0, 1.0 }).transpose() / std::sqrt(2.0),
  rotation() * geometry::point3d({ -1.0, 1.0, 1.0 }).transpose() / std::sqrt(2.0)
} } {
} catch (const std::exception&) {
  std::terminate();
}

dual_lattice_vectors::dual_lattice_vectors() noexcept try : base{ {
  rotation() * geometry::point3d({ 1.0, 1.0, 0.0 }).transpose() / std::sqrt(2.0),
  rotation() * geometry::point3d({ 1.0, 0.0, 1.0 }).transpose() / std::sqrt(2.0),
  rotation() * geometry::point3d({ 0.0, 1.0, 1.0 }).transpose() / std::sqrt(2.0)
} } {
} catch (const std::exception&) {
  std::terminate();
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

}  // namespace detail

const detail::lattice_vectors LatticeVectors;

const detail::dual_lattice_vectors DualLatticeVectors;

}  // namespace isosurface
}  // namespace polatory
