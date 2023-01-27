#include <exception>
#include <polatory/isosurface/rmt_primitive_lattice.hpp>

namespace polatory {
namespace isosurface {

namespace detail {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4297)  // 'function' : function assumed not to throw an exception but does
#endif
lattice_vectors::lattice_vectors() noexcept try : base{ {
  geometry::transform_vector(rotation(), { 1.0, 1.0, -1.0 }) / std::sqrt(2.0),
  geometry::transform_vector(rotation(), { 1.0, -1.0, 1.0 }) / std::sqrt(2.0),
  geometry::transform_vector(rotation(), { -1.0, 1.0, 1.0 }) / std::sqrt(2.0)
}  // namespace detail
}  // namespace isosurface
{}
catch (const std::exception&) {
  std::terminate();
}

dual_lattice_vectors::dual_lattice_vectors() noexcept try : base{ {
  geometry::transform_vector(rotation(), { 1.0, 1.0, 0.0 }) / std::sqrt(2.0),
  geometry::transform_vector(rotation(), { 1.0, 0.0, 1.0 }) / std::sqrt(2.0),
  geometry::transform_vector(rotation(), { 0.0, 1.0, 1.0 }) / std::sqrt(2.0)
}  // namespace polatory
}
{}
catch (const std::exception&) {
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
