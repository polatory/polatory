#include <numbers>
#include <polatory/isosurface/rmt_primitive_lattice.hpp>

namespace polatory::isosurface {

namespace detail {

lattice_vectors::lattice_vectors()
    : base{{geometry::transform_vector<3>(rotation(), geometry::vector3d{-1.0, 1.0, 1.0}) /
                std::numbers::sqrt2,
            geometry::transform_vector<3>(rotation(), geometry::vector3d{1.0, -1.0, 1.0}) /
                std::numbers::sqrt2,
            geometry::transform_vector<3>(rotation(), geometry::vector3d{1.0, 1.0, -1.0}) /
                std::numbers::sqrt2}} {}

dual_lattice_vectors::dual_lattice_vectors()
    : base{{geometry::transform_vector<3>(rotation(), geometry::vector3d{0.0, 1.0, 1.0}) /
                std::numbers::sqrt2,
            geometry::transform_vector<3>(rotation(), geometry::vector3d{1.0, 0.0, 1.0}) /
                std::numbers::sqrt2,
            geometry::transform_vector<3>(rotation(), geometry::vector3d{1.0, 1.0, 0.0}) /
                std::numbers::sqrt2}} {}

}  // namespace detail

// NOLINTNEXTLINE(cert-err58-cpp)
const detail::lattice_vectors LatticeVectors;

// NOLINTNEXTLINE(cert-err58-cpp)
const detail::dual_lattice_vectors DualLatticeVectors;

}  // namespace polatory::isosurface
