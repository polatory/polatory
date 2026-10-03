// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/spherical/p2m.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_SPHERICAL_P2M_HPP
#define SCALFMM_OPERATORS_SPHERICAL_P2M_HPP

#include "scalfmm/spherical/sperical.hpp"

namespace scalfmm::operators
{
    // -------------------------------------------------------------
    // P2M operators
    // -------------------------------------------------------------

    /**
     * @brief
     *
     * @tparam T
     * @tparam LeafType
     * @tparam CellType
     * @param interp
     * @param source_leaf
     * @param cell
     */
    template<typename T, typename LeafType, typename CellType>
    inline auto apply_p2m(spherical::spherical<T> const& interp, LeafType const& source_leaf, CellType& cell) -> void
    {
        throw("Spherical P2M not implemented yet !");
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_SPHERICAL_P2M_HPP
