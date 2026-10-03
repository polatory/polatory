// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/spherical/l2p.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_SPHERICAL_L2P_HPP
#define SCALFMM_OPERATORS_SPHERICAL_L2P_HPP

#include "scalfmm/spherical/sperical.hpp"

namespace scalfmm::operators
{
    // -------------------------------------------------------------
    // L2P operators
    // -------------------------------------------------------------

    /**
     * @brief
     *
     * @tparam T
     * @tparam CellType
     * @tparam LeafType
     * @param far_field
     * @param source_cell
     * @param target_leaf
     */
    template<typename T, typename CellType, typename LeafType>
    inline auto apply_l2p(spherical::spherical<T> const& far_field, CellType const& source_cell,
                          LeafType& target_leaf) -> void
    {
        throw("Spherical L2P not implemented yet !");
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_INTERPOLATION_L2P_HPP
