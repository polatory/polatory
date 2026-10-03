// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/m2p.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_M2P_HPP
#define SCALFMM_OPERATORS_M2P_HPP

#include "scalfmm/operators/interpolation/m2p.hpp"

namespace scalfmm::operators
{
    /**
     * @brief
     *
     * @tparam FarFieldType
     * @tparam CellType
     * @tparam LeafType
     * @param far_field
     * @param source_cell
     * @param target_leaf
     */
    template<typename FarFieldType, typename CellType, typename LeafType>
    inline auto m2p(FarFieldType const& far_field, CellType const& source_cell, LeafType& target_leaf) -> void
    {
        apply_m2p(far_field, source_cell, target_leaf);
    }
}   // namespace scalfmm::operators
#endif   // SCALFMM_OPERATORS_M2P_HPP
