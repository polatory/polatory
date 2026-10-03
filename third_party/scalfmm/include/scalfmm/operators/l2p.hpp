// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/l2p.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_L2P_HPP
#define SCALFMM_OPERATORS_L2P_HPP

#include "scalfmm/operators/interpolation/l2p.hpp"

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
    inline void l2p(FarFieldType const& far_field, CellType const& source_cell, LeafType& target_leaf)
    {
        apply_l2p(far_field, source_cell, target_leaf);
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_L2P_HPP
