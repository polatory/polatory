// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/p2l.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_P2L_HPP
#define SCALFMM_OPERATORS_P2L_HPP

#include "scalfmm/operators/interpolation/p2l.hpp"

namespace scalfmm::operators
{
    /**
     * @brief
     *
     * @tparam FarFieldType
     * @tparam CellType
     * @tparam LeafType
     * @param far_field
     * @param target_cell
     * @param source_leaf
     */
    template<typename FarFieldType, typename CellType, typename LeafType>
    inline auto p2l(FarFieldType const& far_field, CellType& target_cell, LeafType const& source_leaf) -> void
    {
        apply_p2l(far_field, target_cell, source_leaf);
    }
}   // namespace scalfmm::operators
#endif   // SCALFMM_OPERATORS_P2L_HPP
