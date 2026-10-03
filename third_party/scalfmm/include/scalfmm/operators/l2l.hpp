// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/l2l.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_L2L_HPP
#define SCALFMM_OPERATORS_L2L_HPP

#include "scalfmm/operators/interpolation/m2m_l2l.hpp"

namespace scalfmm::operators
{
    /**
     * @brief
     *
     * @tparam ApproximationType
     * @tparam CellType
     * @param approximation
     * @param parent_cell
     * @param child_index
     * @param child_cell
     * @param tree_level
     */
    template<typename ApproximationType, typename CellType>
    inline void l2l(ApproximationType const& approximation, CellType const& parent_cell, std::size_t child_index,
                    CellType& child_cell, std::size_t tree_level = 2)
    {
        apply_l2l(approximation, parent_cell, child_index, child_cell, tree_level);
    }
}   // namespace scalfmm::operators
#endif   // SCALFMM_OPERATORS_M2M_HPP
