// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/m2m.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_M2M_HPP
#define SCALFMM_OPERATORS_M2M_HPP

#include "scalfmm/operators/interpolation/m2m_l2l.hpp"

namespace scalfmm::operators
{
    /**
     * @brief
     *
     * @tparam ApproximationType
     * @tparam CellType
     * @param approximation
     * @param child_cell
     * @param child_index
     * @param parent_cell
     * @param tree_level
     */
    template<typename ApproximationType, typename CellType>
    inline void m2m(ApproximationType const& approximation, CellType const& child_cell, std::size_t child_index,
                    CellType& parent_cell, std::size_t tree_level = 2)
    {
        apply_m2m(approximation, child_cell, child_index, parent_cell, tree_level);
    }
}   // namespace scalfmm::operators
#endif   // SCALFMM_OPERATORS_M2M_HPP
