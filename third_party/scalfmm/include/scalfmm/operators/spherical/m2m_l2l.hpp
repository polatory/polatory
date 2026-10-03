// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/spherical/m2m_l2l.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_SPHERICAL_M2M_L2L_HPP
#define SCALFMM_OPERATORS_SPHERICAL_M2M_L2L_HPP

#include "scalfmm/spherical/sperical.hpp"

namespace scalfmm::operators
{
    // -------------------------------------------------------------
    // M2M operators
    // -------------------------------------------------------------

    /**
     * @brief
     *
     * @tparam T
     * @tparam CellType
     * @param far_field
     * @param child_cell
     * @param parent_cell
     */
    template<typename T, typename CellType>
    inline auto apply_m2m(spherical::spherical<T> const& far_field, CellType const& child_cell,
                          CellType& parent_cell) -> void
    {
        throw("Spherical M2M not implemented yet !");
    }

    // -------------------------------------------------------------
    // L2L operators
    // -------------------------------------------------------------

    /**
     * @brief
     *
     * @tparam T
     * @tparam CellType
     * @param far_field
     * @param parent_cell
     * @param child_cell
     */
    template<typename T, typename CellType>
    inline auto apply_l2l(spherical::spherical<T> const& far_field, CellType const& parent_cell,
                          CellType& child_cell) -> void
    {
        throw("Spherical L2L not implemented yet !");
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_SPHERICAL_M2M_L2L_HPP
