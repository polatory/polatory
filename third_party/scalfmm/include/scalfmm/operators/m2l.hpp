// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/m2l.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_M2L_HPP
#define SCALFMM_OPERATORS_M2L_HPP

#include <cstddef>

#include "scalfmm/operators/interpolation/m2l.hpp"

namespace scalfmm::operators
{
    /**
     * @brief Compute M2L operator between two cells (only used in periodic case)
     *
     * @tparam ApproximationType
     * @tparam CellType
     * @param approximation
     * @param source_cell
     * @param neighbor_idx
     * @param target_cell
     * @param current_tree_level
     * @param buffer
     */
    template<typename ApproximationType, typename CellType>
    inline auto m2l(ApproximationType const& approximation, CellType const& source_cell, std::size_t neighbor_idx,
                    CellType& target_cell, std::size_t current_tree_level,
                    [[maybe_unused]] typename ApproximationType::buffer_type& buffer) -> void
    {
        approximation.apply_m2l_single(source_cell, target_cell, neighbor_idx, current_tree_level, buffer);
    }

    /**
     * @brief
     *
     * @tparam ApproximationType
     * @tparam CellType
     * @param approximation
     * @param target_cell
     * @param current_tree_level
     * @param buffer
     */
    template<typename ApproximationType, typename CellType>
    inline auto m2l_loop(ApproximationType const& approximation, CellType& target_cell, std::size_t current_tree_level,
                         [[maybe_unused]] typename ApproximationType::buffer_type& buffer) -> void
    {
        approximation.apply_m2l_loop(target_cell, current_tree_level, buffer);
    }

    /**
     * @brief
     *
     * @tparam ApproximationType
     * @tparam CellType
     * @param approximation
     * @param source_cell
     * @param target_cell
     */
    template<typename ApproximationType, typename CellType>
    inline auto m2l_naive(ApproximationType const& approximation, CellType const& source_cell,
                          CellType& target_cell) -> void
    {
        apply_naive_m2l(approximation, source_cell, target_cell);
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_M2L_HPP
