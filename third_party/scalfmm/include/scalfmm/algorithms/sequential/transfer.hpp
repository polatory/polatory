// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/sequential/transfer.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_TRANSFER_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_TRANSFER_HPP

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/m2l.hpp"
#include "scalfmm/operators/mutual_apply.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/header.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"

#include <algorithm>
#include <array>
#include <ios>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace scalfmm::algorithms::sequential::pass
{

    namespace impl
    {
        /**
	 * @brief This function transfer the multipoles to the local expansions by applying the matrix kernel.
	 * This is performed by the m2l operator (This is the optimized version).
	 *
	 * @tparam SourceTreeType
	 * @tparam TargetTreeType
	 * @tparam FarFieldType
	 *
	 * @param source_tree
	 * @param target_tree
	 * @param far_field
	 */
        template<typename SourceTreeType, typename TargetTreeType, typename FarFieldType>
        inline auto transfer(SourceTreeType& source_tree, TargetTreeType& target_tree, FarFieldType const& far_field)
          -> void
        {
            bool source_eq_target{false};
            if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
            {
                source_eq_target = (&source_tree == &target_tree);
            }

            using operators::m2l_loop;   // Useful to overload m2l(...) function

            auto tree_height = target_tree.height();
            // Should be set by the kernel !
            auto const& approximation = far_field.approximation();

            // Buffer for m2l optimization : we calculate the inverse fft on the fly and free memory.
            // get the shape of the buffer used in the m2l pass (get it from the far field)
            // and allocate it
            auto buffer(approximation.buffer_initialization());

            /////////////////////////////////////////////////////////////////////////////////////////
            ///  Loop on the level from the level 1/2 to the leaf level (Top to Bottom)
            //
            const auto top_height = target_tree.box().is_periodic() ? 1 : 2;

            // Iterate on the target tree
            for(std::size_t level = top_height; level < tree_height; ++level)
            {
                auto begin_cell_target_level_it = target_tree.begin_cells(level);
                auto end_cell_target_level_it = target_tree.end_cells(level);

                /////////////////////////////////////////////////////////////////////////////////////////
                ///            loop on the target groups at level level
                for(auto cell_target_level_it = begin_cell_target_level_it;
                    cell_target_level_it != end_cell_target_level_it; ++cell_target_level_it)
                {
                    auto group_target = cell_target_level_it->get();

                    /////////////////////////////////////////////////////////////////////////////////////////
                    ///          loop on the target leaves of the current group
                    for(std::size_t index_in_group{0};
                        index_in_group < group_target->size() /*std::size(*group_target)*/; ++index_in_group)
                    {
                        auto& target_cell = group_target->component(index_in_group);
                        auto const& cell_symbolics = target_cell.csymbolics();

                        m2l_loop(approximation, target_cell, cell_symbolics.level, buffer);

                        /// post-processing the leaf if necessary
                        approximation.apply_multipoles_postprocessing(target_cell, buffer);

                        approximation.buffer_reset(buffer);
                    }
                    /// post-processing the group if necessary
                }
            }
        }

        /**
	 * @brief This function transfer the multipoles to the local expansions by applying the matrix kernel.
	 * This is performed by the m2l operator (This is the naive version that computes the M2L pass on the
	 * fly without any optimization).
	 *
	 * @tparam SourceTreeType
	 * @tparam TargetTreeType
	 * @tparam FarFieldType
	 *
	 * @param source_tree
	 * @param target_tree
	 * @param far_field
	 */
        template<typename SourceTreeType, typename TargetTreeType, typename FarFieldType>
        inline auto transfer_naive(SourceTreeType& source_tree, TargetTreeType& target_tree,
                                   FarFieldType const& far_field) -> void
        {
            bool source_eq_target{false};
            if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
            {
                source_eq_target = (&source_tree == &target_tree);
            }

            using operators::m2l_naive;

            auto tree_height = target_tree.height();
            // Should be set by the kernel !
            auto const& approximation = far_field.approximation();

            // Buffer for m2l optimization : we calculate the inverse fft on the fly and free memory.
            // get the shape of the buffer used in the m2l pass (get it from the far field)
            // and allocate it
            // auto buffer(approximation.buffer_initialization());

            /////////////////////////////////////////////////////////////////////////////////////////
            ///  Loop on the level from the level 1/2 to the leaf level (Top to Bottom)
            //
            const auto top_height = target_tree.box().is_periodic() ? 1 : 2;

            // Iterate on the target tree
            for(std::size_t level = top_height; level < tree_height; ++level)
            {
                auto begin_cell_target_level_it = target_tree.begin_cells(level);
                auto end_cell_target_level_it = target_tree.end_cells(level);

                /////////////////////////////////////////////////////////////////////////////////////////
                ///            loop on the target groups at level level
                for(auto cell_target_level_it = begin_cell_target_level_it;
                    cell_target_level_it != end_cell_target_level_it; ++cell_target_level_it)
                {
                    auto group_target = cell_target_level_it->get();

                    /////////////////////////////////////////////////////////////////////////////////////////
                    ///          loop on the target leaves of the current group
                    for(std::size_t index_in_group{0};
                        index_in_group < group_target->size() /*std::size(*group_target)*/; ++index_in_group)
                    {
                        auto& target_cell = group_target->component(index_in_group);
                        auto const& cell_symbolics = target_cell.csymbolics();
                        auto const& interaction_iterators = cell_symbolics.interaction_iterators;

                        // loop on the neighboring cells
                        for(std::size_t index{0}; index < cell_symbolics.existing_neighbors; ++index)
                        {
                            auto const& source_cell = *interaction_iterators.at(index);
                            m2l_naive(approximation, source_cell, target_cell);
                        }

                        /// post-processing the leaf if necessary
                        // approximation.apply_multipoles_postprocessing(target_cell, buffer);

                        // approximation.buffer_reset(buffer);
                    }
                }
            }
        }
    }   // namespace impl

    /**
     * @brief This function transfer the multipoles to the local expansions by applying the matrix kernel.
     * This is performed by the m2l operator (Interface that picks either the optimized or the naive version)
     *
     * @tparam SourceTreeType
     * @tparam TargetTreeType
     * @tparam FarFieldType
     *
     * @param source_tree
     * @param target_tree
     * @param far_field
     */
    template<typename SourceTreeType, typename TargetTreeType, typename FarFieldType>
    inline auto transfer(SourceTreeType& source_tree, TargetTreeType& target_tree, FarFieldType const& far_field)
      -> void
    {
#if defined(scalfmm_DEBUG)
        impl::transfer_naive(source_tree, target_tree, far_field);
#else
        impl::transfer(source_tree, target_tree, far_field);
#endif
    }

}   // namespace scalfmm::algorithms::sequential::pass
#endif   // SCALFMM_ALGORITHMS_SEQUENTIAL_TRANSFER_HPP
