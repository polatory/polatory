/**
 * @file periodic.hpp
 * @author Olivier Coulaud (olivier.coulaud@inria.fr)
 * @brief  methods to compute periodic field at level 0
 * @version 0.1
 * @date 2022-02-07
 *
 * @copyright Copyright (c) 2022
 *  See LICENCE file at project root
 *
 */
// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/omp/periodic.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_PERIODIC_HPP
#define SCALFMM_ALGORITHMS_OMP_PERIODIC_HPP

#include "scalfmm/operators/l2l.hpp"
#include "scalfmm/operators/m2l.hpp"
#include "scalfmm/operators/m2m.hpp"
#include <scalfmm/algorithms/sequential/periodic.hpp>
#include <scalfmm/meta/utils.hpp>

#include <cpp_tools/colors/colorized.hpp>

#include <iostream>
#include <numeric>

namespace scalfmm::algorithms::omp::pass
{

    /**
    * @brief Updates the level (move to the fake level) for the FMM operators.
    *
    * @tparam[inout] TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto preprocessing_execute(TreeType& tree) -> void
    {
        auto offset_real_tree = 2 + tree.levels_above_root();
        if(offset_real_tree > 2)
        {
            const auto top_height = 0;

            auto begin = std::begin(tree);
            auto cell_level_it = std::get<1>(begin) + top_height;

            /////////////////////////////////////////////////////////////////////////////////////////
            ///  Loop on the level from the level 2 to the leaf level (Top to Bottom)
            for(std::size_t level = top_height; level < tree.height(); ++level)
            {
                for(std::size_t group_index{0}; group_index < std::size(*cell_level_it); ++group_index)
                {
                    auto& group = cell_level_it->at(group_index);
                    static constexpr int prio{priorities::max};
                    auto length = std::size(*group);
                    /////////////////////////////////////////////////////////////////////////////////////////
                    ///          loop on the leaves of the current group
#pragma omp task untied default(none) firstprivate(length, group_index) shared(offset_real_tree) priority(prio)
                    {
                        for(std::size_t index_in_group{0}; index_in_group < length; ++index_in_group)
                        {
                            auto& target_cell = group->component(index_in_group);
                            target_cell.symbolics().level = target_cell.symbolics().level + offset_real_tree;
                        }
                    }
                }
                ++cell_level_it;
            }
#pragma omp taskwait
        }
    }

    /**
    * @brief Removes the fake level (fake level) for the FMM operators.
    *
    * @tparam[inout] TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto postprocessing_execute(TreeType& tree) -> void
    {
        auto offset_real_tree = 2 + tree.levels_above_root();
        if(offset_real_tree > 2)
        {
            const auto top_height = 0;

            auto begin = std::begin(tree);
            auto cell_level_it = std::get<1>(begin) + top_height;

            /////////////////////////////////////////////////////////////////////////////////////////
            ///  Loop on the level from the level 2 to the leaf level (Top to Bottom)
            for(std::size_t level = top_height; level < tree.height(); ++level)
            {
                for(std::size_t group_index{0}; group_index < std::size(*cell_level_it); ++group_index)
                {
                    auto& group = cell_level_it->at(group_index);
                    static constexpr int prio{priorities::max};
                    auto length = std::size(*group);
                    /////////////////////////////////////////////////////////////////////////////////////////
                    ///          loop on the leaves of the current group
#pragma omp task untied default(none) firstprivate(length, group_index) shared(offset_real_tree) priority(prio)
                    {
                        for(std::size_t index_in_group{0}; index_in_group < length; ++index_in_group)
                        {
                            auto& target_cell = group->component(index_in_group);
                            target_cell.symbolics().level = target_cell.symbolics().level - offset_real_tree;
                        }
                    }
                }
                ++cell_level_it;
            }
#pragma omp taskwait
        }
    }

#ifdef SCALFMMM_UPDATE_LEVEL_OLD
    /**
    * @brief Updates the level (move to the fake level) for the FMM operators.
    *
    * @tparam[inout] TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    auto preprocessing_execute(TreeType& tree) -> void
    {
        auto offset_real_tree = 2 + tree.levels_above_root();
        if(offset_real_tree > 2)
        {
            // Add parallelism (loop on the group then cells)
            for(std::size_t level{0}; level < tree.height(); ++level)
            {
                for_each_cell(std::begin(tree), std::end(tree), level, [&offset_real_tree](auto& cell)
                              { cell.symbolics().level = cell.symbolics().level + offset_real_tree; });
            }
        }
    }

    /**
    * @brief Removes the fake level (fake level) for the FMM operators.
    *
    * @tparam[inout] TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    auto postprocessing_execute(TreeType& tree) -> void
    {
        auto offset_real_tree = 2 + tree.levels_above_root();
        if(offset_real_tree > 2)
        {
            for(std::size_t level{0}; level < tree.height(); ++level)
            {
                // Add parallelism (loop on the group then cells)

                for_each_cell(std::begin(tree), std::end(tree), level, [&offset_real_tree](auto& cell)
                              { cell.symbolics().level = cell.symbolics().level - offset_real_tree; });
            }
        }
    }
#endif

    /**
     * @brief Builds the field at level 0.
     *
     * @tparam Tree
     * @tparam InterpolatorType
     *
     * @param tree
     * @param approximation
     */
    template<typename TreeType, typename InterpolatorType>
    inline auto build_field_level0(TreeType& tree, InterpolatorType const& approximation) -> void
    {
        //
        // if kernel is absolutely convergent
        //    Perform MM2 / M2L / L2L on negative levels
        //      need height on negative level (algo in scalfmm2)
        // if kernel is conditionally convergent
        //       new algorithm to develop
        if(tree.levels_above_root() >= 1)
        {
            // dependencies
            //    in: on Multipole at root
            //    out: on local values for root

            auto group_level_it = std::get<1>(std::begin(tree));   // we are at the root.
            // The number of group
            // The unique group at level 0
            auto group_of_cell_begin = std::cbegin(*(group_level_it));
            // The unique cell at level 0
            auto cell_begin = std::begin(**group_of_cell_begin);
            auto& root_cell = *cell_begin;
            auto* ptr_dep_in = &(root_cell.multipoles(0));
            auto* ptr_dep_out = &(root_cell.locals(0));
#pragma omp task untied default(none) shared(tree, approximation) depend(in : ptr_dep_in[0])                           \
  depend(out : ptr_dep_out[0])
            {
                scalfmm::algorithms::sequential::pass::full_periodic_classical_summation(tree, approximation);
            }
        }
    }
}   // namespace scalfmm::algorithms::omp::pass
#endif
