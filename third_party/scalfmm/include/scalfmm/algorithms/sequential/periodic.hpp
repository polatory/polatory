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
// File : scalfmm/algorithms/sequential/periodic.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_PERIODIC_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_PERIODIC_HPP

#include <cpp_tools/colors/colorized.hpp>

#include "scalfmm/operators/l2l.hpp"
#include "scalfmm/operators/m2l.hpp"
#include "scalfmm/operators/m2m.hpp"
#include <scalfmm/meta/utils.hpp>
#include <scalfmm/tree/utils.hpp>

#include <iostream>
#include <numeric>

namespace scalfmm::algorithms::sequential::pass
{

    /**
    * @brief
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto preprocessing_execute(TreeType& tree) -> void
    {
        auto offset_real_tree = 2 + tree.levels_above_root();
        if(offset_real_tree > 2)
        {
            for(std::size_t level{0}; level < tree.height(); ++level)
            {
                for_each_cell(std::begin(tree), std::end(tree), level, [&offset_real_tree](auto& cell)
                              { cell.symbolics().level = cell.symbolics().level + offset_real_tree; });
            }
        }
    }

    /**
    * @brief
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto postprocessing_execute(TreeType& tree) -> void
    {
        auto offset_real_tree = 2 + tree.levels_above_root();
        if(offset_real_tree > 2)
        {
            for(std::size_t level{0}; level < tree.height(); ++level)
            {
                for_each_cell(std::begin(tree), std::end(tree), level, [&offset_real_tree](auto& cell)
                              { cell.symbolics().level = cell.symbolics().level - offset_real_tree; });
            }
        }
    }

    /**
     * @brief Performs the full summation in the periodic case.
     *
     *   The kernel is absolutely convergent, so we perform the FMM with negative level in order to
     *      increase the domain taken into account in the far field.
     *
     * @tparam TreeType
     * @tparam InterpolatorType
     *
     * @param tree
     * @param interpolator
     */
    template<typename TreeType, typename InterpolatorType>
    inline auto full_periodic_classical_summation(TreeType& tree, InterpolatorType const& interpolator) -> void
    {
        //
        // if kernel is absolutely convergent
        //    Perform MM2 / M2L / L2L on negative levels
        //      need height on negative level (algo in scalfmm2)

        using tree_type = TreeType;
        using interpolator_type = InterpolatorType;
        using cell_type = typename tree_type::cell_type;
        using value_type = typename tree_type::position_value_type;
        static constexpr std::size_t dimension = tree_type::dimension;

        auto levels_above_root = tree.levels_above_root();

        if(levels_above_root < 0)
        {
            return;
        }

        // all cells at a same level have the same multipole
        const auto order = tree.order();
        const auto center = tree.box().center();
        const auto nb_upper_cells = levels_above_root + 2;
        const auto width_ext = tree.box().extended_width(levels_above_root);

        // number of child (for the M2M pass)
        constexpr int nb_child{1 << dimension};

        int root_level = 0;

        auto group_level_it = std::get<1>(std::begin(tree)) + root_level;
        // The number of group
        // The unique group at level 0
        auto group_of_cell_begin = std::cbegin(*(group_level_it));
        // The unique cell at level 0
        auto cell_begin = std::begin(**group_of_cell_begin);
        auto nb_cells = (**group_of_cell_begin).size();
        if(nb_cells != 1)
        {
            std::cerr << " Bad news number of cells at root level should be 1 and not " << nb_cells << " !!!!\n ";
            std::exit(-1);
        }

        auto& root_cell = *cell_begin;
        //
        std::vector<cell_type> upper_cells;
        // fill the cells
        // upper level ;
        const int new_root_level = nb_upper_cells - 1;
        // Compute the global box size
        value_type width = tree.box().width(0);
        width *= value_type(4 << (levels_above_root));
        value_type extended_width{width_ext};

        for(int level{0}; level < new_root_level; ++level)
        {
            // width and center to change !
            extended_width /= 2;
            cell_type cell(center, extended_width, order);
            cell.symbolics().level = level + 1;
            upper_cells.push_back(cell);
        }
        {
            //  nb_upper_cells -> level 0
            //
            root_cell.symbolics().level = new_root_level + 1;
            upper_cells.push_back(root_cell);                   //
            upper_cells.back().set_width(extended_width / 2);   // width = the original box size
        }
        //  Start the different pass to compute the field on the root_cell
        // M2M pass
        {
            //  build multipole at level 1 ( level 1 is already built)
            // Build the 2^d child at level 1
            //
            using scalfmm::operators::m2m;
            for(int level{new_root_level}; level > 1; --level)
            {
                auto& parent_cell = upper_cells[level - 1];
                const auto& children_cell = upper_cells[level];
                for(int i = 0; i < nb_child; ++i)
                {
                    m2m(interpolator, children_cell, i, parent_cell);
                }
                interpolator.apply_multipoles_preprocessing(parent_cell);
            }
        }
        // M2L pass
        // the pass is decomposed in 2 steps 1)
        {
            using scalfmm::operators::m2l;
            static constexpr int nbGridSize = math::pow(7, dimension);
            std::array<cell_type*, nbGridSize> neighbors{};
            std::array<int, nbGridSize> neighborPositions{};
            //
            auto buffer(interpolator.buffer_initialization());
            //
            int existing_neighbors{0};
            auto separation_criterion{
              interpolation::interpolator_traits<interpolator_type>::matrix_kernel_type::separation_criterion};
            int idxUpperLevel{2};
            //
            // lambda function to set the position of the cell
            auto fill_neighbors = [&neighbors, &neighborPositions, &existing_neighbors, &upper_cells,
                                   &separation_criterion, &idxUpperLevel](auto... is)
            {
                if(((std::abs(is) > separation_criterion) || ...))
                {
                    std::array<int, dimension> a{is...};
                    neighbors[existing_neighbors] = &upper_cells[idxUpperLevel - 1];
                    neighborPositions[existing_neighbors] = scalfmm::index::neighbor_index(a);
                    ++existing_neighbors;
                }
            };
            // STEP 1
            ////////////////////////////////////////////////////////////////////////////////////
            std::array<int, dimension> start, end;
            // loop range [-3,3[^d
            start.fill(-3);
            end.fill(3);
            existing_neighbors = 0;
            // apply the lambda function on each element of the nested loop
            meta::looper_range<dimension>{}(fill_neighbors, start, end);
            //
            int cell_index{idxUpperLevel - 1};
            for(int index{0}; index < existing_neighbors; ++index)
            {
                m2l(interpolator, upper_cells[cell_index], neighborPositions[index], upper_cells[cell_index],
                    idxUpperLevel, buffer);
            };
            /// post-processing the leaf if necessary
            interpolator.apply_multipoles_postprocessing(upper_cells[cell_index], buffer);
            buffer = interpolator.buffer_initialization();
            // STEP 2
            ////////////////////////////////////////////////////////////////////////////////////
            // note: the triple loop bounds are not the same than for the previous piece of code
            // which handles the topmost virtual cell
            for(idxUpperLevel = 3; idxUpperLevel <= new_root_level + 1; ++idxUpperLevel)
            {
                existing_neighbors = 0;   // reset the number of neighbors
                cell_index = idxUpperLevel - 1;
                // loop range [-2,4[^d
                start.fill(-2);
                end.fill(4);
                meta::looper_range<dimension>{}(fill_neighbors, start, end);
                //
                for(int index{0}; index < existing_neighbors; ++index)
                {   // same source cell
                    m2l(interpolator, upper_cells[cell_index], neighborPositions[index], upper_cells[cell_index],
                        idxUpperLevel, buffer);
                };
                /// post-processing the local expansion if necessary
                interpolator.apply_multipoles_postprocessing(upper_cells[cell_index], buffer);
                buffer = interpolator.buffer_initialization();
            }
        }
        // L2L pass up to level 1
        {
            using scalfmm::operators::l2l;
            // the loop is on the child
            for(int level = 2; level < new_root_level; ++level)
            {
                const auto& parent_cell = upper_cells[level - 1];
                auto& children_cell = upper_cells[level];
                // Same cell so no loop on the child
                l2l(interpolator, parent_cell, 0, children_cell);
            }
            l2l(interpolator, upper_cells[new_root_level - 1], 7, upper_cells[new_root_level]);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////
        // set the contribution to the root level
        root_cell = upper_cells.back();
        root_cell.symbolics().level = root_level;
    }

    /**
     * @brief Builds the field at level 0.
     *
     * @tparam SourceTreeType
     * @tparam TargetTreeType
     * @tparam InterpolatorType
     *
     * @param source_tree
     * @param target_tree
     * @param approximation
     */
    template<typename SourceTreeType, typename TargetTreeType, typename InterpolatorType>
    inline auto build_field_level0(SourceTreeType& source_tree, TargetTreeType& target_tree,
                                   InterpolatorType const& approximation) -> void
    {
        //
        // if kernel is absolutely convergent
        //    Perform MM2 / M2L / L2L on negative levels
        //      need height on negative level (algo in scalfmm2)
        // if kernel is conditionally convergent
        //       new algorithm to develop
        bool same_tree{false};
        if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
        {
            same_tree = (&source_tree == &target_tree);
        }
        if(!same_tree)
        {
            throw std::runtime_error("In periodic algorithm works only when source==target.");
        }
        if(source_tree.levels_above_root() > 0)
        {
            full_periodic_classical_summation(source_tree, approximation);
        }
    }
}   // namespace scalfmm::algorithms::sequential::pass
#endif
