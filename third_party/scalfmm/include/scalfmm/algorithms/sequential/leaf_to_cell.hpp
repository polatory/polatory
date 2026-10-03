// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/sequential/leaf_to_cell.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_LEAF_TO_CELL_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_LEAF_TO_CELL_HPP

#include "scalfmm/operators/p2m.hpp"
#include "scalfmm/utils/massert.hpp"

#include <iterator>
#include <tuple>
#include <utility>

namespace scalfmm::algorithms::sequential::pass
{
    /**
    * @brief This function applies the p2m operator from the leaf level of the tree.
    *
    *  This function applies the p2m operator from the leaf level to
    *    the first cell level. The inputs of the particles contained in the leaves
    *     will be approximated by the technique used in the far_field
    *
    * @tparam TreeType the tree type
    * @tparam FarFieldType the far-field type
    *
    * @param tree reference to the tree source
    * @param far_field reference to the far field.
    */
    template<typename TreeType, typename FarFieldType>
    inline auto leaf_to_cell(TreeType& tree, FarFieldType const& far_field) -> void
    {
        using operators::p2m;

        auto begin = std::begin(tree);
        auto end = std::end(tree);
        auto leaf_level = tree.height() - 1;

        auto group_of_leaf_begin = std::get<0>(begin);
        auto group_of_leaf_end = std::get<0>(end);

        auto& cells_at_leaf_level = *(std::get<1>(begin) + leaf_level);
        auto group_of_cell_begin = std::begin(cells_at_leaf_level);
        auto group_of_cell_end = std::end(cells_at_leaf_level);
        //
        auto const& approximation = far_field.approximation();
        // Loop over the group
        while(group_of_leaf_begin != group_of_leaf_end && group_of_cell_begin != group_of_cell_end)
        {
            // Can be a task(in:iterParticles, out:iterCells)
            [[maybe_unused]] auto const& leaf_group_symbolics = (*group_of_leaf_begin)->csymbolics();
            [[maybe_unused]] auto const& cell_group_symbolics = (*group_of_cell_begin)->csymbolics();
            auto leaf_begin = (*group_of_leaf_begin)->cbegin();
            auto cell_begin = (*group_of_cell_begin)->begin();
            auto leaf_end = (*group_of_leaf_begin)->cend();
            auto cell_end = (*group_of_cell_begin)->end();

            assertm(leaf_group_symbolics.number_of_component_in_group ==
                      cell_group_symbolics.number_of_component_in_group,
                    "Bottom pass : number of components in leaf and cell groups are not the same.");
            //
            // Leaf inside the group
            while(leaf_begin != leaf_end && cell_begin != cell_end)
            {
                auto const& leaf = *leaf_begin;
                auto& cell = *cell_begin;
                assertm(leaf.csymbolics().morton_index == cell.symbolics().morton_index,
                        "Bottom pass : morton indexes of leaf and cell does not match.");

                p2m(far_field, leaf, cell);
                //
                approximation.apply_multipoles_preprocessing(cell);
                ++leaf_begin;
                ++cell_begin;
            }

            ++group_of_leaf_begin;
            ++group_of_cell_begin;
        }

        assertm(group_of_leaf_begin == group_of_leaf_end && group_of_cell_begin == group_of_cell_end,
                "Bottom pass : missing components at the end of the pass, iterators differs.");
    }
}   // namespace scalfmm::algorithms::sequential::pass

#endif   // SCALFMM_ALGORITHMS_SEQUENTIAL_LEAF_TO_CELL_HPP
