// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/omp/leaf_to_cell.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_LEAF_TO_CELL_HPP
#define SCALFMM_ALGORITHMS_OMP_LEAF_TO_CELL_HPP

#ifdef _OPENMP

#include "scalfmm/algorithms/omp/priorities.hpp"
#include "scalfmm/operators/p2m.hpp"
#include "scalfmm/utils/massert.hpp"

#include <omp.h>

namespace scalfmm::algorithms::omp::pass
{
    /**
    * @brief This function applies the p2m operator from the leaf level of the tree.
    *
    *  This function applies the p2m operator from the leaf level to
    *    the first cell level. The inputs of the particles contained in the leaves
    *     will be approximated by the technique used in the far_field
    *
    * @tparam TreeType
    * @tparam FarFieldType
    *
    * @param tree
    * @param far_field
    */
    template<typename TreeType, typename FarFieldType>
    inline auto leaf_to_cell(TreeType& tree, FarFieldType const& far_field) -> void
    {
        using operators::p2m;
        static constexpr auto prio{priorities::p2m};
        auto const& approximation = far_field.approximation();
        // the leaves
        auto group_of_leaf_begin = tree.begin_mine_leaves();
        auto group_of_leaf_end = tree.end_mine_leaves();
        // the cells
        auto leaf_level = tree.height() - 1;
        auto group_of_cell_begin = tree.begin_mine_cells(leaf_level);
        auto group_of_cell_end = tree.end_mine_cells(leaf_level);

        assertm(std::distance(group_of_leaf_begin, group_of_leaf_end) ==
                  std::distance(group_of_cell_begin, group_of_cell_end),
                "Bottom pass : nb group of leaves and nb first level of group cells differs !");

        // Loop over the group
        while(group_of_leaf_begin != group_of_leaf_end && group_of_cell_begin != group_of_cell_end)
        {
            // pointer on the first multipole of the  group
            auto group_cell_raw = &(*group_of_cell_begin)->ccomponent(0).cmultipoles(0);
            // clang-format off
#pragma omp task untied default(none) firstprivate(group_of_leaf_begin, group_of_cell_begin, group_cell_raw)  \
  shared(approximation, far_field, std::cout) depend(inout : group_cell_raw[0]) priority(prio)
            // clang-format on
            {
                // std::cout << "leaf_to_Cell(task) dependencies(out)  " << group_cell_raw << std::endl << std::flush;
                // Can be a task(in:iterParticles, out:iterCells)
                auto leaf_begin = (*group_of_leaf_begin)->cbegin();
                auto cell_begin = (*group_of_cell_begin)->begin();
                auto leaf_end = (*group_of_leaf_begin)->cend();
                auto cell_end = (*group_of_cell_begin)->end();

                // Leaf inside the group
                while(leaf_begin != leaf_end && cell_begin != cell_end)
                {
                    auto const& leaf = *leaf_begin;
                    auto& cell = *cell_begin;
                    assertm(leaf.csymbolics().morton_index == cell.symbolics().morton_index,
                            "Bottom pass : morton indexes of leaf and cell does not match.");
                    p2m(far_field, leaf, cell);
                    //
                    approximation.apply_multipoles_preprocessing(cell, omp_get_thread_num());
                    ++leaf_begin;
                    ++cell_begin;
                }
            }
            ++group_of_leaf_begin;
            ++group_of_cell_begin;
        }

        assertm(group_of_leaf_begin == group_of_leaf_end && group_of_cell_begin == group_of_cell_end,
                "Bottom pass : missing components at the end of the pass, iterators differs.");
    }
}   // namespace scalfmm::algorithms::omp::pass

#endif   // _OPENMP

#endif   // SCALFMM_ALGORITHMS_LEAF_TO_CELL_BOTTOM_HPP
