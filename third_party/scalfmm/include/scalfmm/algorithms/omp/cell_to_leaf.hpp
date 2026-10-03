// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/omp/cell_to_leaf.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_CELL_TO_LEAF_HPP
#define SCALFMM_ALGORITHMS_OMP_CELL_TO_LEAF_HPP

#ifdef _OPENMP

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/omp/priorities.hpp"
#include "scalfmm/operators/l2p.hpp"
#include "scalfmm/utils/massert.hpp"

namespace scalfmm::algorithms::omp::pass
{
    /**
    * @brief This function applies the l2p operator from the lower cell level to the leaf level.
    *
    * It applies the far field on the particles.
    * We pass here the entire far-field operator, this allows
    * us to catch combination of matrix kernels and trigger
    * optimizations.
    *
    * @tparam TreeType the tree type
    * @tparam FarFieldType the far-field operator type
    *
    * @param tree reference to the tree
    * @param FarFieldType const reference to the far-field operator of the fmm_operator
    */
    template<typename TreeType, typename FarFieldType>
    inline auto cell_to_leaf(TreeType& tree, FarFieldType const& far_field) -> void
    {
        using operators::l2p;
        static constexpr auto prio{priorities::l2p};

        // The leaves
        auto group_of_leaf_begin = tree.begin_mine_leaves();
        auto group_of_leaf_end = tree.end_mine_leaves();

        // the cells
        auto leaf_level = tree.height() - 1;
        auto group_of_cell_begin = tree.begin_mine_cells(leaf_level);
        auto group_of_cell_end = tree.end_mine_cells(leaf_level);

        assertm((*group_of_leaf_begin)->size() == (*group_of_cell_begin)->size(),
                "cell_to_leaf : nb group of leaves and nb first level of group cells differs !");

        while(group_of_leaf_begin != group_of_leaf_end && group_of_cell_begin != group_of_cell_end)
        {
            auto locals = group_of_cell_begin->get();
            auto locals_raw = &locals->ccomponent(0).clocals(0);
            const auto inout_particles = (*group_of_leaf_begin).get()->depends_update();
            //            depend(mutexinoutset : inout_particles[0]) depend(in : locals_raw[0])
            //	    std::cout << "cell_to_leaf dep  " << inout_particles << std::endl;
            // clang-format off
#pragma omp task untied default(none) \
            firstprivate(group_of_leaf_begin, group_of_cell_begin) shared(far_field) \
             depend(inout : inout_particles[0]) depend(in : locals_raw[0]) priority(prio)
            // clang-format on
            {   // Can be a task(in:iterCells, inout:iterParticles)
                for(std::size_t leaf_index = 0; leaf_index < (*group_of_leaf_begin)->size(); ++leaf_index)
                {
                    auto const& source_cell = (*group_of_cell_begin)->ccomponent(leaf_index);
                    auto& target_leaf = (*group_of_leaf_begin)->component(leaf_index);
                    l2p(far_field /*fmm_operator*/, source_cell, target_leaf);
                }
            }

            ++group_of_leaf_begin;
            ++group_of_cell_begin;
        }
    }
}   // namespace scalfmm::algorithms::omp::pass

#endif   // _OPENMP
#endif
