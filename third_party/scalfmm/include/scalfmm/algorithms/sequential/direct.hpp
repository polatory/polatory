// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/sequential/direct.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_DIRECT_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_DIRECT_HPP

#include "scalfmm/algorithms/common.hpp"
// #include "scalfmm/operators/l2p.hpp"
#include "scalfmm/operators/mutual_apply.hpp"
#include "scalfmm/operators/p2p.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace scalfmm::algorithms::sequential::pass
{
    /**
    * @brief Applies the near-field on the leaves of the tree.
    * The pass will update the leaves' interaction lists, retrieve the
    * neighbors for each leaf and apply the p2p operator on the list.
    *
    * @tparam SourceTreeType source tree type
    * @tparam TargetTreeType target tree type
    * @tparam NearFieldType near_field type
    *
    * @param source_tree a reference to the tree containing the sources
    * @param target_tree a reference to the tree containing the target
    * @param near_field a const reference of the near-field
    */
    template<typename SourceTreeType, typename TargetTreeType, typename NearFieldType>
    inline auto direct(SourceTreeType& source_tree, TargetTreeType& target_tree, NearFieldType const& near_field)
      -> void
    {
        bool source_target{false};
        if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
        {
            source_target = (&source_tree == &target_tree);
        }
        {
            using scalfmm::operators::p2p_full_mutual;
            using scalfmm::operators::p2p_inner;
            using scalfmm::operators::p2p_outer;

            const auto separation_criterion = near_field.separation_criterion();
            const auto mutual = near_field.mutual();   // get parameter to check if we are using mutual interactions
            const auto period = source_tree.box().get_periodicity();
            const auto box_width = source_tree.box_width();

            // move test to execute
            // p2p stage
            const auto level_leaves = target_tree.height() - 1;
            auto begin = std::begin(target_tree);   // iterators on target leaves (begin)
            auto end = std::end(target_tree);       // iterators on target leaves (end)

            std::size_t number_of_groups_processed{0};
            const auto& matrix_kernel = near_field.matrix_kernel();

            auto group_of_source_leaf_begin = std::get<0>(std::begin(source_tree));
            auto group_of_source_leaf_end = std::get<0>(std::end(source_tree));

            // loop on the groups
            auto l_group_p2p = [&matrix_kernel, begin, &level_leaves, group_of_source_leaf_begin,
                                group_of_source_leaf_end, &period, &box_width, &number_of_groups_processed,
                                separation_criterion, source_target, mutual](auto& group)
            {
                std::size_t index_in_group{0};

                // loop on the leaves of the current group
                auto l_leaf_p2p = [&matrix_kernel, &group, &index_in_group, &level_leaves, group_of_source_leaf_begin,
                                   group_of_source_leaf_end, &period, &box_width, separation_criterion, source_target,
                                   mutual](auto& leaf)
                {
                    // Get interation infos
                    auto const& leaf_symbolics = leaf.csymbolics();
                    auto const& interaction_iterators = leaf_symbolics.interaction_iterators;

                    // No test on separation_criterion
                    if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
                    {
                        if(source_target)   // sources == targets
                        {
                            p2p_inner(matrix_kernel, leaf, mutual);
                            if(leaf_symbolics.number_of_neighbors > 0)
                            {
                                if(mutual)
                                {
                                    // Optimization for mutual interactions
                                    // The p2p interaction list contains only indexes bellow mine.
                                    // Mutual with neighbors + inner of current component
                                    p2p_full_mutual(matrix_kernel, leaf, interaction_iterators,
                                                    leaf_symbolics.existing_neighbors_in_group, period, box_width);
                                }
                                else   // non mutual
                                {
                                    p2p_outer(matrix_kernel, leaf, interaction_iterators, period, box_width);
                                }
                            }
                        }
                        else   // source != tree
                        {
                            p2p_outer(matrix_kernel, leaf, interaction_iterators, period, box_width);
                        }
                    }
                    else
                    {
                        p2p_outer(matrix_kernel, leaf, interaction_iterators, period, box_width);
                    }
                };

                // Evaluate inside the group
                component::for_each(std::begin(*group), std::end(*group), l_leaf_p2p);

                // if  the sources and the target are identical and we are using the mutual
                //  algorithm then we have to proceed teh interactions with leaves inside groups
                // bellow mine
                if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
                {
                    if(mutual && source_target)
                    {
                        operators::apply_out_of_group_p2p(*group, matrix_kernel, period, box_width);
                    }
                }
            };

            // Evaluate  P2P for all group group
            component::for_each(std::get<0>(begin), std::get<0>(end), l_group_p2p);
        }
    }
}   // namespace scalfmm::algorithms::sequential::pass

#endif   // SCALFMM_ALGORITHMS_SEQUENTIAL_DIRECT_HPP
