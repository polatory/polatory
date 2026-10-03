// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/omp/direct.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_DIRECT_HPP
#define SCALFMM_ALGORITHMS_OMP_DIRECT_HPP

#ifdef _OPENMP

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/omp/macro.hpp"
#include "scalfmm/algorithms/omp/priorities.hpp"
#include "scalfmm/lists/utils.hpp"
#include "scalfmm/operators/mutual_apply.hpp"
#include "scalfmm/operators/p2p.hpp"
#include "scalfmm/operators/tags.hpp"

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace scalfmm::algorithms::omp::pass
{

    /**
    * @brief Compute direct interaction between particles when the source tree and the target tree are the same.
    *
    * Compute the near field interaction when source=target, the p2p interactions lists
    * should be construct before this call
    *
    * @tparam TreeType
    * @tparam NearFieldType
    *
    * @param tree  the octree
    * @param nearfield the nearfield operator
    */
    template<typename TreeType, typename NearFieldType>
    inline auto direct(TreeType const& tree, NearFieldType const& nearfield) -> void
    {
        using operators::p2p_full_mutual;
        using operators::p2p_inner;
        using operators::p2p_outer;

        // static constexpr int limit = (TreeType::dimension - 1) * 6 + (TreeType::dimension - 1);
        const auto separation_criterion = nearfield.separation_criterion();
        const auto mutual = nearfield.mutual();
        const auto period = tree.box().get_periodicity();
        const auto box_width = tree.box_width();

        // move test to execute
        // p2p stage
        auto begin = std::begin(tree);
        auto end = std::end(tree);

        const auto& matrix_kernel = nearfield.matrix_kernel();
        // loop on the groups
        // auto begin_groups{std::get<0>(begin)};
        // const auto end_groups{std::get<0>(end)};
        auto begin_groups{tree.begin_mine_leaves()};
        auto end_groups{tree.end_leaves()};
        // auto end_groups{tree.end_mine_leaves()};
        const auto prio_big{priorities::p2p_big};
        const auto prio_small{priorities::p2p_small};
        if(mutual)
        {
            using group_type = typename TreeType::group_of_leaf_type;
            struct out_of_group_task
            {
                group_type* group;
                group_type* other_group;
                std::size_t first_out_interaction;
                std::size_t last_out_interaction;
                std::size_t color;
            };
            std::vector<out_of_group_task> out_of_group_tasks;
            std::unordered_map<group_type*, std::vector<bool>> used_colors;
            for(auto group_it = begin_groups; group_it != end_groups; ++group_it)
            {
                group_type* group = group_it->get();
                std::size_t current_out_interaction{0};
                auto const& sym_g = group->csymbolics();
                for(auto& other_group_ptr_part: sym_g.group_dependencies)
                {
                    std::size_t first_out_interaction{0};
                    std::size_t last_out_interaction{0};
                    std::tie(first_out_interaction, last_out_interaction) = list::get_outside_interaction_range(
                      sym_g, other_group_ptr_part->csymbolics(), current_out_interaction);
                    current_out_interaction = last_out_interaction;

                    auto& colors_g = used_colors[group];
                    auto& colors_o = used_colors[other_group_ptr_part];
                    std::size_t color{0};
                    while((color < colors_g.size() && colors_g.at(color)) ||
                          (color < colors_o.size() && colors_o.at(color)))
                    {
                        ++color;
                    }
                    colors_g.resize(std::max(colors_g.size(), color + 1));
                    colors_o.resize(std::max(colors_o.size(), color + 1));
                    colors_g.at(color) = true;
                    colors_o.at(color) = true;
                    out_of_group_tasks.push_back(
                      {group, other_group_ptr_part, first_out_interaction, last_out_interaction, color});
                }
            }
            std::stable_sort(out_of_group_tasks.begin(), out_of_group_tasks.end(),
                             [](auto const& a, auto const& b) { return a.color < b.color; });

            for(auto const& task: out_of_group_tasks)
            {
                const auto group = task.group;
                const auto current_group_ptr_particles = group->depends_update();
                const auto other_group_ptr_particles = task.other_group->depends_update();
                const auto first_out_interaction = task.first_out_interaction;
                const auto last_out_interaction = task.last_out_interaction;
                // clang-format off
#pragma omp task untied default(none) shared(period, box_width, matrix_kernel) firstprivate(group, \
  current_group_ptr_particles, other_group_ptr_particles, first_out_interaction, last_out_interaction)    \
  depend(inout  : current_group_ptr_particles[0], other_group_ptr_particles[0]) priority(prio_small)
                // clang-format on
                {
                    operators::apply_out_of_group_p2p(*group, first_out_interaction, last_out_interaction,
                                                      matrix_kernel, period, box_width);
                }
            }
        }
        //
        // begin_groups = std::get<0>(begin);
        begin_groups = tree.begin_mine_leaves();
        end_groups = tree.end_mine_leaves();
        while(begin_groups != end_groups)
        {
            const auto current_group_ptr_particles = (*begin_groups).get()->depends_update();
            // clang does not manage to pass the separation_criterion correctly if it is not firstprivate
#pragma omp task untied default(none) shared(period, box_width, matrix_kernel)                                         \
  firstprivate(begin_groups, separation_criterion, mutual) depend(inout : current_group_ptr_particles[0])              \
  priority(prio_big)
            {
                // loop on the leaves of the current group
                for(std::size_t leaf_index = 0; leaf_index < (*begin_groups)->size(); ++leaf_index)
                {
                    auto& leaf = (*begin_groups)->component(leaf_index);
                    // get the group type to retrieve internal type definitions
                    // auto mine_index = leaf.index();

                    // Get interation infos
                    auto const& leaf_symbolics = leaf.csymbolics();
                    auto const& interaction_iterators = leaf_symbolics.interaction_iterators;
                    // auto const& interaction_indexes = leaf_symbolics.interaction_indexes;
                    if(separation_criterion == 1)
                    {
                        if(leaf_symbolics.number_of_neighbors > 0)
                        {
                            if(mutual)
                            {
                                // Optimization for mutual interation
                                // Mutual with neighbors + inner of current component
                                p2p_full_mutual(matrix_kernel, leaf, interaction_iterators,
                                                leaf_symbolics.existing_neighbors_in_group, period, box_width);
                            }
                            else
                            {
                                p2p_outer(matrix_kernel, leaf, interaction_iterators, period, box_width);
                            }
                        }
                    }
                    p2p_inner(matrix_kernel, leaf, mutual);
                }   // end of for_each on leaves
            }   // end task
            ++begin_groups;
        }
    }

    /**
     * @brief Compute direct interaction between particles
     *
     *  When source_tree = target_tree we call the direct(source_tree, nearfield) function
     *
     * @tparam SourceTreeType
     * @tparam TargetTreeType
     * @tparam NearFieldType
     *
     * @param source_tree the source tree
     * @param target_tree  tke target tree where we compute the field
     * @param nearfield  the near-field operator
     */
    template<typename SourceTreeType, typename TargetTreeType, typename NearFieldType>
    inline auto direct(SourceTreeType& source_tree, TargetTreeType& target_tree, NearFieldType const& nearfield) -> void
    {
        bool source_target{false};
        if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
        {
            source_target = (&source_tree == &target_tree);
        }
        if(source_target)
        {
            if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
            {
                direct(source_tree, nearfield);
            }
        }
        else
        {
            // source != target
            using operators::p2p_outer;

            const auto period = source_tree.box().get_periodicity();
            const auto box_width = source_tree.box_width();
            // move test to execute
            // p2p stage

            const auto& matrix_kernel = nearfield.matrix_kernel();

            static constexpr auto prio_big{priorities::p2p_big};
            // iterators on target leaves
            auto begin_groups{target_tree.begin_mine_leaves()};
            auto end_groups{target_tree.end_mine_leaves()};

            while(begin_groups != end_groups)
            {
                //   depends on ptr on first particles of the grp
                const auto current_group_ptr_particles = (*begin_groups).get()->depends_update();

#pragma omp task untied default(none) shared(period, box_width, matrix_kernel) firstprivate(begin_groups)              \
  depend(inout : current_group_ptr_particles[0]) priority(prio_big)
                {   // mutexinoutset
                    // loop on the leaves of the current group
                    for(std::size_t leaf_index = 0; leaf_index < (*begin_groups)->size(); ++leaf_index)
                    {
                        auto& leaf = (*begin_groups)->component(leaf_index);
                        // get the group type to retrieve internal type definitions

                        // Get interation infos
                        auto const& leaf_symbolics = leaf.csymbolics();
                        auto const& interaction_iterators = leaf_symbolics.interaction_iterators;
                        //
                        // No test on separation_criterion
                        if(leaf_symbolics.number_of_neighbors > 0)
                        {
                            p2p_outer(matrix_kernel, leaf, interaction_iterators, period, box_width);
                        }
                    }
                };
                ++begin_groups;
            }
        }   // end else
    }
}   // namespace scalfmm::algorithms::omp::pass

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_OMP_DIRECT_HPP
