// --------------------------------
// See LICENCE file at project root
// File : scalfmm/lists/omp.hpp
// --------------------------------
#ifndef SCALFMM_LISTS_OMP_HPP
#define SCALFMM_LISTS_OMP_HPP

#define SCALFMM_LISTS_OMP

#include "scalfmm/algorithms/omp/priorities.hpp"
#include "scalfmm/lists/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"

namespace scalfmm::list::omp
{
    /**
     * @brief Construct the P2P  interaction list for the target tree
     *
     * @tparam SourceTreeType
     * @tparam TargetTreeType
     * @param[in] source_tree the tree containing the sources
     * @param[inout] target_tree the tree containing the targets.
     * @param[in] neighbour_separation separation criterion use to separate teh near and the far field
     * @param[in] mutual boolean to specify if the direct pass use a symmetric algorithm (mutual interactions)
     */
    template<typename SourceTreeType, typename TargetTreeType>
    inline auto build_p2p_interaction_list(SourceTreeType const& source_tree, TargetTreeType& target_tree,
                                           const int& neighbour_separation, const bool mutual) -> void
    {
        // We iterate on the leaves
        // true if source == target
        bool source_target{false};
        if constexpr(std::is_same_v<std::decay_t<SourceTreeType>, std::decay_t<TargetTreeType>>)
        {
            source_target = (&source_tree == &target_tree);
        }
        if((!source_target) && mutual)
        {
            throw std::invalid_argument(
              " Mutual set to true is prohibited when the sources are different from the targets.\n");
        }
        // std::cout << std::boolalpha << "source == target " << source_target << std::endl;
        const auto& period = target_tree.box().get_periodicity();
        const auto leaf_level = target_tree.leaf_level();
        auto begin_of_source_groups = std::get<0>(source_tree.begin());
        auto end_of_source_groups = std::get<0>(source_tree.end());

        // Iterate on the group of leaves I own
        component::for_each(
          target_tree.begin_mine_leaves(), target_tree.end_mine_leaves(),
          [&period, &neighbour_separation, &begin_of_source_groups, &end_of_source_groups, &leaf_level, mutual,
           source_target](auto& group_target)
          {
              auto group_target_ptr = group_target;
              static constexpr int prio{scalfmm::algorithms::omp::priorities::max};
#pragma omp task untied default(none) firstprivate(begin_of_source_groups, end_of_source_groups, group_target_ptr)     \
  shared(leaf_level, mutual, source_target, period, neighbour_separation) priority(prio)
              {
                  std::size_t index_in_group{0};

                  // Iterate on  leaves inside the group
                  component::for_each(
                    std::begin(*group_target_ptr), std::end(*group_target_ptr),
                    [&group_target_ptr, &index_in_group, &begin_of_source_groups, &end_of_source_groups, &period,
                     &neighbour_separation, &leaf_level, mutual, source_target](auto& leaf_target)
                    {
                        scalfmm::list::build_p2p_interaction_list_inside_group(
                          leaf_target, begin_of_source_groups, end_of_source_groups, *group_target_ptr, index_in_group,
                          leaf_level, mutual, period, neighbour_separation, source_target);

                        ++index_in_group;
                    });
                  //
                  if(mutual)
                  {
                      // here we are in the case that source == target and mutual interactions (optimization)
                      scalfmm::list::build_out_of_group_interactions(begin_of_source_groups, *group_target_ptr);
                  }
              }
          });
#pragma omp taskwait
        target_tree.is_interaction_p2p_lists_built() = true;
    }

    /**
     * @brief Construct the M2L interaction list for the target tree
     *
     * @tparam SourceTreeType
     * @tparam TargetTreeType
     * @param[in] source_tree the tree containing the sources
     * @param[inout] target_tree the tree containing the targets.
     * @param[in] neighbour_separation separation criterion use to separate teh near and the far field
     */
    template<typename SourceTreeType, typename TargetTreeType>
    inline auto build_m2l_interaction_list(SourceTreeType& source_tree, TargetTreeType& target_tree,
                                           const int& neighbour_separation, const int leaf_neighbour_separation) -> void
    {
        // Iterate on the group of leaves
        // here we get the first level of cells (leaf_level up to the top_level)
        auto tree_height = target_tree.height();
        int leaf_level = int(tree_height) - 1;
        auto const& period = target_tree.box().get_periodicity();
        auto const& top_level = target_tree.box().is_periodic() ? 1 : 2;
        //
        // auto cell_source_level_it = std::get<1>(source_tree.begin()) + leaf_level;
        for(int level = leaf_level; level >= top_level; --level)
        {
            auto separation_criterion = level == leaf_level ? leaf_neighbour_separation : neighbour_separation;

            // target
            auto group_of_cell_begin = target_tree.begin_mine_cells(level);
            auto group_of_cell_end = target_tree.end_mine_cells(level);
            // source
            // auto begin_of_source_cell_groups = std::begin(*cell_source_level_it);
            // auto end_of_source_cell_groups = std::end(*cell_source_level_it);
            auto begin_of_source_cell_groups = source_tree.begin_cells(level);
            auto end_of_source_cell_groups = source_tree.end_cells(level);
            // loop on target group

            component::for_each(group_of_cell_begin, group_of_cell_end,
                                [begin_of_source_cell_groups, end_of_source_cell_groups, level, &period,
                                 &separation_criterion](auto& group_target)
                                {
                                    static constexpr int prio{scalfmm::algorithms::omp::priorities::max};

#pragma omp task untied default(none)                                                                                  \
  firstprivate(group_target, begin_of_source_cell_groups, end_of_source_cell_groups, level)                            \
  shared(period, separation_criterion) priority(prio)
                                    {
                                        // loop on target cell group
                                        component::for_each(
                                          std::begin(*group_target), std::end(*group_target),
                                          [&group_target, begin_of_source_cell_groups, end_of_source_cell_groups, level,
                                           &period, &separation_criterion](auto& cell)
                                          {
                                              list::build_m2l_interaction_list_for_group(
                                                *group_target, cell, begin_of_source_cell_groups,
                                                end_of_source_cell_groups, level, period, separation_criterion);
                                          });
                                    }
                                });

            // --cell_source_level_it;
        }
#pragma omp taskwait
        target_tree.is_interaction_m2l_lists_built() = true;
    }

    /**
     * @brief Construct the P2P and the M2L interaction lists for the target tree.
     *
     * @tparam SourceTreeType
     * @tparam TargetTreeType
     * @param[in] source_tree the tree containing the sources.
     * @param[inout] target_tree the tree containing the targets.
     * @param[in] neighbour_separation separation criterion use to separate teh near and the far field.
     * @param[in] mutual boolean to specify if the direct pass use a symmetric algorithm (mutual interactions).
     */
    template<typename SourceTreeType, typename TargetTreeType>
    inline auto build_interaction_lists(SourceTreeType& source_tree, TargetTreeType& target_tree,
                                        const int& neighbour_separation, const bool mutual) -> void
    {
#pragma omp parallel default(none) shared(source_tree, target_tree, neighbour_separation, mutual)
        {
#pragma omp single nowait
            {
                build_m2l_interaction_list(source_tree, target_tree, neighbour_separation, neighbour_separation);
            }
#pragma omp single
            {
                build_p2p_interaction_list(source_tree, target_tree, neighbour_separation, mutual);
            }
        }
    }

    /**
     * @brief Construct the P2P and the M2L interaction lists for the target tree
     *
     * @tparam SourceTreeType
     * @tparam TargetTreeType
     * @tparam FmmOperatorType
     * @param[in] source_tree the tree containing the sources
     * @param[inout] target_tree the tree containing the targets.
     * @param[in] fmm_operator The FMM operator used to get separtion criterion and mutual compultation
     * @param[in] verbose boolean to specify if the direct pass use a symmetric algorithm (mutual interactions)
     */
    template<typename SourceTreeType, typename TargetTreeType,
             typename FmmOperatorType>   //NearFieldType, typename NearFieldType>
    inline auto build_interaction_lists(SourceTreeType& source_tree, TargetTreeType& target_tree,
                                        FmmOperatorType const& fmm_operator, bool verbose = false) -> void
    {
        int leaf_neighbour_separation{fmm_operator.near_field().separation_criterion()};
        int neighbour_separation{fmm_operator.near_field().separation_criterion()};
        // if constexpr(scalfmm::meta::is_smooth_v<typename FmmOperatorType::near_field_type::matrix_kernel_type>)
        // {
        //     leaf_neighbour_separation = fmm_operator.near_field().matrix_kernel().leaf_separation_criterion;
        // }
        bool const& mutual = fmm_operator.near_field().mutual();

        if(verbose)
        {
            std::cout << "list::neighbour_separation      " << neighbour_separation
                      << std::endl
                      //   << "list::leaf_neighbour_separation " << leaf_neighbour_separation << std::endl
                      << "list::mutual                    " << std::boolalpha << mutual << std::endl
                      << std::flush;
        }

        // std::cout << " build_p2p_interaction_list start \n" << std::flush;
        try
        {
            build_p2p_interaction_list(source_tree, target_tree, leaf_neighbour_separation, mutual);
            // std::cout << " build_m2l_interaction_list start \n" << std::flush;
            build_m2l_interaction_list(source_tree, target_tree, neighbour_separation, leaf_neighbour_separation);

            // std::cout << " build_interaction_lists end \n" << std::flush;
        }
        catch(const std::out_of_range& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

}   // namespace scalfmm::list::omp
#endif   // SCALFMM_LISTS_OMP_HPP
