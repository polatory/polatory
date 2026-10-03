// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/omp/upward.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_UPWARD_HPP
#define SCALFMM_ALGORITHMS_OMP_UPWARD_HPP

#include <limits>

#include "scalfmm/operators/m2m.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"

#ifdef _OPENMP
#include <omp.h>

namespace scalfmm::algorithms::omp::pass
{
    /**
     * @brief perform the m2m operator for a given level
     *
     * @tparam TreeType
     * @tparam ApproximationType
     *
     * @param level current level to construct the m2m
     * @param tree the tree
     * @param approximation the approximation
     */
    template<typename TreeType, typename ApproximationType>
    inline auto upward_level(const int& level, TreeType& tree, ApproximationType const& approximation) -> void
    {
        using operators::m2m;
        //
        using interpolator_type = typename std::decay_t<ApproximationType>;
        using value_type = typename interpolator_type::value_type;

        static constexpr auto dimension{interpolator_type::dimension};
        static constexpr auto prio{priorities::m2m};
        // Get the index of the corresponding child-parent interpolator
        std::size_t level_interpolator_index =
          (approximation.cell_width_extension() < std::numeric_limits<value_type>::epsilon()) ? 2 : level;
        //
        // iterator on the groups of cells (child level)
        //
        auto group_of_child_cell_begin = tree.begin_mine_cells(level + 1);
        auto group_of_child_cell_end = tree.end_cells(level + 1);

        // iterator on the groups of cells (current level)
        auto group_of_cell_begin = tree.begin_mine_cells(level);
        auto group_of_cell_end = tree.end_mine_cells(level);

        auto start_range_dependencies{group_of_child_cell_begin};
        auto end_range_dependencies{group_of_child_cell_begin};
        // We iterate on the parent cells

        using ptr_child_groups_type = std::decay_t<decltype(group_of_child_cell_begin->get())>;
        while(group_of_cell_begin != group_of_cell_end)
        {
            auto group_parent = group_of_cell_begin->get();
            auto const& grp_parent_sym = group_parent->csymbolics();
            // pointer on the first multipole of the group
            auto group_parent_raw = &group_parent->ccomponent(0).cmultipoles(0);

            auto& child_dependencies{(*group_of_cell_begin)->symbolics().group_dependencies_m2m_in};

            std::tie(start_range_dependencies, end_range_dependencies) = index::get_child_group_range<dimension>(
              group_of_child_cell_begin, group_of_child_cell_end, *group_parent);
            auto start_range_dependencies_tmp{start_range_dependencies};
            const auto end_range_dependencies_tmp{end_range_dependencies};
            while(start_range_dependencies != end_range_dependencies)
            {
                child_dependencies.push_back(&(*start_range_dependencies)->ccomponent(0).cmultipoles(0));
                ++start_range_dependencies;
            }

            start_range_dependencies = --end_range_dependencies;
            // clang-format off
  #pragma omp task untied default(none) firstprivate( group_parent, start_range_dependencies_tmp, end_range_dependencies_tmp,level_interpolator_index, group_parent_raw)       \
     shared(std::cout, approximation )                                                    \
  depend(iterator(std::size_t it = 0 : grp_parent_sym.group_dependencies_m2m_in.size()),  \
                   in: (grp_parent_sym.group_dependencies_m2m_in.at(it))[0])              \
  depend(inout: group_parent_raw[0]) priority(prio)
            // clang-format on
            {
                // io::print("m2m(task) dependencies(in): ", group_parent->csymbolics().group_dependencies_m2m_in);
                // std::cout << "m2m(task) run  task dep(inout) on group group_parent_raw  " << group_parent_raw
                //           << std::endl
                //           << std::flush;
                std::vector<ptr_child_groups_type> child_groups;

                while(start_range_dependencies_tmp != end_range_dependencies_tmp)
                {
                    child_groups.push_back(start_range_dependencies_tmp->get());
                    ++start_range_dependencies_tmp;
                }

                for(std::size_t cell_index = 0; cell_index < group_parent->size(); ++cell_index)
                {
                    auto& parent_cell = group_parent->component(cell_index);
                    auto parent_morton_index = parent_cell.index();
                    static constexpr auto number_of_child = math::pow(2, dimension);
                    for(auto p: child_groups)
                    {
                        for(auto const& child_cell: p->components())
                        {
                            auto child_morton_index{child_cell.index()};
                            if((child_morton_index >> dimension) == parent_morton_index)
                            {
                                const std::size_t child_index = child_morton_index & (number_of_child - 1);
                                m2m(approximation, child_cell, child_index, parent_cell, level_interpolator_index);
                            }
                        }
                        // std::cout << "m2m(task) end cell index " << parent_cell.index() << " multipoles "
                        //           << parent_cell.multipoles().at(0) << std::endl;
                    }
                    approximation.apply_multipoles_preprocessing(parent_cell, omp_get_thread_num());
                }
            }   // end pragma task
            ++group_of_cell_begin;
        }
        assert(group_of_cell_begin == group_of_cell_end);
    }

    /**
    * @brief This function constructs the local approximation for all the cells of the tree by applying the
    * operator m2m
    *
    * @tparam TreeType
    * @tparam ApproximationType
    *
    * @param tree the tree target
    * @param approximation the approximation to construct the local approximation
    */
    template<typename TreeType, typename ApproximationType>
    inline auto upward(TreeType& tree, ApproximationType const& approximation) -> void
    {
        auto leaf_level = tree.height() - 1;
        //
        // upper working level is
        const int top_height = tree.box().is_periodic() ? 0 : 2;
        for(int level = leaf_level - 1; level >= top_height; --level)   // int because top_height could be 0
        {
            upward_level(level, tree, approximation);
        }
    }
}   // namespace scalfmm::algorithms::omp::pass

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_OMP_UPWARD_HPP
