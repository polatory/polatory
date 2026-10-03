// --------------------------------
// See LICENCE file at project root
// File : algorithm/sequential/upward.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_UPWARD_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_UPWARD_HPP

#include "scalfmm/operators/m2m.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <tuple>
#include <vector>

namespace scalfmm::algorithms::sequential::pass
{
    /**
    * @brief This function builds the multipole for all the cells of the tree by applying the operator m2m.
    *
    * @tparam TreeType
    * @tparam InterpolatorType
    *
    * @param tree
    * @param interpolator
    */
    template<typename TreeType, typename InterpolatorType>
    inline auto upward(TreeType& tree, InterpolatorType const& interpolator) -> void
    {
        using scalfmm::operators::m2m;

        using tree_type = TreeType;
        using interpolator_type = InterpolatorType;
        using cell_type = typename tree_type::cell_type;
        using value_type = typename interpolator_type::value_type;

        static constexpr auto dimension = cell_type::dimension;
        static constexpr auto number_of_child = math::pow(2, dimension);

        auto begin = std::begin(tree);
        auto leaf_level = tree.height() - 1;

        auto cell_level_it = std::get<1>(begin) + leaf_level;

        // upper working level is 1 because we work at the parent level inside the loop
        const int top_height = tree.box().is_periodic() ? 0 : 2;
        // start to the level above the leaf level
        for(int level = leaf_level - 1; level >= top_height; --level)   // int because top_height could be 0
        {
            auto group_of_child_cell_begin = std::cbegin(*(cell_level_it));
            auto group_of_child_cell_end = std::cend(*(cell_level_it));
            std::advance(cell_level_it, -1);
            // Here cell_level_it is at parent level for the next loop iteration
            auto group_of_cell_begin = std::begin(*cell_level_it);
            auto group_of_cell_end = std::end(*cell_level_it);
            std::size_t index_child_cell{0};
            // Get the index of the corresponding child-parent interpolator
            std::size_t level_interpolator_index = 2;
            if(interpolator.cell_width_extension() > std::numeric_limits<value_type>::epsilon())
                level_interpolator_index = level;
            //
            while(group_of_cell_begin != group_of_cell_end && group_of_child_cell_begin != group_of_child_cell_end)
            {
                {   // Can be a task(in:iterParticles, out:iterChildCells ...)
                    auto cell_begin = std::begin(**group_of_cell_begin);
                    for(std::size_t cell_index = 0;
                        cell_index < (*group_of_cell_begin)->csymbolics().number_of_component_in_group; ++cell_index)
                    {
                        auto& parent_cell = *cell_begin;
                        auto const& parent_symbolics = parent_cell.csymbolics();

                        // using cell_type = std::decay_t<decltype(parent_cell)>;
                        using cell_iterator_type = typename std::decay_t<
                          decltype(*group_of_child_cell_begin)>::element_type::const_iterator_type;
                        std::vector<std::tuple<std::size_t, cell_iterator_type>> indexes_of_childs{};

                        assertm(group_of_child_cell_begin != group_of_child_cell_end,
                                "Upward pass : reach the end of child's cells.");

                        while(group_of_child_cell_begin != group_of_child_cell_end &&
                              (((*group_of_child_cell_begin)->ccomponent(index_child_cell).csymbolics().morton_index >>
                                dimension) == parent_symbolics.morton_index))
                        {
                            auto const& child_cell_group_symbolics = (*group_of_child_cell_begin)->csymbolics();
                            const std::size_t child_index =
                              (*group_of_child_cell_begin)->ccomponent(index_child_cell).csymbolics().morton_index &
                              (number_of_child - 1);

                            indexes_of_childs.emplace_back(std::make_tuple(
                              child_index, (*group_of_child_cell_begin)->ccomponent_iterator(index_child_cell)));

                            ++index_child_cell;
                            if(index_child_cell == child_cell_group_symbolics.number_of_component_in_group)
                            {
                                index_child_cell = 0;
                                ++group_of_child_cell_begin;
                            }
                        }

                        std::for_each(std::cbegin(indexes_of_childs), std::cend(indexes_of_childs),
                                      [&parent_cell, &interpolator, level_interpolator_index](auto indexes) {
                                          m2m(interpolator, *std::get<1>(indexes), std::get<0>(indexes), parent_cell,
                                              level_interpolator_index);
                                      });

                        if(!indexes_of_childs.empty())
                        {
                            interpolator.apply_multipoles_preprocessing(parent_cell);
                        }
                        ++cell_begin;
                    }
                }
                ++group_of_cell_begin;
            }

            assert(group_of_cell_begin == group_of_cell_end);
            assert(group_of_child_cell_begin == group_of_child_cell_end);
        }
    }
}   // namespace scalfmm::algorithms::sequential::pass

#endif   // SCALFMM_ALGORITHMS_SEQUENTIAL_UPWARD_HPP
