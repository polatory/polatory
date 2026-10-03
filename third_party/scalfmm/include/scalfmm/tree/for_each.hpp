// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tree/for_each.hpp
// --------------------------------
#ifndef SCALFMM_TREE_FOR_EACH_HPP
#define SCALFMM_TREE_FOR_EACH_HPP

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>

namespace scalfmm::component
{
    /**
     * @brief generate a lambda function to iterate linearly over a vector of component groups
     *
     *  The method generates a lambda function to iterate linearly over a vector
     *   of component groups. This allows us to iterate in a for loop. The size of
     *   the loop MUST be lower than the number of components. If not a Segmentation fault may occur.
     *   To move to the next cell in the structure, we call the operator () on the lambda.
     * example
     * @code {.c++}
     *   auto it_group = tree.begin_mine_cells(level);
     *   auto it_cell = it_group->get()->begin();
     *   auto next = scalfmm::component::generate_linear_iterator(gs, it_group, it_cell);
     *   int nmax{3} //
     *   for(int i = 0; i < nmax; ++i, next())
     *   {
     *       std::cout << i << " " << *it_group << std::endl << std::flush;
     *  }
     * @endcode
     *
     * @tparam GroupIterator
     * @tparam ComponentIterator
     * @param gs   not used
     * @param it_group  the iterator of the first group
     * @param it_cell,the first cell of the fist group
     * @return auto
     */
    template<typename GroupIterator, typename ComponentIterator>
    inline auto generate_linear_iterator(int gs, GroupIterator& it_group, ComponentIterator& it_cell)
    {
        return [&it_group, &it_cell, gs, n = int(1)]() mutable
        {
            // if(n % gs == 0)
            // auto ptr = it_group->get();
            if(n == it_group->get()->size())
            {
                ++it_group;
                it_cell = it_group->get()->begin();
                n = 1;
            }
            else
            {
                ++it_cell;
            }
            ++n;
        };
    }

    /**
     * @brief
     *
     * @tparam ComponentIterator
     * @tparam F
     * @param begin
     * @param end
     * @param f
     * @return auto
     */
    template<typename ComponentIterator, typename F>
    inline auto for_each(ComponentIterator begin, ComponentIterator end, F&& f)
    {
        std::for_each(begin, end, std::forward<F>(f));
        // TODO: parallel version
        // while(begin != end)
        // {
        //     std::invoke(std::forward<F>(f), *begin);
        //     ++begin;
        // }
        return begin;
    }

    /**
     * @brief
     *
     * code example to print leaf info
     *  @code
     *    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
     *                                    [&tree](auto& leaf) { scalfmm::io::print_leaf(leaf); });
     * @endcode
     *
     * @tparam InputTreeIterator
     * @tparam UnaryFunction
     * @param begin begin iterator of the group tree
     * @param end end iterator of the group tree
     * @param f lambda function
     * @return UnaryFunction
     */
    template<typename InputTreeIterator, typename UnaryFunction>
    inline auto for_each_leaf(InputTreeIterator begin, InputTreeIterator end, UnaryFunction f) -> UnaryFunction
    {
        auto group_leaf_iterator_begin = std::get<0>(begin);
        auto group_leaf_iterator_end = std::get<0>(end);

        for(; group_leaf_iterator_begin != group_leaf_iterator_end; ++group_leaf_iterator_begin)
        {
            for(auto&& leaf: (*group_leaf_iterator_begin)->components())
            {
                f(leaf);
            }
        }

        return f;
    }

    /**
     * @brief
     *
     * @tparam InputTreeIterator
     * @tparam UnaryFunction
     * @param begin
     * @param end
     * @param f
     * @return UnaryFunction
     */
    template<typename InputTreeIterator, typename UnaryFunction>
    inline auto for_each_mine_leaf(InputTreeIterator begin, InputTreeIterator end, UnaryFunction f) -> UnaryFunction
    {
        for(auto group_leaf_iterator_begin = begin; group_leaf_iterator_begin != end; ++group_leaf_iterator_begin)
        {
            for(auto&& leaf: (*group_leaf_iterator_begin)->components())
            {
                f(leaf);
            }
        }

        return f;
    }
    /// @brief  For iterating on the components between the begin and the end iterator
    /// @tparam InputTreeIterator
    /// @tparam UnaryFunction
    /// @param begin
    /// @param end
    /// @param f
    /// @return
    template<typename InputTreeIterator, typename UnaryFunction>
    inline auto for_each_mine_component(InputTreeIterator begin, InputTreeIterator end,
                                        UnaryFunction f) -> UnaryFunction
    {
        for(auto group_leaf_iterator_begin = begin; group_leaf_iterator_begin != end; ++group_leaf_iterator_begin)
        {
            for(auto&& leaf: (*group_leaf_iterator_begin)->components())
            {
                f(leaf);
            }
        }

        return f;
    }
    /**
     * @brief iterate en two (same) leaf struture (same groupe size)
     *
     * @tparam InputTreeIterator
     * @tparam BinaryFunction
     * @param begin begin iterator of the first group tree
     * @param end end iterator of the first group tree
     * @param begin2 begin iterator on second lgroup tree
     * @param f
     * @return BinaryFunction
     */
    template<typename InputTreeIterator, typename BinaryFunction>
    inline auto for_each_leaf(InputTreeIterator begin, InputTreeIterator end, InputTreeIterator begin2,
                              BinaryFunction f) -> BinaryFunction
    {
        auto group_leaf_iterator_begin = std::get<0>(begin);
        auto group_leaf_iterator_end = std::get<0>(end);
        auto group_leaf_iterator_begin2 = std::get<0>(begin2);

        for(; group_leaf_iterator_begin != group_leaf_iterator_end; ++group_leaf_iterator_begin)
        {
            auto begin_leaf = std::begin((*group_leaf_iterator_begin)->components());
            auto begin_leaf2 = std::begin((*group_leaf_iterator_begin2)->components());
            for(std::size_t i{0}; i < (*group_leaf_iterator_begin)->components().size(); ++i)
            {
                f(*begin_leaf, *begin_leaf2);
                ++begin_leaf;
                ++begin_leaf2;
            }
            ++group_leaf_iterator_begin2;
        }

        return f;
    }

    /**
     * @brief
     *
     * code example to print cell info at level 3
     *  @code
     *     calfmm::component::for_each_cell(tree.begin(), tree.end(), 3,
     *                                    [&tree](auto& cell) { scalfmm::io::print_cell(cell); });
     *
     * @endcode
     *
     * @tparam InputTreeIterator
     * @tparam UnaryFunction
     * @param begin begin iterator of the group tree
     * @param end end iterator of the group tree
     * @param level level of the tree to iterate
     * @param f lambda function
     * @return UnaryFunction
     */
    template<typename InputTreeIterator, typename UnaryFunction>
    inline auto for_each_cell(InputTreeIterator begin, InputTreeIterator end, std::size_t level,
                              UnaryFunction f) -> UnaryFunction
    {
        using iterator_type = std::decay_t<decltype(std::get<1>(begin))>;
        using difference_type = typename iterator_type::difference_type;

        auto cell_level = *(std::get<1>(begin) + static_cast<difference_type>(level));
        auto group_cell_iterator_begin = std::begin(cell_level);
        auto group_cell_iterator_end = std::end(cell_level);

        for(; group_cell_iterator_begin != group_cell_iterator_end; ++group_cell_iterator_begin)
        {
            for(auto&& cell: (*group_cell_iterator_begin)->components())
            {
                f(cell);
            }
        }

        return f;
    }

    /**
     * @brief Iterate both en leaves and cells at leaf level
     *
     * @tparam InputTreeIterator
     * @tparam BinaryFunction
     * @param begin begin iterator of the group tree
     * @param end end iterator of the group tree
     * @param tree_height  tree height
     * @param f
     * @return BinaryFunction
     */
    template<typename InputTreeIterator, typename BinaryFunction>
    inline auto for_each_cell_leaf(InputTreeIterator begin, InputTreeIterator end, std::size_t tree_height,
                                   BinaryFunction f) -> BinaryFunction
    {
        using iterator_type = std::decay_t<decltype(std::get<1>(begin))>;
        using difference_type = typename iterator_type::difference_type;

        auto cell_level = *(std::get<1>(begin) + static_cast<difference_type>(tree_height - 1));

        auto group_cell_iterator_begin = std::begin(cell_level);
        auto group_cell_iterator_end = std::end(cell_level);
        auto group_leaf_iterator_begin = std::get<0>(begin);
        //    auto group_leaf_iterator_end = std::get<0>(end);

        for(; group_cell_iterator_begin != group_cell_iterator_end; ++group_cell_iterator_begin)
        {
            auto cell_it = std::begin((*group_cell_iterator_begin)->components());
            auto cell_end = std::end((*group_cell_iterator_begin)->components());
            auto leaf_it = std::begin((*group_leaf_iterator_begin)->components());
            auto leaf_end = std::end((*group_leaf_iterator_begin)->components());

            while(cell_it != cell_end && leaf_it != leaf_end)
            {
                f(*cell_it, *leaf_it);
                ++cell_it;
                ++leaf_it;
            }
            ++group_leaf_iterator_begin;
        }

        return f;
    }
}   // namespace scalfmm::component

#endif   // SCALFMM_TREE_FOR_EACH_HPP
