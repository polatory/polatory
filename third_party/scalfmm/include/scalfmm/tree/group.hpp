// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tree/group.hpp
// --------------------------------
#ifndef SCALFMM_TREE_GROUP_HPP
#define SCALFMM_TREE_GROUP_HPP

#include "scalfmm/meta/traits.hpp"
#include "scalfmm/tree/header.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/source_target.hpp"

#include <cpp_tools/colors/colorized.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

namespace scalfmm::component
{
    /**
     * @brief Group class class manages cells or leaves in block allocation
     *
     * @tparam Component
     */
    template<typename Component>
    struct group
    {
      public:
        using component_type = Component;
        using symbolics_type = symbolics_data<group<Component>>;
        using block_type = std::vector<component_type>;
        using iterator_type = typename block_type::iterator;
        using const_iterator_type = typename block_type::const_iterator;
        using group_source_type = group;
        using iterator_source_type = iterator_type;

        /**
         * @brief Construct a new group object
         *
         */
        group() = default;

        /**
         * @brief Construct a new group object
         *
         */
        group(group const&) = default;

        /**
         * @brief Construct a new group object
         *
         */
        group(group&&) noexcept = default;

        /**
         * @brief
         *
         * @return group&
         */
        inline auto operator=(group const&) -> group& = default;

        /**
         * @brief
         *
         * @return group&
         */
        inline auto operator=(group&&) noexcept -> group& = default;

        /**
         * @brief Destroy the group object
         *
         */
        ~group() = default;

        /**
         * @brief Construct a new group object
         *
         * @param starting_morton_idx
         * @param ending_morton_idx
         * @param number_of_component
         * @param index_global
         * @param is_mine
         */
        explicit group(std::size_t starting_morton_idx, std::size_t ending_morton_idx, std::size_t number_of_component,
                       std::size_t index_global = 0, bool is_mine = true)
          : m_vector_of_component(number_of_component)
          , m_symbolics{starting_morton_idx, ending_morton_idx, number_of_component, index_global, is_mine}
          , m_number_of_component{number_of_component}
        {
        }

        /**
         * @brief
         *
         * @tparam S
         */
        template<typename S = symbolics_type>
        explicit group(std::size_t starting_morton_idx, std::size_t ending_morton_idx, std::size_t number_of_component,
                       std::size_t number_of_particles_in_group, std::size_t index_global = 0, bool is_mine = true,
                       typename std::enable_if_t<meta::is_leaf_group_symbolics<S>::value>* /*unused*/ = nullptr)
          : m_vector_of_component(number_of_component)
          , m_number_of_component{number_of_component}
        {
            m_symbolics.starting_index = starting_morton_idx;
            m_symbolics.ending_index = ending_morton_idx;
            m_symbolics.number_of_component_in_group = number_of_component;
            m_symbolics.idx_global = index_global;
            m_symbolics.is_mine = is_mine;
        }

        /**
         * @brief
         *
         * @tparam S
         */
        template<typename S = symbolics_type>
        explicit group(std::size_t starting_morton_idx, std::size_t ending_morton_idx, std::size_t number_of_component,
                       std::size_t index_global = 0, bool is_mine = true,
                       typename std::enable_if_t<meta::is_leaf_group_symbolics<S>::value>* /*unused*/ = nullptr)
          : m_vector_of_component(number_of_component)
          , m_number_of_component{number_of_component}
        {
            m_symbolics.starting_index = starting_morton_idx;
            m_symbolics.ending_index = ending_morton_idx;
            m_symbolics.number_of_component_in_group = number_of_component;
            m_symbolics.idx_global = index_global;
            m_symbolics.is_mine = is_mine;
        }

        /**
         * @brief  Display the elements of the group (views and the storage)
         *
         * @param os  the stream
         * @param group teh group to print
         * @return std::ostream&
         */
        inline friend auto operator<<(std::ostream& os, group const& grp) -> std::ostream&
        {
            grp.print(os);
            return os;
        }

        /**
         * @brief return the symbolic structure of the group
         *
         * @return symbolics_type&
         */
        [[nodiscard]] inline auto symbolics() -> symbolics_type& { return m_symbolics; }

        /**
         * @brief return the symbolic structure of the group
         *
         * @return symbolics_type&
         */
        [[nodiscard]] inline auto symbolics() const -> symbolics_type const& { return m_symbolics; }

        /**
         * @brief return the symbolic structure of the group
         *
         * @return symbolics_type&
         */
        [[nodiscard]] inline auto csymbolics() const -> symbolics_type const& { return m_symbolics; }

        /**
         * @brief
         *
         * @return block_type&
         */
        [[nodiscard]] inline auto components() -> block_type& { return m_vector_of_component; }

        /**
         * @brief
         *
         * @return block_type&
         */
        [[nodiscard]] inline auto components() const -> block_type const& { return m_vector_of_component; }

        /**
         * @brief
         *
         * @return block_type&
         */
        [[nodiscard]] inline auto ccomponents() const -> block_type const& { return m_vector_of_component; }

        /**
         * @brief
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto begin() -> iterator_type { return std::begin(m_vector_of_component); }

        /**
         * @brief
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto begin() const -> const_iterator_type { return std::cbegin(m_vector_of_component); }

        /**
         * @brief
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto cbegin() const -> const_iterator_type { return std::cbegin(m_vector_of_component); }

        /**
         * @brief
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto end() -> iterator_type { return std::end(m_vector_of_component); }

        /**
         * @brief
         *
         * @return const_iterator_type
         */
        [[nodiscard]] inline auto end() const -> const_iterator_type { return std::cend(m_vector_of_component); }

        /**
         * @brief
         *
         * @return const_iterator_type
         */
        [[nodiscard]] inline auto cend() const -> const_iterator_type { return std::cend(m_vector_of_component); }

        /**
         * @brief
         *
         * @return auto
         */
        [[nodiscard]] inline auto size() const noexcept { return m_number_of_component; }

        /**
         * @brief Check if morton_index is inside the range of morton indexes of the group
         *
         * @param morton_index  morton index
         * @return true if morton_index is inside the group
         * @return false otherwise
         */
        [[nodiscard]] inline auto is_inside(std::size_t morton_index) const -> bool
        {
            return ((morton_index >= m_symbolics.starting_index) && (morton_index < m_symbolics.ending_index));
        }

        /**
         * @brief Check if morton_index is below the last morton index of the group
         *
         * @param morton_index  morton index
         * @return true if morton_index is below the last morton index of the group
         * @return false otherwise
         */
        [[nodiscard]] inline auto is_below(std::size_t morton_index) const -> bool
        {
            return ((morton_index < m_symbolics.ending_index));
        }

        /**
         * @brief Check if morton_index exists inside the group
         *
         * @param morton_index  morton index
         * @return true if morton_index exists inside the group
         * @return false otherwise
         */
        [[nodiscard]] inline auto exists(std::size_t morton_index) const -> bool
        {
            return is_inside(morton_index) && (component_index(morton_index) != -1);
        }

        /**
         * @brief
         *
         * @param component_index
         * @return component_type&
         */
        [[nodiscard]] inline auto component(std::size_t component_index) -> component_type&
        {
            assertm(component_index < std::size(m_vector_of_component), "Out of range in group.");
            return m_vector_of_component.at(component_index);
        }

        /**
         * @brief
         *
         * @param component_index
         * @return component_type const&
         */
        [[nodiscard]] inline auto component(std::size_t component_index) const -> component_type const&
        {
            assertm(component_index < std::size(m_vector_of_component), "Out of range in group.");
            return m_vector_of_component.at(component_index);
        }

        /**
         * @brief
         *
         * @param component_index
         * @return component_type const&
         */
        [[nodiscard]] inline auto ccomponent(std::size_t component_index) const -> component_type const&
        {
            assertm(component_index < std::size(m_vector_of_component), "Out of range in group.");
            return m_vector_of_component.at(component_index);
        }

        /**
         * @brief
         *
         * @param cell_index
         * @return iterator_type
         */
        [[nodiscard]] inline auto component_iterator(std::size_t cell_index) -> iterator_type
        {
            assertm(cell_index < std::size(m_vector_of_component), "Out of range in group.");
            auto it = std::begin(m_vector_of_component);
            std::advance(it, cell_index);
            return it;
        }

        /**
         * @brief
         *
         * @param cell_index
         * @return const_iterator_type
         */
        [[nodiscard]] inline auto component_iterator(std::size_t cell_index) const -> const_iterator_type
        {
            assertm(cell_index < std::size(m_vector_of_component), "Out of range in group.");
            auto it = std::cbegin(m_vector_of_component);
            std::advance(it, cell_index);
            return it;
        }

        /**
         * @brief
         *
         * @param cell_index
         * @return const_iterator_type
         */
        [[nodiscard]] inline auto ccomponent_iterator(std::size_t cell_index) const -> const_iterator_type
        {
            assertm(cell_index < std::size(m_vector_of_component), "Out of range in group.");
            auto it = std::cbegin(m_vector_of_component);
            std::advance(it, cell_index);
            return it;
        }

        /**
         * @brief
         *
         * @tparam MortonIndex
         * @param morton_index
         * @return int
         */
        template<typename MortonIndex = std::size_t>
        [[nodiscard]] inline auto component_index(MortonIndex morton_index) const -> int
        {
            int idx_left{0};
            int idx_right{static_cast<int>(m_number_of_component) - 1};
            while(idx_left <= idx_right)
            {
                const int idx_middle = (idx_left + idx_right) / 2;
                auto component_index{m_vector_of_component.at(static_cast<std::size_t>(idx_middle)).index()};
                if(component_index == morton_index)
                {
                    return idx_middle;
                }
                if(morton_index < component_index)
                {
                    idx_right = idx_middle - 1;
                }
                else
                {
                    idx_left = idx_middle + 1;
                }
            }
            return -1;
        }

        /**
         * @brief
         *
         * @tparam MortonIndex
         * @param morton_index
         * @param idx_left
         * @param idx_right
         * @return int
         */
        template<typename MortonIndex = std::size_t>
        [[nodiscard]] inline auto component_index(MortonIndex morton_index, int idx_left, int idx_right) const -> int
        {
            while(idx_left <= idx_right)
            {
                const int idx_middle = (idx_left + idx_right) / 2;
                auto component_index{m_vector_of_component.at(static_cast<std::size_t>(idx_middle)).index()};
                if(component_index == morton_index)
                {
                    return idx_middle;
                }
                if(morton_index < component_index)
                {
                    idx_right = idx_middle - 1;
                }
                else
                {
                    idx_left = idx_middle + 1;
                }
            }
            return -1;
        }

        /**
         * @brief
         *
         * @param os
         */
        inline auto print(std::ostream& os) const -> void
        {
            os << "group: idx=" << m_symbolics.idx_global << " range [" << m_symbolics.starting_index << ", "
               << m_symbolics.ending_index << "[ is_mine " << std::boolalpha << m_symbolics.is_mine << "\n";
        }

        /**
         * @brief
         *
         */
        inline auto print() const -> void { this->print(std::cout); }

      private:
        /**
         * @brief Vector of (component (cells or leaves) inside the group
         *
         */
        block_type m_vector_of_component{};

        /**
         * @brief the symbolic information of the group
         *
         */
        symbolics_type m_symbolics{};

        /**
         * @brief The number of component in the block same as m_vector_of_component.size() @todo to remove.
         *
         */
        const std::size_t m_number_of_component{};
    };

}   // namespace scalfmm::component

#endif   // SCALFMM_TREE_GROUP_HPP
