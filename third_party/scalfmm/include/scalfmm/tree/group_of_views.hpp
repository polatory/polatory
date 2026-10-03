// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tree/group_of_views.hpp
// --------------------------------
#ifndef SCALFMM_TREE_GROUP_OF_PARTICLES_HPP
#define SCALFMM_TREE_GROUP_OF_PARTICLES_HPP

#include "scalfmm/container/block.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/tree/header.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/massert.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <vector>

namespace scalfmm::component
{
    /**
     * @brief Group class class manages leaves in block allocation
     *
     * The group is composed of two elements. The storage which contains the
     *   particles or multipole/local values and symbolic information of the
     *   components in the group and an array of components (view) that allows
     *   to access the data inside the storage.
     *
     * @tparam LeafType
     * @tparam ParticleType
     */
    template<typename LeafType, typename ParticleType>
    struct group_of_particles
    {
      public:
        using particle_type = ParticleType;
        using component_type = LeafType;
        using symbolics_type = symbolics_data<group_of_particles<component_type, particle_type>>;
        using symbolics_component_type = symbolics_data<component_type>;
        using block_type = std::vector<component_type>;
        using iterator_type = typename block_type::iterator;
        using const_iterator_type = typename block_type::const_iterator;
        using storage_type = typename symbolics_type::storage_type;

        /**
         * @brief Construct a new group of particles object
         *
         */
        group_of_particles() = default;

        /**
         * @brief Construct a new group of particles object
         *
         */
        group_of_particles(group_of_particles const&) = default;

        /**
         * @brief Construct a new group of particles object
         *
         */
        group_of_particles(group_of_particles&&) noexcept = default;

        /**
         * @brief
         *
         * @return group_of_particles&
         */
        inline auto operator=(group_of_particles const&) -> group_of_particles& = default;

        /**
         * @brief
         *
         * @return group_of_particles&
         */
        inline auto operator=(group_of_particles&&) noexcept -> group_of_particles& = default;

        /**
         * @brief Destroy the group of particles object
         *
         */
        ~group_of_particles() = default;

        /**
         * @brief Construct a new group of particles object
         *
         * @param starting_morton_idx  Morton index of the first object (leaf or cell) inside the group
         * @param ending_morton_idx    Morton index of the last object (leaf or cell) inside the group
         * @param number_of_component  Number of leaves or cells inside the group
         * @param number_of_particles_in_group  Number of particles inside the group
         * @param index_global         Global index of the group
         * @param is_mine              I am the owner of the group
         */
        group_of_particles(std::size_t starting_morton_idx, std::size_t ending_morton_idx,
                           std::size_t number_of_component, std::size_t number_of_particles_in_group,
                           std::size_t index_global, bool is_mine)
          : m_vector_of_components(number_of_component)
          , m_components_storage(number_of_particles_in_group, number_of_component)
          , m_number_of_component{number_of_component}
        {
            auto& group_symbolics = m_components_storage.header();

            group_symbolics.starting_index = starting_morton_idx;
            group_symbolics.ending_index = ending_morton_idx;
            group_symbolics.number_of_component_in_group = number_of_component;
            group_symbolics.number_of_particles_in_group = number_of_particles_in_group;
            group_symbolics.idx_global = index_global;
            group_symbolics.is_mine = is_mine;
        }

        // /**
        //  * @brief Rebuilt all the pointers inside the vector of blocks
        //  *
        //  */
        // auto rebuilt_leaf_view() -> void
        // {
        //     std::cout << "rebuilt_leaf_view nb view " << m_number_of_component << "   " <<
        //     m_vector_of_components.size()
        //               << std::endl;
        // }

        /**
         * @brief  Display the elements of the group (views and the storage)
         *
         * @param os
         * @param group
         * @return std::ostream&
         */
        inline friend auto operator<<(std::ostream& os, const group_of_particles& group) -> std::ostream&
        {
            os << cpp_tools::colors::green;
            os << "group:" << group.csymbolics().idx_global << " nb_leaves: " << group.size() << " is_mine "
               << group.csymbolics().is_mine << std::endl;
            os << "leaves_view: \n";
            for(auto v: group.components())
            {
                os << v << "\n";
            }
            os << "storage: \n";
            os << group.cstorage() << std::endl;
            os << cpp_tools::colors::reset;
            return os;
        }

        /**
         * @brief Returns the symbolic structure of the group_of_particles
         *
         * @return symbolics_type&
         */
        [[nodiscard]] inline auto symbolics() -> symbolics_type& { return m_components_storage.header(); }

        /**
         * @brief Returns the symbolic structure of the group_of_particles
         *
         * @return symbolics_type&
         */
        [[nodiscard]] inline auto symbolics() const -> symbolics_type const& { return m_components_storage.header(); }

        /**
         * @brief Returns the symbolic structure of the group_of_particles
         *
         * @return symbolics_type&
         */
        [[nodiscard]] inline auto csymbolics() const -> symbolics_type const& { return m_components_storage.cheader(); }

        /**
         * @brief  Get the vector of components (leaves)
         *
         * @return block_type&
         */
        [[nodiscard]] inline auto components() -> block_type& { return m_vector_of_components; }

        /**
         * @brief  Get the vector of components (leaves)
         *
         * @return block_type&
         */
        [[nodiscard]] inline auto components() const -> block_type const& { return m_vector_of_components; }

        /**
         * @brief  Get the vector of components (leaves)
         *
         * @return block_type&
         */
        [[nodiscard]] inline auto ccomponents() const -> block_type const& { return m_vector_of_components; }

        /**
         * @brief Returns the storage of the particles inside the group
         *
         * @return storage_type&
         */
        [[nodiscard]] inline auto storage() -> storage_type& { return m_components_storage; }

        /**
         * @brief Returns the storage of the particles inside the group
         *
         * @return storage_type&
         */
        [[nodiscard]] inline auto storage() const -> storage_type const& { return m_components_storage; }

        /**
         * @brief Returns the storage of the particles inside the group
         *
         * @return storage_type&
         */
        [[nodiscard]] inline auto cstorage() const -> storage_type const& { return m_components_storage; }

        /**
         * @brief Gets the first component iterator
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto begin() -> iterator_type { return std::begin(m_vector_of_components); }

        /**
         * @brief Gets the first component iterator
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto begin() const -> const_iterator_type { return std::cbegin(m_vector_of_components); }

        /**
         * @brief Gets the first component iterator
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto cbegin() const -> const_iterator_type { return std::cbegin(m_vector_of_components); }

        /**
         * @brief Gets the last component iterator
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto end() -> iterator_type { return std::end(m_vector_of_components); }

        /**
         * @brief Gets the last component iterator
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto end() const -> const_iterator_type { return std::cend(m_vector_of_components); }

        /**
         * @brief Gets the last component iterator
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto cend() const -> const_iterator_type { return std::cend(m_vector_of_components); }

        /**
         * @brief Returns the number of components (cells or leaves)
         *
         */
        [[nodiscard]] inline auto size() const noexcept { return m_number_of_component; }

        /**
         * @brief Checks if morton_index is inside the range of morton indexes of the group_of_particles
         *
         * @param morton_index  morton index
         * @return true if morton_index is inside the group_of_particles
         * @return false otherwise
         */
        [[nodiscard]] inline auto is_inside(std::size_t morton_index) const -> bool
        {
            return ((morton_index >= this->csymbolics().starting_index) &&
                    (morton_index < this->csymbolics().ending_index));
        }

        /**
         * @brief Check if morton_index is below the last morton index of the group_of_particles
         *
         * @param morton_index  morton index
         * @return true if morton_index is below the last morton index of the group_of_particles
         * @return false otherwise
         */
        [[nodiscard]] inline auto is_below(std::size_t morton_index) const -> bool
        {
            return ((morton_index < this->csymbolics().ending_index));
        }

        /**
         * @brief Check if morton_index exists inside the group_of_particles
         *
         * @param morton_index  morton index
         * @return true if morton_index exists inside the group_of_particles
         * @return false otherwise
         */
        [[nodiscard]] inline auto exists(std::size_t morton_index) const -> bool
        {
            return is_inside(morton_index) && (component_index(morton_index) != -1);
        }

        /**
         * @brief Returns the targeted component
         *
         * @param component_index  the index of the component
         * @return component_type&
         */
        [[nodiscard]] inline auto component(std::size_t component_index) -> component_type&
        {
            assertm(component_index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            return m_vector_of_components.at(component_index);
        }

        /**
         * @brief
         *
         * @param component_index
         * @return component_type const&
         */
        [[nodiscard]] inline auto component(std::size_t component_index) const -> component_type const&
        {
            assertm(component_index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            return m_vector_of_components.at(component_index);
        }

        /**
         * @brief
         *
         * @param component_index
         * @return component_type const&
         */
        [[nodiscard]] inline auto ccomponent(std::size_t component_index) const -> component_type const&
        {
            assertm(component_index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            return m_vector_of_components.at(component_index);
        }

        /**
         * @brief
         *
         * @param index
         * @return iterator_type
         */
        [[nodiscard]] inline auto component_iterator(std::size_t index) -> iterator_type
        {
            assertm(index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            auto it = std::begin(m_vector_of_components);
            std::advance(it, index);
            return it;
        }

        /**
         * @brief
         *
         * @param index
         * @return const_iterator_type
         */
        [[nodiscard]] inline auto component_iterator(std::size_t index) const -> const_iterator_type
        {
            assertm(index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            auto it = std::cbegin(m_vector_of_components);
            std::advance(it, index);
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
            assertm(cell_index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            auto it = std::cbegin(m_vector_of_components);
            std::advance(it, cell_index);
            return it;
        }

        /**
         * @brief Returns the index of the component with the targeted Morton index
         *
         * @tparam MortonIndex
         * @param morton_index  the morton index
         * @return int the index of the component
         */
        template<typename MortonIndex = std::size_t>
        [[nodiscard]] inline auto component_index(MortonIndex morton_index) const -> int
        {
            int idx_left{0};
            int idx_right{static_cast<int>(m_number_of_component) - 1};
            while(idx_left <= idx_right)
            {
                const int idx_middle = (idx_left + idx_right) / 2;
                auto component_index{m_vector_of_components.at(static_cast<std::size_t>(idx_middle)).index()};
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
                auto component_index{m_vector_of_components.at(static_cast<std::size_t>(idx_middle)).index()};
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
         * @brief Display the interval of Morton index of the components inside the group
         *
         */
        inline auto print_morton_interval() -> void
        {
            std::cout << "group_of_particles: [" << this->csymbolics().starting_index << ", "
                      << this->csymbolics().ending_index << "]\n";
        }

        /**
         * @brief get the dependencies for the task-based algorithm on the first particle output inside the block
         *
         * @return auto
         */
        inline auto depends_update() { return this->cstorage().cptr_on_output(); }

      private:
        /**
         * @brief Vector of components (cells or leaves) inside the group_of_particles.
         *
         */
        block_type m_vector_of_components{};

        /**
         * @brief The storage of the particle or multipole/local values.
         *
         */
        storage_type m_components_storage{};

        /**
         * @brief The number of components in the group.
         *
         */
        const std::size_t m_number_of_component{};
    };

    /**
     * @brief The symbolics type that stores information about the group of leaves.
     *
     * @tparam ComponentType
     * @tparam ParticleType
     */
    template<typename ComponentType, typename ParticleType>
    struct symbolics_data<group_of_particles<ComponentType, ParticleType>>
    {
        // using self_type = symbolics_data<group_of_particles<P,D>>;
        using self_type = symbolics_data<group_of_particles<ComponentType, ParticleType>>;
        // the current group type
        using group_type = group_of_particles<ComponentType, ParticleType>;
        using component_type = ComponentType;
        // the leaf type
        using particle_type = ParticleType;
        static constexpr std::size_t dimension{particle_type::dimension};
        using iterator_type = typename group_type::iterator_type;
        // the distant group type  // same if source = target or same tree
        using seq_iterator_type =
          std::conditional_t<meta::exist_v<meta::inject<group_type>>, meta::exist_t<meta::inject<group_type>>,
                             std::tuple<iterator_type, group_type>>;
        using group_source_type = typename std::tuple_element_t<1, seq_iterator_type>;
        using iterator_source_type = typename std::tuple_element_t<0, seq_iterator_type>;
        using out_of_block_interaction_type = out_of_block_interaction<iterator_source_type, std::size_t>;
        // storage type to store data in group
        using storage_type = container::particles_block<self_type, particle_type, component_type>;

        /**
         * @brief The starting morton index in the group.
         *
         */
        std::size_t starting_index{0};

        /**
         * @brief The ending morton index in the group.
         *
         */
        std::size_t ending_index{0};

        /**
         * @brief
         *
         */
        std::size_t number_of_component_in_group{0};

        /**
         * @brief The number of particles in the group.
         *
         */
        std::size_t number_of_particles_in_group{0};

        /**
         * @brief Index of the group.
         *
         */
        std::size_t idx_global{0};

        /**
         * @brief
         *
         */
        bool is_mine{true};

        /**
         * @brief Vector storing the out_of_block_interaction structure to handle the outside interactions.
         *
         */
        std::vector<out_of_block_interaction_type> outside_interactions{};

        /**
         * @brief Flag if the vector is constructed.
         *
         */
        bool outside_interactions_exists{false};

        /**
         * @brief Flag if the vector is sorted.
         *
         */
        bool outside_interactions_sorted{false};

        // #if _OPENMP
        /**
         * @brief the P2P dependencies are set on the pointer on particles container
         *
         */
        std::vector<group_source_type*> group_dependencies{};
        // #endif

        /**
         * @brief Construct a new symbolics data object
         *
         * @param starting_morton_idx
         * @param ending_morton_idx
         * @param number_of_component
         * @param number_of_particles
         * @param in_index_global
         * @param in_is_mine
         */
        symbolics_data(std::size_t starting_morton_idx, std::size_t ending_morton_idx, std::size_t number_of_component,
                       std::size_t number_of_particles, std::size_t in_index_global, bool in_is_mine)
          : starting_index(starting_morton_idx)
          , ending_index(ending_morton_idx)
          , number_of_component_in_group(number_of_component)
          , number_of_particles_in_group(number_of_particles)
          , idx_global(in_index_global)
          , is_mine(in_is_mine)
        {
        }
    };

}   // namespace scalfmm::component

#endif   // SCALFMM_TREE_GROUP_HPP
