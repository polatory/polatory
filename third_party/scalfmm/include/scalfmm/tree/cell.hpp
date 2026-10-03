// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tree/cell.hpp
// --------------------------------
#ifndef SCALFMM_TREE_CELL_HPP
#define SCALFMM_TREE_CELL_HPP

#include "scalfmm/container/point.hpp"
#include "scalfmm/container/variadic_adaptor.hpp"
#include "scalfmm/memory/storage.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/tree/group.hpp"
#include "scalfmm/tree/header.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xfixed.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xtensor_config.hpp"
#include "xtensor/core/xtensor_forward.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace scalfmm::component
{
    /**
     * @brief The cell type stores the multipoles and the local expansions
     *
     * @tparam Storage the storage type that gives the interface the storage of the cell.
     * @tparam D
     */
    template<typename Storage, typename D = void>
    class alignas(XTENSOR_FIXED_ALIGN) cell : public Storage
    {
      public:
        using storage_type = Storage;
        using self_type = cell<storage_type>;
        using value_type = typename storage_type::value_type;
        using symbolics_type = symbolics_data<std::conditional_t<std::is_void_v<D>, self_type, D>>;
        using coordinate_type = typename symbolics_type::coordinate_type;
        using position_type = typename symbolics_type::position_type;

        // Constructors generated

        /**
         * @brief Construct a new cell object
         *
         */
        cell() = default;

        /**
         * @brief Construct a new cell object
         *
         */
        cell(cell const&) = default;

        /**
         * @brief Construct a new cell object
         *
         */
        cell(cell&&) noexcept = default;

        /**
         * @brief
         *
         * @return cell&
         */
        inline auto operator=(cell const&) -> cell& = default;

        /**
         * @brief
         *
         * @return cell&
         */
        inline auto operator=(cell&&) noexcept -> cell& = default;

        /**
         * @brief Destroy the cell object
         *
         */
        ~cell() = default;

        /**
         * @brief Construct a new cell object
         *
         * @param center : the center of the cell
         * @param width : the width of the cell
         * @param order : the order of the approximation
         */
        explicit cell(position_type const& center, value_type width, std::size_t order)
          : storage_type(order)
          , m_center(center)
          , m_width(width)
          , m_order(order)
        {
        }

        /**
         * @brief Construct a new cell object
         *
         * @param center : the center of the cell
         * @param width : the width of the cell
         * @param order : the order of the approximation
         * @param level : the level of the cell
         * @param morton_index : the morton index of the cell
         * @param coordinate_in_tree : the coordinate of the cell in the tree
         */
        explicit cell(position_type const& center, value_type width, std::size_t order, std::size_t level,
                      std::size_t morton_index, coordinate_type const& coordinate_in_tree)
          : storage_type(order)
          , m_center(center)
          , m_width(width)
          , m_order(order)
          , m_symbolics{level, morton_index, coordinate_in_tree}
        {
        }

        /**
         * @brief Access to the symbolic type
         *
         * @return a symbolics_type reference
         */
        [[nodiscard]] inline auto symbolics() -> symbolics_type& { return m_symbolics; }

        /**
         * @brief Access to the symbolic type
         *
         * @return a const symbolics_type reference
         */
        [[nodiscard]] inline auto symbolics() const -> symbolics_type const& { return m_symbolics; }

        /**
         * @brief Access to the symbolic type
         *
         * @return a const symbolics_type reference
         */
        [[nodiscard]] inline auto csymbolics() const -> symbolics_type const& { return m_symbolics; }

        /**
         * @brief Returns the width of the leaf
         *
         * @return value_type
         */
        [[nodiscard]] inline auto width() const noexcept -> value_type { return m_width; }

        /**
         * @brief Set the width object
         *
         * @param in_width teh size of teh cell
         */
        inline auto set_width(value_type in_width) -> void { m_width = in_width; }

        /**
         * @brief Returns the center of the leaf
         *
         * @return position_type
         */
        [[nodiscard]] inline auto center() const noexcept -> position_type const& { return m_center; }

        /**
         * @brief Returns the order of the approximation
         *
         * @return std::size_t
         */
        [[nodiscard]] inline auto order() const noexcept -> std::size_t { return m_order; }

        /**
         * @brief Returns the morton index of the cell
         *
         * @return std::size_t
         */
        [[nodiscard]] inline auto index() const noexcept -> std::size_t { return m_symbolics.morton_index; }

      private:
        /**
         * @brief
         *
         */
        position_type m_center{};

        /**
         * @brief
         *
         */
        value_type m_width{};

        /**
         * @brief
         *
         */
        std::size_t m_order{};

        /**
         * @brief
         *
         */
        symbolics_type m_symbolics{};
    };

    /**
     * @brief The symbolics type stores information about the cell
     * It represents a generic that also exists on the leaves
     *
     * @tparam S
     */
    template<typename S>
    struct symbolics_data<cell<S>>
    {
        // the storage type
        using storage_type = S;
        // the cell type
        using component_type = cell<S>;
        // the group type
        using group_type = group<component_type>;
        // the position value type
        using position_value_type = typename storage_type::value_type;
        // the position type
        using position_type = container::point<position_value_type, component_type::dimension>;
        // the morton_type
        using morton_type = std::size_t;
        // the coordinate type to store the coordinate in the tree
        using coordinate_type = container::point<std::int64_t, component_type::dimension>;
        // the number of interactions of the cell
        static constexpr std::size_t number_of_interactions{math::pow(6, component_type::dimension) -
                                                            math::pow(3, component_type::dimension)};
        // type of the array storing the indexes of the theoretical interaction list
        using interaction_index_array_type = std::array<std::size_t, number_of_interactions>;
        // type of the array storing the linear position of the interaction.
        using interaction_position_array_type =
          std::array<typename coordinate_type::value_type, number_of_interactions>;
        // type of the array storing the iterators to all the interacting cells
        using iterator_array_type = std::array<typename group_type::iterator_type, number_of_interactions>;
        using iterator_type = typename iterator_array_type::value_type;

        // the level of the cell
        std::size_t level{0};
        // the morton index of the cell
        morton_type morton_index{0};
        // the coordinate in the tree
        coordinate_type coordinate_in_tree{};
        // the array storing the indexes of the theoretical interaction list
        interaction_index_array_type interaction_indexes{};
        // type of the array storing the linear position of the interaction, this is necessary to get the corresponding
        // interaction matrix.
        interaction_position_array_type interaction_positions{};
        // the array storing the iterators to all the existing interacting cells
        iterator_array_type interaction_iterators{};
        // theoretical number of neighbors
        std::size_t number_of_neighbors{0};
        // existing number of neighbors
        std::size_t existing_neighbors{0};

        /**
         * @brief
         *
         * @param counter
         * @param idx
         * @param cell_iter
         */
        auto set(int counter, std::size_t const& idx, const iterator_type& cell_iter) -> void
        {
            interaction_positions.at(counter) = interaction_positions.at(idx);
            interaction_iterators.at(counter) = cell_iter;
        }

        /**
         * @brief
         *
         * @param done
         * @param counter_existing_component
         */
        auto finalize(bool done, std::size_t const& counter_existing_component) -> void
        {
            existing_neighbors = counter_existing_component;
        }
    };

    /**
     * @brief
     *
     * @tparam S
     */
    template<typename S>
    struct symbolics_data<group<cell<S>>>
    {
        using storage_type = S;
        using component_type = cell<S>;
        using group_type = group<component_type>;
        using self_type = symbolics_data<group<component_type>>;
        using morton_type = std::size_t;
        // using morton_type = typename symbolics_data<component_type>::morton_type;

        /**
         * @brief the starting morton index in the group.
         *
         */
        morton_type starting_index{0};

        /**
         * @brief the ending morton index in the group.
         *
         */
        morton_type ending_index{0};

        /**
         * @brief number of cells in the group.
         *
         */
        std::size_t number_of_component_in_group{0};

        /**
         * @brief Global index of the group in the Octree at each level
         *
         */
        std::size_t idx_global{0};

        /**
         * @brief
         *
         */
        bool is_mine{false};

        // debug
        std::size_t idx_global_tree{0};
#if _OPENMP
        using ptr_multi_dependency_type = const std::decay_t<decltype(std::declval<S>().cmultipoles(0))>*;
        using ptr_loc_dependency_type = const std::decay_t<decltype(std::declval<S>().clocals(0))>*;
        std::vector<ptr_multi_dependency_type> group_dependencies_m2m_in{};
        std::vector<ptr_multi_dependency_type> group_dependencies_l2l_in{};
        /// The group dependencies on multipole for M2L (transfer) pass
        std::vector<ptr_loc_dependency_type> group_dependencies_m2l{};
#endif
    };

}   // namespace scalfmm::component

#endif   // SCALFMM_TREE_CELL_HPP
