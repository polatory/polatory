// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/count_kernel/count_interactions_kernel.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_COUNT_KERNEL_COUNT_KERNEL_HPP
#define SCALFMM_OPERATORS_COUNT_KERNEL_COUNT_KERNEL_HPP

#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/matrix_kernels/mk_common.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace count_kernel
{
    /**
     * @brief
     *
     */
    struct execution_infos
    {
        /**
         * @brief
         *
         */
        const std::string operator_name{};

        /**
         * @brief
         *
         */
        std::size_t particles{0};

        /**
         * @brief
         *
         */
        std::size_t number_of_component{0};

        /**
         * @brief
         *
         */
        std::size_t call_count{0};

        /**
         * @brief
         *
         */
        std::array<std::size_t, 10> multipoles_count{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        /**
         * @brief
         *
         */
        std::array<std::size_t, 10> locals_count{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        /**
         * @brief
         *
         * @param os
         * @param infos
         * @return std::ostream&
         */
        friend inline auto operator<<(std::ostream& os, const execution_infos& infos) -> std::ostream&;
    };

    /**
     * @brief
     *
     * @param os
     * @param infos
     * @return std::ostream&
     */
    inline auto operator<<(std::ostream& os, const execution_infos& infos) -> std::ostream&
    {
        os << "[\n operator : " << infos.operator_name << "\n  particles : " << infos.particles
           << "\n  call_count : " << infos.call_count << '\n';
        std::size_t level{0};
        for(auto l: infos.multipoles_count)
        {
            std::cout << "  multipoles_count[" << level++ << "] : " << l << '\n';
        }
        level = 0;
        for(auto l: infos.locals_count)
        {
            std::cout << "  locals_count[" << level++ << "] : " << l << '\n';
        }
        std::cout << " ]";
        return os;
    }

    /**
     * @brief The count matrix kernel.
     *
     */
    struct count_matrix_kernel
    {
        using map_type = std::unordered_map<std::string, execution_infos>;
        using mapped_type = execution_infos;

        //  static constexpr auto homogeneity_tag{scalfmm::matrix_kernels::homogeneity::homogenous};
        //   static constexpr int km{1};
        //  static constexpr int kn{1};

        //    inline auto separation_criterion() const -> std::size_t { return 1; }
        //     inline auto get(std::string const& key) -> mapped_type& { return m_map.at(key); }

        /**
         * @brief
         *
         */
        inline auto print() -> void
        {
            for(auto const& pair: m_map)
            {
                std::cout << pair.second << '\n';
            }
        }

      private:
        /**
         * @brief
         *
         */
        map_type m_map{{
          {"p2p_full_mutual", {"p2p_full_mutual"}},
          {"p2p_inner", {"p2p_inner"}},
          {"p2p_remote", {"p2p_remote"}},
        }};
    };

    /**
     * @brief An implementation of the near field for the counter kernel
     *
     */
    struct count_near_field
    {
        /**
         * @brief
         *
         */
        count_matrix_kernel mat;

        /**
         * @brief
         *
         * @return std::size_t
         */
        inline auto separation_criterion() const -> std::size_t { return 1; }

        /**
         * @brief
         *
         * @return count_matrix_kernel
         */
        inline count_matrix_kernel matrix_kernel() { return mat; };

        /**
         * @brief
         *
         */
        inline auto print() -> void { mat.print(); }
    };

    /**
     * @brief An implementation of the far field for the counter kernel.
     *
     */
    struct count_far_field
    {
      public:
        using map_type = std::unordered_map<std::string, execution_infos>;
        using map_value_type = typename map_type::value_type;
        using mapped_type = execution_infos;

        //        static constexpr std::size_t dimension{3};
        //        static constexpr std::size_t kn = count_matrix_kernel::kn;
        //        static constexpr std::size_t km = count_matrix_kernel::km;

        /**
         * @brief
         *
         * @param key
         * @return mapped_type&
         */
        inline auto get(std::string const& key) -> mapped_type& { return m_map.at(key); }

        /**
         * @brief
         *
         * @param key
         * @return mapped_type const&
         */
        inline auto get(std::string const& key) const -> mapped_type const& { return m_map.at(key); }

        /**
         * @brief
         *
         */
        inline auto print() -> void
        {
            for(auto const& pair: m_map)
            {
                std::cout << pair.second << '\n';
            }
        }

        /**
         * @brief
         *
         * @return std::size_t
         */
        inline auto separation_criterion() const -> std::size_t { return 1; }

        /**
         * @brief
         *
         * @tparam CellType
         * @param current_cell
         * @param order
         */
        template<typename CellType>
        void apply_multipoles_preprocessing(CellType& current_cell, std::size_t order)
        {
        }

        /**
         * @brief
         *
         * @tparam CellType
         * @param current_cell
         * @param order
         */
        template<typename CellType>
        void apply_multipoles_postprocessing(CellType& current_cell, std::size_t order)
        {
        }

      private:
        /**
         * @brief
         *
         */
        map_type m_map{{
          {"p2m", {"p2m"}},
          {"m2m", {"m2m"}},
          {"m2l", {"m2l"}},
          {"l2l", {"l2l"}},
          {"l2p", {"l2p"}},
          //          {"p2p_full_mutual", {"p2p_full_mutual"}},
          //          {"p2p_inner", {"p2p_inner"}},
          //          {"p2p_remote", {"p2p_remote"}},
        }};
    };

    /**
     * @brief
     *
     */
    struct count_fmm_operator
    {
        /**
         * @brief
         *
         */
        count_far_field m_far_field;

        /**
         * @brief
         *
         */
        count_near_field m_near_field;

        /**
         * @brief
         *
         * @return count_far_field
         */
        inline count_far_field far_field() { return m_far_field; };

        /**
         * @brief
         *
         * @return count_near_field
         */
        inline count_near_field near_field() const { return m_near_field; };

        /**
         * @brief
         *
         */
        inline auto print() -> void
        {
            //            far_field.print();
            //            near_field.print();
        }
    };

    /**
     * @brief
     *
     * @tparam LeafType
     * @tparam CellType
     */
    template<typename LeafType, typename CellType>
    inline void p2m(count_far_field& interp, LeafType const& source_leaf, CellType& target_cell, std::size_t /*order*/)
    {
        auto& infos = interp.get("p2m");
        auto const& target_symb = target_cell.csymbolics();
        auto& multipoles = std::get<0>(target_cell.multipoles());
        multipoles += source_leaf.size();
        infos.call_count++;
        infos.multipoles_count[target_symb.level]++;
        infos.particles += source_leaf.size();
    }

    /**
     * @brief
     *
     * @tparam CellType
     * @param interp
     * @param parent_cell
     * @param child_index
     * @param child_cell
     * @param order
     * @param tree_level
     */
    template<typename CellType>
    inline void l2l(count_far_field& interp, CellType const& parent_cell, std::size_t child_index, CellType& child_cell,
                    std::size_t order, std::size_t tree_level = 2)
    {
        auto& infos = interp.get("l2l");
        auto const& target_symb = child_cell.csymbolics();
        auto& child_locals = std::get<0>(child_cell.locals());
        auto const& current_locals = std::get<0>(parent_cell.clocals());
        child_locals += current_locals;
        infos.call_count++;
        infos.locals_count[target_symb.level]++;
    }

    /**
     * @brief
     *
     * @tparam CellType
     * @param interp
     * @param child_cell
     * @param child_index
     * @param parent_cell
     * @param order
     * @param tree_level
     */
    template<typename CellType>
    inline void m2m(count_far_field& interp, CellType const& child_cell, std::size_t child_index, CellType& parent_cell,
                    std::size_t order, std::size_t tree_level = 2)
    {
        auto& infos = interp.get("m2m");
        auto const& target_symb = parent_cell.csymbolics();
        auto& parent_multipoles = std::get<0>(parent_cell.multipoles());
        auto const& child_multipoles = std::get<0>(child_cell.cmultipoles());
        parent_multipoles += child_multipoles;
        infos.call_count++;
        infos.multipoles_count[target_symb.level]++;
    }

    /**
     * @brief
     *
     * @tparam CellType
     */
    template<typename CellType>
    inline void m2l(count_far_field& interp, CellType const& source_cell, std::size_t /*neighbor_idx*/,
                    CellType& target_cell, std::size_t /*order*/, std::size_t /*tree_level*/)
    {
        auto& infos = interp.get("m2l");
        auto const& target_symb = target_cell.csymbolics();
        auto const& source_multipoles = std::get<0>(source_cell.cmultipoles());
        auto& target_locals = std::get<0>(target_cell.locals());
        target_locals += source_multipoles;
        infos.locals_count[target_symb.level]++;
    }

    /**
     * @brief
     *
     * @tparam CellType
     * @tparam LeafType
     * @tparam ComputeForces
     */
    template<typename CellType, typename LeafType, bool ComputeForces = false>
    inline void l2p(count_fmm_operator& fmm_operator, CellType const& source_cell, LeafType& target_leaf,
                    std::size_t /*order*/)
    {
        auto interp = fmm_operator.far_field();
        auto& infos = interp.get("l2p");
        std::get<0>(*scalfmm::container::outputs_begin(target_leaf.particles())) =
          *std::begin(std::get<0>(source_cell.clocals()));
        infos.call_count++;
    }

    /**
     * @brief
     *
     * @tparam LeafType
     * @tparam ContainerOfLeafIteratorType
     */
    template<typename LeafType, typename ContainerOfLeafIteratorType>
    inline void p2p_full_mutual(count_matrix_kernel const& /*mat*/, LeafType& target_leaf,
                                ContainerOfLeafIteratorType const& neighbor)
    {
        using value_type = typename LeafType::value_type;
        //      auto& infos = mat.get("p2p_full_mutual");
        value_type nb_val{0};
        std::for_each(std::begin(neighbor), std::end(neighbor),
                      [&target_leaf, &nb_val](auto const& n)
                      {
                          auto neighbor_leaf_output = scalfmm::container::outputs_begin(n->particles());
                          std::get<0>(*neighbor_leaf_output) += static_cast<value_type>(target_leaf.size());
                          nb_val += static_cast<value_type>(n->size());
                      });
        std::get<0>(*scalfmm::container::outputs_begin(target_leaf.particles())) += nb_val;
        //       infos.call_count++;
    }

    /**
     * @brief
     *
     * @tparam LeafType
     */
    template<typename LeafType>
    inline void p2p_inner(count_matrix_kernel const& /*mat*/, LeafType& target_leaf,
                          [[maybe_unused]] const bool mutual = true)
    {
        using value_type = typename LeafType::value_type;
        //        auto& infos = mat.get("p2p_inner");
        std::get<0>(*scalfmm::container::outputs_begin(target_leaf.particles())) +=
          static_cast<value_type>(target_leaf.size());
        //       infos.call_count++;
    }

    /**
     * @brief
     *
     * @tparam LeafType
     * @tparam ContainerOfLeafIteratorType
     */
    template<typename LeafType, typename ContainerOfLeafIteratorType>
    inline void p2p_remote(count_matrix_kernel const&, LeafType&, ContainerOfLeafIteratorType const&)
    {
    }
}   // namespace count_kernel

#endif   // SCALFMM_CORE_COUNT_KERNEL_HPP
