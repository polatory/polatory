// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/count_kernel/count_kernel.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_COUNT_KERNEL_COUNT_KERNEL_HPP
#define SCALFMM_OPERATORS_COUNT_KERNEL_COUNT_KERNEL_HPP

#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/interpolation/permutations.hpp"
#include "scalfmm/matrix_kernels/mk_common.hpp"
#include "scalfmm/meta/utils.hpp"

#include <cpp_tools/colors/colorized.hpp>

namespace count_kernels
{
    namespace particles
    {
        /**
         * @brief
         *
         */
        struct empty_shape
        {
        };

        /**
         * @brief
         *
         */
        struct empty_inner
        {
            /**
             * @brief Construct a new empty inner object
             *
             * @tparam S
             */
            template<typename S>
            empty_inner(S, double){};
        };

        /**
         * @brief
         *
         */
        struct empty
        {
            /**
             * @brief Construct a new empty object
             *
             */
            empty() = default;

            /**
             * @brief Construct a new empty object
             *
             */
            empty(empty_shape, empty_inner) {};
        };

        /**
         * @brief The count matrix kernel (for compatibility).
         *
         */
        struct count_matrix_kernel
        {
            /**
             * @brief
             *
             */
            static constexpr std::size_t kn{1};

            /**
             * @brief
             *
             */
            static constexpr std::size_t km{1};

            /**
             * @brief
             *
             */
            static constexpr int separation_criterion{1};
            const std::string name() const { return std::string("count kernel "); }
        };

        /**
         * @brief An implementation of the near field for the count kernel.
         *
         */
        struct count_near_field
        {
            using matrix_kernel_type = count_matrix_kernel;

            /**
             * @brief
             *
             */
            count_matrix_kernel _mat;

            /**
             * @brief
             *
             */
            bool _mutual;

            /**
             * @brief
             *
             * @return int
             */
            inline auto separation_criterion() const -> int { return matrix_kernel_type::separation_criterion; }

            /**
             * @brief
             *
             * @return count_matrix_kernel
             */
            inline count_matrix_kernel matrix_kernel() const { return _mat; };

            /**
             * @brief
             *
             * @return true
             * @return false
             */
            inline auto mutual() const -> bool { return _mutual; }

            /**
             * @brief Construct a new count near field object
             *
             * @param mutual
             */
            count_near_field(const bool mutual)
              : _mutual(mutual)
            {
            }
        };

        /**
         * @brief
         *
         * @tparam Dimension
         */
        template<std::size_t Dimension = 3>
        struct count_approximation
        {
            using value_type = double;
            using matrix_kernel_type = count_matrix_kernel;
            using storage_type = scalfmm::component::grid_storage<value_type, Dimension, 1, 1>;

            using buffer_type = empty;
            using buffer_shape_type = empty_shape;
            using buffer_inner_type = empty_inner;

            /**
             * @brief
             *
             */
            static constexpr std::size_t dimension{Dimension};

            /**
             * @brief
             *
             */
            count_matrix_kernel mat;

            /**
             * @brief
             *
             * @return count_matrix_kernel const&
             */
            inline auto matrix_kernel() const noexcept -> count_matrix_kernel const& { return mat; }

            /**
             * @brief
             *
             * @return auto
             */
            inline auto cell_width_extension() const noexcept { return 0.; }

            /**
             * @brief
             *
             * @tparam CellType
             */
            template<typename CellType>
            void apply_multipoles_preprocessing(CellType& /*current_cell*/, std::size_t t = 0) const
            {
            }

            /**
             * @brief
             *
             * @tparam CellType
             */
            template<typename CellType>
            void apply_multipoles_postprocessing(CellType& /*current_cell*/, buffer_type, std::size_t t = 0) const
            {
            }

            /**
             * @brief
             *
             * @return buffer_type
             */
            inline auto buffer_initialization() const -> buffer_type { return empty{}; }

            /**
             * @brief
             *
             * @return buffer_shape_type
             */
            inline auto buffer_shape() const -> buffer_shape_type { return empty_shape{}; }

            /**
             * @brief
             *
             */
            inline auto buffer_reset(buffer_type) const -> void {}
        };

        /**
         * @brief An implementation of the far field for the count kernel.
         *
         * @tparam Dimension
         */
        template<std::size_t Dimension = 3>
        struct count_far_field
        {
            using approximation_type = count_approximation<Dimension>;

          private:
            /**
             * @brief
             *
             */
            count_approximation<Dimension> m_count_approximation;

          public:
            /**
             * @brief
             *
             */
            static constexpr bool compute_gradient = false;

            /**
             * @brief
             *
             * @return approximation_type const&
             */
            auto approximation() const -> approximation_type const& { return m_count_approximation; }

            /**
             * @brief
             *
             * @return int
             */
            inline auto separation_criterion() const noexcept -> int { return 1; }
        };

        /**
         * @brief
         *
         * @tparam LeafType
         * @tparam CellType
         * @tparam Dimension
         */
        template<typename LeafType, typename CellType, std::size_t Dimension>
        inline auto p2m(count_far_field<Dimension> const& /*interp*/, LeafType const& source_leaf,
                        CellType& target_cell) -> void
        {
            auto& multipoles = target_cell.multipoles().at(0);
            multipoles += source_leaf.size();
        }

        /**
         * @brief
         *
         * @tparam CellType
         * @tparam Dimension
         */
        template<typename CellType, std::size_t Dimension>
        inline auto l2l(count_approximation<Dimension> const& /*interp*/, CellType const& parent_cell,
                        std::size_t /*child_index*/, CellType& child_cell, std::size_t tree_level = 2) -> void
        {
            auto& child_locals = child_cell.locals().at(0);
            auto const& current_locals = parent_cell.clocals().at(0);
            child_locals += current_locals;
        }

        /**
         * @brief
         *
         * @tparam CellType
         * @tparam Dimension
         */
        template<typename CellType, std::size_t Dimension>
        inline auto m2m(count_approximation<Dimension> const& /*interp*/, CellType const& child_cell,
                        std::size_t /*child_index*/, CellType& parent_cell, std::size_t tree_level = 2) -> void
        {
            auto& parent_multipoles = parent_cell.multipoles().at(0);
            auto const& child_multipoles = child_cell.cmultipoles().at(0);
            parent_multipoles += child_multipoles;
        }

        /**
         * @brief
         *
         * @tparam CellType
         * @tparam Dimension
         */
        template<typename CellType, std::size_t Dimension>
        inline auto m2l(count_approximation<Dimension> const& /*interp*/, CellType const& source_cell,
                        std::size_t /*neighbor_idx*/, CellType& target_cell, std::size_t /*tree_level*/,
                        typename count_approximation<Dimension>::buffer_type& /*buffer*/) -> void
        {
            auto const& source_multipoles = source_cell.cmultipoles().at(0);
            auto& target_locals = target_cell.locals().at(0);
            target_locals += source_multipoles;
        }

        /**
         * @brief
         *
         * @tparam CellType
         * @tparam Dimension
         */
        template<typename CellType, std::size_t Dimension>
        inline auto m2l_loop(count_approximation<Dimension> const& /*interp*/, CellType& target_cell,
                             std::size_t /*tree_level*/,
                             typename count_approximation<Dimension>::buffer_type& /*buffer*/) -> void
        {
            auto const& target_symb = target_cell.csymbolics();
            auto const& interaction_iterators = target_symb.interaction_iterators;
            auto& target_locals = target_cell.locals().at(0);
            for(std::size_t index{0}; index < target_symb.existing_neighbors; ++index)
            {
                auto const& source_cell = *interaction_iterators.at(index);
                auto const& source_multipoles = source_cell.cmultipoles().at(0);
                target_locals += source_multipoles;
            }
        }

        /**
         * @brief
         *
         * @tparam CellType
         * @tparam LeafType
         * @tparam Dimension
         */
        template<typename CellType, typename LeafType, std::size_t Dimension>
        inline auto l2p(count_far_field<Dimension> const& /*fmm_operator*/, CellType const& source_cell,
                        LeafType& target_leaf) -> void
        {
            auto p = typename LeafType::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += *std::begin(source_cell.clocals().at(0));
        }

        /**
         * @brief
         *
         * @tparam LeafType
         * @tparam ContainerOfLeafIteratorType
         * @tparam ArrayType
         * @tparam ValueType
         */
        template<typename LeafType, typename ContainerOfLeafIteratorType, typename ArrayType, typename ValueType>
        inline auto p2p_full_mutual(count_matrix_kernel const& /*mat*/, LeafType& target_leaf,
                                    ContainerOfLeafIteratorType const& neighbors, const int& size, ArrayType const&,
                                    ValueType const&) -> void
        {
            using value_type = typename LeafType::value_type;
            value_type nb_val{0};
            std::for_each(std::begin(neighbors), std::begin(neighbors) + size,
                          [&nb_val, &target_leaf](auto& leaf_r)
                          {
                              nb_val += static_cast<value_type>(leaf_r->size());
                              auto p_r = typename LeafType::proxy_type(*(leaf_r->begin()));
                              p_r.outputs()[0] += target_leaf.size();
                          });

            auto p = typename LeafType::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += nb_val;
        }

        /**
         * @brief
         *
         * @tparam MutualComputation
         * @tparam LEAF_T
         * @param mat
         * @param target_leaf
         * @param mutual
         */
        template<bool MutualComputation = false, typename LEAF_T>
        inline auto p2p_inner([[maybe_unused]] count_matrix_kernel const& mat, LEAF_T& target_leaf,
                              [[maybe_unused]] const bool mutual = true) -> void
        {
            using value_type = typename LEAF_T::value_type;

            auto p = typename LEAF_T::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += static_cast<value_type>(target_leaf.size());
        }

        /**
         * @brief
         *
         * @tparam LeafType
         * @tparam ContainerOfLeafIteratorType
         * @tparam ArrayType
         * @tparam ValueType
         * @param mat
         * @param target_leaf
         * @param neighbors
         */
        template<typename LeafType, typename ContainerOfLeafIteratorType, typename ArrayType, typename ValueType>
        inline auto p2p_outer([[maybe_unused]] count_matrix_kernel const& mat, LeafType& target_leaf,
                              ContainerOfLeafIteratorType const& neighbors, ArrayType const&, ValueType const&) -> void
        {
            using value_type = typename LeafType::value_type;
            value_type nb_val{0};
            std::for_each(std::begin(neighbors), std::begin(neighbors) + target_leaf.csymbolics().number_of_neighbors,
                          [&nb_val](auto const& leaf)
                          {
                              {
                                  nb_val += static_cast<value_type>(leaf->size());
                              }
                          });
            auto p = typename LeafType::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += nb_val;
        }

        /**
         * @brief
         *
         * @tparam LeafType
         * @tparam ContainerOfLeafIteratorType
         * @tparam ArrayType
         * @tparam ValueType
         * @param mat
         * @param target_leaf
         * @param neighbors
         * @param number_of_neighbors
         */
        template<typename LeafType, typename ContainerOfLeafIteratorType, typename ArrayType, typename ValueType>
        inline auto p2p_outer([[maybe_unused]] count_matrix_kernel const& mat, LeafType& target_leaf,
                              ContainerOfLeafIteratorType const& neighbors, const int& number_of_neighbors,
                              ArrayType const&, ValueType const&) -> void
        {
            using value_type = typename LeafType::value_type;
            value_type nb_val{0};
            std::for_each(std::begin(neighbors), std::begin(neighbors) + number_of_neighbors,
                          [&nb_val](auto const& leaf)
                          {
                              {
                                  nb_val += static_cast<value_type>(leaf->size());
                              }
                          });
            auto p = typename LeafType::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += nb_val;
        }

        /**
         * @brief
         *
         * @tparam TargetLeafType
         * @tparam SourceLeafType
         * @tparam ArrayType
         * @param target_leaf
         * @param source_leaf
         * @param shift
         */
        template<typename TargetLeafType, typename SourceLeafType, typename ArrayType>
        inline auto p2p_outer(count_matrix_kernel const&, TargetLeafType& target_leaf,
                              SourceLeafType const& source_leaf, [[maybe_unused]] ArrayType const& shift) -> void
        {
            using value_type = typename TargetLeafType::value_type;
            auto p = typename TargetLeafType::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += static_cast<value_type>(source_leaf.size());
        }
    }   // namespace particles
}   // namespace count_kernels

namespace scalfmm::interpolation
{

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct interpolator_traits;

    /**
     * @brief
     *
     * @tparam Dimension
     */
    template<std::size_t Dimension>
    struct interpolator_traits<count_kernels::particles::count_approximation<Dimension>>
    {
        using matrix_kernel_type = count_kernels::particles::count_matrix_kernel;
        static constexpr std::size_t dimension = Dimension;
        static constexpr std::size_t kn = matrix_kernel_type::kn;
        static constexpr std::size_t km = matrix_kernel_type::km;
        static constexpr bool enable_symmetries = (dimension < 4) ? true : false;
        using storage_type = scalfmm::component::grid_storage<double, dimension, km, kn>;
        using buffer_type = count_kernels::particles::empty;
        using buffer_shape_type = count_kernels::particles::empty_shape;
        using buffer_inner_type = count_kernels::particles::empty_inner;
        using multipoles_container_type = typename storage_type::multipoles_container_type;
        using locals_container_type = typename storage_type::locals_container_type;
    };
}   // namespace scalfmm::interpolation

#endif   // SCALFMM_CORE_COUNT_KERNEL_HPP
