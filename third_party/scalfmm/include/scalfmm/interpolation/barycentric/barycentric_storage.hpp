
// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/barycentric/barycentric_storage.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_BARYCENTRIC_BARYCENTRIC_STORAGE_HPP
#define SCALFMM_INTERPOLATION_BARYCENTRIC_BARYCENTRIC_STORAGE_HPP

#include "scalfmm/memory/storage.hpp"

namespace scalfmm::component
{
    /**
     * @brief
     *
     * This is the specialization for the barycentric interpolator that
     * requires to store the transformed multipoles.
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam Inputs
     * @tparam Outputs
     */
    template<typename ValueType, std::size_t Dimension, std::size_t Inputs, std::size_t Outputs>
    struct alignas(XTENSOR_FIXED_ALIGN) barycentric_storage
      : public memory::aggregate_storage<memory::multipoles_storage<ValueType, Dimension, Inputs>,
                                         memory::locals_storage<ValueType, Dimension, Outputs>>
    {
        /**
         * @brief
         *
         */
        struct empty
        {
        };

        /**
         * @brief
         *
         */
        static constexpr std::size_t dimension = Dimension;

        /**
         * @brief
         *
         */
        static constexpr std::size_t inputs_size = Inputs;

        /**
         * @brief
         *
         */
        static constexpr std::size_t outputs_size = Outputs;

        using value_type = ValueType;

        using multipoles_storage_type = memory::multipoles_storage<ValueType, Dimension, Inputs>;
        using locals_storage_type = memory::locals_storage<ValueType, Dimension, Outputs>;

        using base_type = memory::aggregate_storage<multipoles_storage_type, locals_storage_type>;

        using multipoles_container_type = typename memory::storage_traits<multipoles_storage_type>::tensor_type;
        using locals_container_type = typename memory::storage_traits<locals_storage_type>::tensor_type;
        using buffer_value_type = empty;
        using buffer_inner_type = empty;
        using buffer_shape_type = empty;
        using buffer_type = empty;

        using base_type::base_type;

        /**
         * @brief Construct a new barycentric storage object
         *
         */
        barycentric_storage() = default;

        /**
         * @brief Construct a new barycentric storage object
         *
         */
        barycentric_storage(barycentric_storage const&) = default;

        /**
         * @brief Construct a new barycentric storage object
         *
         */
        barycentric_storage(barycentric_storage&&) noexcept = default;

        /**
         * @brief
         *
         * @return barycentric_storage&
         */
        inline auto operator=(barycentric_storage const&) -> barycentric_storage& = default;

        /**
         * @brief
         *
         * @return barycentric_storage&
         */
        inline auto operator=(barycentric_storage&&) noexcept -> barycentric_storage& = default;

        /**
         * @brief Destroy the barycentric storage object
         *
         */
        ~barycentric_storage() = default;
    };
}   // namespace scalfmm::component

#endif   // SCALFMM_INTERPOLATION_BARYCENTRIC_BARYCENTRIC_STORAGE_HPP
