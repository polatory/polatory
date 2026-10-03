// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/uniform/uniform_storage.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_UNIFORM_UNIFORM_STORAGE_HPP
#define SCALFMM_INTERPOLATION_UNIFORM_UNIFORM_STORAGE_HPP

#include "scalfmm/container/variadic_adaptor.hpp"
#include "scalfmm/memory/storage.hpp"

#include <complex>

namespace scalfmm::component
{
    /**
     * @brief
     *
     * This is the specialization for the uniform interpolator that
     * requires to store the transformed multipoles.
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam Inputs
     * @tparam Outputs
     */
    template<typename ValueType, std::size_t Dimension, std::size_t Inputs, std::size_t Outputs>
    struct alignas(XTENSOR_FIXED_ALIGN) uniform_fft_storage
      : public memory::aggregate_storage<
          memory::multipoles_storage<ValueType, Dimension, Inputs>,
          memory::locals_storage<ValueType, Dimension, Outputs>,
          memory::transformed_multipoles_storage<std::complex<ValueType>, Dimension, Inputs>>
    {
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
        using transfer_multipole_type = std::complex<ValueType>;

        using multipoles_storage_type = memory::multipoles_storage<ValueType, Dimension, Inputs>;
        using locals_storage_type = memory::locals_storage<ValueType, Dimension, Outputs>;
        using transformed_storage_type =
          memory::transformed_multipoles_storage<transfer_multipole_type, Dimension, Inputs>;
        using base_type =
          memory::aggregate_storage<multipoles_storage_type, locals_storage_type, transformed_storage_type>;
        using multipoles_container_type = typename memory::storage_traits<multipoles_storage_type>::tensor_type;
        using locals_container_type = typename memory::storage_traits<locals_storage_type>::tensor_type;
        using transformed_container_type = typename memory::storage_traits<transformed_storage_type>::tensor_type;
        using buffer_storage_type = memory::transformed_multipoles_storage<transfer_multipole_type, Dimension, Outputs>;
        using buffer_value_type = typename memory::storage_traits<buffer_storage_type>::value_type;
        using buffer_inner_type = typename memory::storage_traits<buffer_storage_type>::inner_type;
        using buffer_shape_type = typename memory::storage_traits<buffer_storage_type>::outer_shape;
        using buffer_type = typename memory::storage_traits<buffer_storage_type>::tensor_type;

        using base_type::base_type;

        /**
         * @brief Construct a new uniform fft storage object
         *
         */
        uniform_fft_storage() = default;

        /**
         * @brief Construct a new uniform fft storage object
         *
         */
        uniform_fft_storage(uniform_fft_storage const&) = default;

        /**
         * @brief Construct a new uniform fft storage object
         *
         */
        uniform_fft_storage(uniform_fft_storage&&) noexcept = default;

        /**
         * @brief
         *
         * @return uniform_fft_storage&
         */
        inline auto operator=(uniform_fft_storage const&) -> uniform_fft_storage& = default;

        /**
         * @brief
         *
         * @return uniform_fft_storage&
         */
        inline auto operator=(uniform_fft_storage&&) noexcept -> uniform_fft_storage& = default;

        /**
         * @brief Destroy the uniform fft storage object
         *
         */
        ~uniform_fft_storage() = default;

        /**
         * @brief
         *
         * @return transformed_container_type const&
         */
        auto transfer_multipoles() const noexcept -> transformed_container_type const&
        {
            return base_type::transformed_multipoles();
        }

        /**
         * @brief
         *
         * @return transformed_container_type&
         */
        auto transfer_multipoles() noexcept -> transformed_container_type&
        {
            return base_type::transformed_multipoles();
        }
    };
}   // namespace scalfmm::component
#endif   // SCALFMM_INTERPOLATION_UNIFORM_UNIFORM_STORAGE_HPP
