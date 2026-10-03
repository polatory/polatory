// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/grid_storage.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_GRID_STORAGE_HPP
#define SCALFMM_INTERPOLATION_GRID_STORAGE_HPP

#include "scalfmm/memory/storage.hpp"

namespace scalfmm::component
{
    // This is the specialization for the uniform interpolator that
    // requires to store the transformed multipoles.
    template<typename ValueType, std::size_t Dimension, std::size_t Inputs, std::size_t Outputs>
    struct alignas(XTENSOR_FIXED_ALIGN) grid_storage
      : public memory::aggregate_storage<memory::multipoles_storage<ValueType, Dimension, Inputs>,
                                         memory::locals_storage<ValueType, Dimension, Outputs>>
    //  ,
    //  memory::transformed_multipoles_storage<ValueType, Dimension, 2>>
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
        using transfer_multipole_type = value_type;

        using multipoles_storage_type = memory::multipoles_storage<ValueType, Dimension, Inputs>;
        using locals_storage_type = memory::locals_storage<ValueType, Dimension, Outputs>;
        // using buffer_storage_type = memory::tensor_storage<ValueType, Dimension, 2>;

        using base_type = memory::aggregate_storage<multipoles_storage_type, locals_storage_type>;

        using multipoles_container_type = typename memory::storage_traits<multipoles_storage_type>::tensor_type;
        using locals_container_type = typename memory::storage_traits<locals_storage_type>::tensor_type;

        using base_type::base_type;
        // more using in memory

        /**
         * @brief Construct a new grid storage object
         *
         */
        grid_storage() = default;

        /**
         * @brief Construct a new grid storage object
         *
         */
        grid_storage(grid_storage const&) = default;

        /**
         * @brief Construct a new grid storage object
         *
         */
        grid_storage(grid_storage&&) noexcept = default;

        /**
         * @brief
         *
         * @return grid_storage&
         */
        inline auto operator=(grid_storage const&) -> grid_storage& = default;

        /**
         * @brief
         *
         * @return grid_storage&
         */
        inline auto operator=(grid_storage&&) noexcept -> grid_storage& = default;

        /**
         * @brief Destroy the grid storage object
         *
         */
        ~grid_storage() = default;

        /**
         * @brief Get the transfer nultipole object
         *
         * @return auto&
         */
        auto& get_transfer_multipole() { base_type::get(); }

        /**
         * @brief Get the transfer nultipole object
         *
         * @return auto const&
         */
        auto const& get_transfer_multipole() const { base_type::get(); }

        /**
         * @brief
         *
         * @return multipoles_container_type const&
         */
        auto transfer_multipoles() const noexcept -> multipoles_container_type const&
        {
            return base_type::multipoles();
        }

        /**
         * @brief
         *
         * @return multipoles_container_type&
         */
        auto transfer_multipoles() noexcept -> multipoles_container_type& { return base_type::multipoles(); }
    };
}   // namespace scalfmm::component
#endif   // SCALFMM_INTERPOLATION_GRID_STORAGE_HPP
