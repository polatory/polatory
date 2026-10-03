// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/interpolation/m2p.hpp
// --------------------------------
#pragma once

#include "scalfmm/container/access.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/container/variadic_adaptor.hpp"
#include "scalfmm/interpolation/interpolator.hpp"
#include "scalfmm/interpolation/mapping.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/operators/interpolation/utils.hpp"
#include "scalfmm/simd/memory.hpp"
#include "scalfmm/utils/math.hpp"

#include "xsimd/xsimd.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

namespace scalfmm::operators
{
    // -------------------------------------------------------------
    // M2P operator
    // -------------------------------------------------------------

    /**
     * @brief
     *
     * @tparam FarFieldType
     * @tparam InterpolatorType
     * @tparam CellType
     * @tparam LeafType
     * @param far_field
     * @param source_cell
     * @param target_leaf
     * @return std::enable_if_t<std::is_base_of_v<interpolation::impl::interpolator<InterpolatorType>, InterpolatorType>>
     */
    template<template<typename, bool> typename FarFieldType, typename InterpolatorType, typename CellType,
             typename LeafType>
    inline auto apply_m2p(FarFieldType<InterpolatorType, false> const& far_field, CellType const& source_cell,
                          LeafType& target_leaf)
      -> std::enable_if_t<std::is_base_of_v<interpolation::impl::interpolator<InterpolatorType>, InterpolatorType>>
    {
        // set-up (aliases and constants)
        using leaf_type = LeafType;
        using interpolator_type = InterpolatorType;
        using matrix_kernel_type = typename interpolator_type::matrix_kernel_type;
        using value_type = typename leaf_type::value_type;
        using position_type = typename leaf_type::position_type;
        using particle_type = typename leaf_type::particle_type;
        using outputs_type = typename particle_type::outputs_type;
        using mapping_type = interpolation::map_loc_glob<position_type>;

        static constexpr std::size_t dimension = position_type::dimension;
        static constexpr std::size_t inputs_size = matrix_kernel_type::km;
        static constexpr std::size_t outputs_size = matrix_kernel_type::kn;

        using simd_type = xsimd::simd_type<value_type>;
        using simd_position_type = container::point<simd_type, position_type::dimension>;
        using simd_outputs_type = std::array<simd_type, particle_type::outputs_size>;

        static constexpr std::size_t inc = simd_type::size;
        const std::size_t leaf_size{target_leaf.size()};
        const std::size_t vec_size = leaf_size - leaf_size % inc;

        // get approximation data
        auto const& interp = far_field.approximation();
        const auto order = interp.order();
        const auto& matrix_kernel = interp.matrix_kernel();

        // mapping operator in scalar
        const mapping_type mapping_part_position_scal(source_cell.center(), position_type(source_cell.width()));

        // iterator on the particle container -> std::tuple<iterators...>
        auto position_iterator = container::position_begin(target_leaf.particles());
        auto outputs_iterator = container::outputs_begin(target_leaf.particles());

        const std::size_t nnodes = math::pow(order, position_type::dimension);

        std::vector<position_type> root_grid(nnodes);
        auto roots = interp.roots();

        std::array<std::size_t, position_type::dimension> stops{};
        stops.fill(order);

        // generate grid of roots
        std::size_t idx = 0;

        // lambda function for assembling S
        auto construct_root_grid = [&roots, &root_grid, &mapping_part_position_scal, &idx](auto... current_indices)
        {
            auto local_root_position = position_type(utils::generate_root<dimension>(roots, {{current_indices...}}));
            mapping_part_position_scal(local_root_position, root_grid[idx]);
            idx++;
        };

        // expand loops over ORDER
        meta::looper<dimension>()(construct_root_grid, stops);

        // Loop through the particles (vector)
        for(std::size_t part = 0; part < vec_size; part += inc)
        {
            auto global_position = simd::load_position<simd_position_type>(position_iterator, simd::unaligned{});

            simd_outputs_type outputs{meta::to_array(simd::load_tuple<simd_type>(outputs_iterator, simd::unaligned{}))};

            auto multipoles_iterator = source_cell.multipoles_begin();

            for(std::size_t m = 0; m < nnodes; ++m)
            {
                auto kxy = matrix_kernel.evaluate(global_position, simd_position_type(root_grid[m]));

                for(std::size_t i = 0; i < outputs_size; ++i)
                {
                    for(std::size_t j = 0; j < inputs_size; ++j)
                    {
                        outputs.at(i) += kxy.at(i * inputs_size + j) * (*multipoles_iterator[j]);
                    }
                }

                meta::repeat([](auto& it) { ++it; }, multipoles_iterator);
            }

            simd::store_tuple<simd_type>(outputs_iterator, outputs, simd::unaligned{});

            position_iterator += inc;
            outputs_iterator += inc;
        }

        // Loop through the particles (scalar)
        for(std::size_t part = vec_size; part < leaf_size; ++part)
        {
            auto global_position = simd::load_position<position_type>(position_iterator);

            outputs_type outputs{};
            meta::for_each(outputs, [](auto& v) { v = value_type(0.); });

            auto multipoles_iterator = source_cell.multipoles_begin();

            for(std::size_t m = 0; m < nnodes; ++m)
            {
                auto kxy = matrix_kernel.evaluate(global_position, root_grid[m]);

                for(std::size_t i = 0; i < outputs_size; ++i)
                {
                    for(std::size_t j = 0; j < inputs_size; ++j)
                    {
                        outputs.at(i) += kxy.at(i * inputs_size + j) * (*multipoles_iterator[j]);
                    }
                }

                meta::repeat([](auto& it) { ++it; }, multipoles_iterator);
            }

            meta::tuple_sum_update(*outputs_iterator, outputs);

            ++position_iterator;
            ++outputs_iterator;
        }
    }
}   // namespace scalfmm::operators
