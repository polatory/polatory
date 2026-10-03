// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/interpolation/p2l.hpp
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
    // L2P operator
    // -------------------------------------------------------------

    /**
     * @brief
     *
     * @tparam FarFieldType
     * @tparam InterpolatorType
     * @tparam CellType
     * @tparam LeafType
     * @param far_field
     * @param target_cell
     * @param source_leaf
     * @return std::enable_if_t<std::is_base_of_v<interpolation::impl::interpolator<InterpolatorType>, InterpolatorType>>
     */
    template<template<typename, bool> typename FarFieldType, typename InterpolatorType, typename CellType,
             typename LeafType>
    inline auto apply_p2l(FarFieldType<InterpolatorType, false> const& far_field, CellType& target_cell,
                          LeafType const& source_leaf)
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
        using simd_position_type = container::point<simd_type, dimension>;
        using simd_locals_type = std::array<simd_type, outputs_size>;

        static constexpr std::size_t inc = simd_type::size;
        const std::size_t leaf_size{source_leaf.size()};
        // const std::size_t vec_size = leaf_size - leaf_size % inc;
        const std::size_t vec_size = 0;

        // get approximation data
        auto const& interp = far_field.approximation();
        const auto order = interp.order();
        const auto& matrix_kernel = interp.matrix_kernel();

        // mapping operator in scalar
        const mapping_type mapping_part_position_scal(target_cell.center(), position_type(target_cell.width()));

        // iterator on the particle container -> std::tuple<iterators...>
        auto position_iterator = container::position_begin(source_leaf.particles());
        auto inputs_iterator = container::inputs_begin(source_leaf.particles());

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
        // for(std::size_t part = 0; part < vec_size; part += inc)
        // {
        //     auto global_position = simd::load_position<simd_position_type>(position_iterator, simd::unaligned{});
        //     auto inputs_values = meta::to_array(simd::load_tuple<simd_type>(inputs_iterator, simd::unaligned{}));

        //     auto locals_iterator = target_cell.locals_begin();

        //     simd_locals_type simd_locals{};
        //     meta::for_each(simd_locals, [](auto& v) { v = value_type(0.); });

        //     for(std::size_t m = 0; m < nnodes; ++m)
        //     {
        //         auto kxy = matrix_kernel.evaluate(simd_position_type(root_grid[m]), global_position);

        //         for(std::size_t i = 0; i < outputs_size; ++i)
        //         {
        //             for(std::size_t j = 0; j < inputs_size; ++j)
        //             {
        //                 simd_locals[i] += kxy.at(i * inputs_size + j) * inputs_values[j];
        //             }
        //         }

        //         auto reduced_locals = meta::apply(simd_locals, [](auto f) { return xsimd::reduce_add(f); });
        //         meta::it_sum_update(locals_iterator, reduced_locals);
        //         meta::repeat([](auto& it) { ++it; }, locals_iterator);
        //     }

        //     position_iterator += inc;
        //     inputs_iterator += inc;
        // }

        // Loop through the particles (scalar)
        for(std::size_t part = vec_size; part < leaf_size; ++part)
        {
            auto global_position = simd::load_position<position_type>(position_iterator);
            auto inputs_values = meta::to_array(simd::load_tuple<value_type>(inputs_iterator));

            auto locals_iterator = target_cell.locals_begin();

            for(std::size_t m = 0; m < nnodes; ++m)
            {
                auto kxy = matrix_kernel.evaluate(root_grid[m], global_position);

                for(std::size_t i = 0; i < outputs_size; ++i)
                {
                    for(std::size_t j = 0; j < inputs_size; ++j)
                    {
                        *locals_iterator[i] += kxy.at(i * inputs_size + j) * inputs_values[j];
                    }
                }

                meta::repeat([](auto& it) { ++it; }, locals_iterator);
            }

            ++position_iterator;
            ++inputs_iterator;
        }
    }
}   // namespace scalfmm::operators
