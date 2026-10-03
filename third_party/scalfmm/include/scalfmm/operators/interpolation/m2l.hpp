// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/interpolation/m2l.hpp
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
    // naive M2L operator
    // -------------------------------------------------------------

    /**
     * @brief
     *
     * @tparam ApproximationType
     * @tparam CellType
     * @param interp
     * @param source_cell
     * @param target_cell
     */
    template<typename ApproximationType, typename CellType>
    inline auto apply_naive_m2l(ApproximationType const& interp, CellType const& source_cell,
                                CellType& target_cell) -> void
    {
        using interpolator_type = ApproximationType;
        using cell_type = CellType;
        using matrix_kernel_type = typename interpolator_type::matrix_kernel_type;
        using value_type = typename cell_type::value_type;
        using position_type = typename cell_type::position_type;
        using mapping_type = interpolation::map_loc_glob<position_type>;

        static constexpr std::size_t dimension = position_type::dimension;
        static constexpr std::size_t inputs_size = matrix_kernel_type::km;
        static constexpr std::size_t outputs_size = matrix_kernel_type::kn;

        // get approximation data
        const auto order = interp.order();
        const auto& matrix_kernel = interp.matrix_kernel();

        // mapping operator in scalar
        const mapping_type source_mapping(source_cell.center(), position_type(source_cell.width()));
        const mapping_type target_mapping(target_cell.center(), position_type(target_cell.width()));

        const std::size_t nnodes = math::pow(order, position_type::dimension);

        std::vector<position_type> source_root_grid(nnodes), target_root_grid(nnodes);
        auto roots = interp.roots();

        std::array<std::size_t, position_type::dimension> stops{};
        stops.fill(order);

        // generate grid of roots
        std::size_t idx = 0;

        // lambda function for assembling S
        auto construct_root_grids = [&roots, &source_root_grid, &target_root_grid, &source_mapping, &target_mapping,
                                     &idx](auto... current_indices)
        {
            auto local_source_root_position =
              position_type(utils::generate_root<dimension>(roots, {{current_indices...}}));
            source_mapping(local_source_root_position, source_root_grid[idx]);
            auto local_target_root_position =
              position_type(utils::generate_root<dimension>(roots, {{current_indices...}}));
            target_mapping(local_target_root_position, target_root_grid[idx]);
            idx++;
        };

        // expand loops over ORDER
        meta::looper<dimension>()(construct_root_grids, stops);

        auto locals_iterator = target_cell.locals_begin();

        for(std::size_t n = 0; n < nnodes; ++n)
        {
            auto multipoles_iterator = source_cell.multipoles_begin();

            for(std::size_t m = 0; m < nnodes; ++m)
            {
                auto kxy = matrix_kernel.evaluate(target_root_grid[n], source_root_grid[m]);

                for(std::size_t i = 0; i < outputs_size; ++i)
                {
                    for(std::size_t j = 0; j < inputs_size; ++j)
                    {
                        (*locals_iterator[i]) += kxy.at(i * inputs_size + j) * (*multipoles_iterator[j]);
                    }
                }
                meta::repeat([](auto& it) { ++it; }, multipoles_iterator);
            }
            meta::repeat([](auto& it) { ++it; }, locals_iterator);
        }
    }
}   // namespace scalfmm::operators
