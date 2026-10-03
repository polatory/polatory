// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/interpolation/p2m.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_INTERPOLATION_P2M_HPP
#define SCALFMM_OPERATORS_INTERPOLATION_P2M_HPP

#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/interpolator.hpp"
#include "scalfmm/interpolation/mapping.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/interpolation/utils.hpp"
#include "scalfmm/simd/memory.hpp"
#include "scalfmm/utils/math.hpp"

#include "xsimd/xsimd.hpp"

#include <array>
#include <cstdlib>
#include <tuple>
#include <vector>

namespace scalfmm::operators
{
    // -------------------------------------------------------------
    // P2M operator
    // -------------------------------------------------------------

    /**
     * @brief
     *
     * @tparam FarFieldOperatorType
     * @tparam CellType
     * @tparam LeafType
     * @param far_field
     * @param source_leaf
     * @param cell
     */
    template<typename FarFieldOperatorType, typename CellType, typename LeafType>
    inline auto apply_p2m(FarFieldOperatorType const& far_field, LeafType const& source_leaf, CellType& cell) -> void
    {
        // Leaf type
        using leaf_type = LeafType;
        // position_type -> scalfmm::container::point<...>
        using position_type = typename leaf_type::position_type;
        // value_type -> inner position_type::value_type -> underlying scalar_type
        using value_type = typename leaf_type::value_type;
        // simd_type -> xsimd::batch<scalar_type>
        using simd_type = xsimd::simd_type<value_type>;
        // simd_position_type -> container::point<xsimd::batch<scalar_type>>
        using simd_position_type = container::point<simd_type, position_type::dimension>;
        static constexpr auto alignment_tag = simd::unaligned{};

        // extract the generic polynomial function of the interpolator and instantiate it for simd computation
        // using polynomials_f_type = typename Interpolator::template polynomials_f_type<simd_type>;
        auto const& interp = far_field.approximation();

        // simd_vector size -> 8 in double for AVX512 for example
        const std::size_t inc = simd_type::size;
        // number of particles in the leaf
        const std::size_t leaf_size{source_leaf.size()};
        // vectorizable particles
        const std::size_t vec_size = leaf_size - leaf_size % inc;
        const auto order = interp.order();
        // width of the current leaf/cell (taking into account the extension)
        const value_type full_width = source_leaf.width() + interp.cell_width_extension();
        // mapping operator in simd
        const interpolation::map_glob_loc<simd_position_type> mapping_part_position(
          source_leaf.center(), simd_position_type(simd_type(full_width)));

        // iterator on the particle container -> std::tuple<itoerators...>
        auto position_iterator = container::position_begin(source_leaf.cparticles());
        auto inputs_iterator = container::inputs_begin(source_leaf.cparticles());

        std::size_t nnodes = math::pow(order, position_type::dimension);
        // result of polynomial function for the input particle, here in simd
        std::vector<simd_position_type, XTENSOR_DEFAULT_ALLOCATOR(simd_position_type)> poly_of_part(order);
        // Resulting S, also in simd
        std::vector<value_type, XTENSOR_DEFAULT_ALLOCATOR(value_type)> S(nnodes * inc);

        std::array<std::size_t, position_type::dimension> stops{};
        stops.fill(order);

        // here, we process simd_size (i.e. inc) particles at each loop turn
        for(std::size_t part = 0; part < vec_size; part += inc)
        {
            // simd load of particles
            simd_position_type part_position(simd::load_position<simd_position_type>(position_iterator, alignment_tag));
            simd_position_type local_position{};
            // mapping
            mapping_part_position(part_position, local_position);
            //
            for(std::size_t o = 0; o < order; ++o)
            {
                meta::for_each(poly_of_part[o], local_position,
                               [&interp, o](auto x) { return interp.polynomials(x, o); });
            }

            // Assembling S
            std::size_t idx = 0;

            // lambda function for assembling S
            auto construct_s = [&S, &poly_of_part, &idx](auto&... current_indices)
            {
                auto s_simd = utils::generate_s<position_type::dimension>(poly_of_part, {{current_indices...}});
                xsimd::store_aligned(&S[idx], s_simd);
                idx += inc;
            };

            // expand N loop over ORDER
            meta::looper<position_type::dimension>()(construct_s, stops);

            std::size_t vec_idx = 0;
            auto simd_physical_values = simd::load_tuple<simd_type>(inputs_iterator, alignment_tag);
            auto multipoles_iterator = cell.multipoles_begin();

            for(std::size_t m = 0; m < nnodes; ++m)
            {
                auto s_to_apply = xsimd::load_aligned(&S[vec_idx]);

                auto update_potential = [&s_to_apply](auto... pot) { return std::make_tuple(s_to_apply * pot...); };

                auto to_update_multipoles = std::apply(update_potential, simd_physical_values);

                auto reduce_update_multipoles = [](auto... simd_to_reduce)
                { return std::make_tuple(xsimd::reduce_add(simd_to_reduce)...); };

                auto weight_to_apply = std::apply(reduce_update_multipoles, to_update_multipoles);

                meta::it_sum_update(multipoles_iterator, weight_to_apply);
                meta::repeat([](auto& it) { ++it; }, multipoles_iterator);
                vec_idx += inc;
            }

            position_iterator += inc;
            inputs_iterator += inc;
        }

        // mapping operator in scalar
        const interpolation::map_glob_loc<position_type> mapping_part_position_scal(source_leaf.center(),
                                                                                    position_type(full_width));

        // extract the generic polynomial function of the interpolator and intantiate it for simd computation
        // using polynomials_f_type_scal = typename Interpolator::template polynomials_f_type<value_type>;

        // result of polynomial function for the input particle, here in simd
        std::vector<position_type> poly_of_part_scal(order);
        // Resulting S, also in simd
        std::vector<value_type, XTENSOR_DEFAULT_ALLOCATOR(value_type)> S_scal(nnodes);

        // Here is the resulting scalar compution
        for(std::size_t part = vec_size; part < leaf_size; ++part)
        {
            // scalar load of particles
            position_type part_position(simd::load_position<position_type>(position_iterator, alignment_tag));
            position_type local_position{};
            // mapping
            mapping_part_position_scal(part_position, local_position);

            // generate polynomials
            for(std::size_t o = 0; o < order; ++o)
            {
                meta::for_each(poly_of_part_scal[o], local_position,
                               [&interp, o](auto x) { return interp.polynomials(x, o); });
                // poly_of_part_scal[o] = simd::apply_f<position_type::dimension>(call_polynomials, local_position, o);
            }

            // Assembling S
            std::size_t idx = 0;

            auto construct_s = [&S_scal, &poly_of_part_scal, &idx](auto&... current_indices)
            {
                S_scal[idx] = utils::generate_s<position_type::dimension>(poly_of_part_scal, {{current_indices...}});
                idx++;
            };

            meta::looper<position_type::dimension>()(construct_s, stops);

            auto multipoles_iterator = cell.multipoles_begin();

            for(std::size_t m = 0; m < nnodes; ++m)
            {
                auto s_to_apply = S_scal[m];
                auto update_potential = [&s_to_apply](auto... pot) { return std::make_tuple(s_to_apply * pot...); };

                auto weight_to_apply = std::apply(update_potential, *inputs_iterator);

                meta::it_sum_update(multipoles_iterator, weight_to_apply);
                meta::repeat([](auto& it) { ++it; }, multipoles_iterator);
            }

            ++position_iterator;
            ++inputs_iterator;
        }
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_INTERPOLATION_P2M_HPP
