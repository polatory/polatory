// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/interpolation/l2p.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_INTERPOLATION_L2P_HPP
#define SCALFMM_OPERATORS_INTERPOLATION_L2P_HPP

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
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"
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
    // l2p operators
    // -------------------------------------------------------------

    /**
     * @brief apply l2p when only the "potential" is evaluated
     *
     * @tparam FarFieldType Far field type  format is FarFieldType<InterpolatorType, false>
     * @tparam InterpolatorType  Interpolator type used in far field
     * @tparam CellType Cell type
     * @tparam LeafType Leaf type
     * @param far_field   the  far-field operator
     * @param source_cell the cell containing the local
     * @param target_leaf the leaf containing the particles
     */
    template<template<typename, bool> typename FarFieldType, typename InterpolatorType, typename CellType,
             typename LeafType>
    inline auto apply_l2p(FarFieldType<InterpolatorType, false> const& far_field, CellType const& source_cell,
                          LeafType& target_leaf)
      -> std::enable_if_t<std::is_base_of_v<interpolation::impl::interpolator<InterpolatorType>, InterpolatorType>>
    {
        // Leaf type
        using leaf_type = LeafType;
        using particle_type = typename leaf_type::particle_type;
        using position_type = typename leaf_type::position_type;
        using outputs_type = typename particle_type::outputs_type;
        using value_type = typename leaf_type::value_type;
        using simd_outputs_type =
          decltype(meta::to_tuple(meta::replace_inner_tuple_type_t<xsimd::simd_type, outputs_type>{}));
        // position_type -> scalfmm::container::point<...>
        using position_type = typename leaf_type::position_type;
        static constexpr std::size_t dimension = position_type::dimension;
        // output_type -> std::tuple<...>
        // value_type -> inner position_type::value_type -> underlying scalar_type
        using value_type = typename LeafType::value_type;
        // simd_type -> xsimd::batch<scalar_type>
        using simd_type = xsimd::simd_type<value_type>;
        // simd_position_type -> container::point<xsimd::batch<scalar_type>>
        using simd_position_type = container::point<simd_type, dimension>;
        // simd_vector size -> 8 in double for AVX512 for example
        const std::size_t inc = simd_type::size;
        //
        // number of particles in the leaf
        const std::size_t leaf_size{target_leaf.size()};

        // vectorizable particles
        const std::size_t vec_size = leaf_size - leaf_size % inc;
        // const std::size_t vec_size = 0;   // leaf_size - leaf_size % inc;

        static constexpr auto alignment_tag = simd::unaligned{};

        auto const& interp = far_field.approximation();
        const auto order{interp.order()};
        // width of the current leaf/cell (taking into account the extension)
        const value_type full_width = target_leaf.width() + interp.cell_width_extension();
        //
        simd_position_type simd_center{};
        meta::for_each(simd_center, target_leaf.center(), [](auto const& c) { return simd_type(c); });

        // mapping operator in simd
        const interpolation::map_glob_loc<simd_position_type> mapping_part_position(
          simd_center, simd_position_type(simd_type(full_width)));

        // iterator on the particle container -> std::tuple<iterators...>
        auto position_iterator = container::position_begin(target_leaf.particles());
        auto outputs_iterator = container::outputs_begin(target_leaf.particles());

        const std::size_t nnodes = math::pow(order, position_type::dimension);

        // result of polynomial function for the input particle, here in simd
        std::vector<simd_position_type, XTENSOR_DEFAULT_ALLOCATOR(simd_position_type)> poly_of_part(order);
        // Resulting S, also in simd
        std::vector<value_type, XTENSOR_DEFAULT_ALLOCATOR(value_type)> S(nnodes * inc);

        auto call_polynomials = [&interp](auto x, std::size_t n) { return interp.polynomials(x, n); };

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
                poly_of_part[o] = simd::apply_f<simd_position_type::dimension>(call_polynomials, local_position, o);
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
            auto locals_iterator = source_cell.clocals_begin();
            simd_outputs_type simd_outputs{simd::load_tuple<simd_type>(outputs_iterator, alignment_tag)};

            for(std::size_t m = 0; m < nnodes; ++m)
            {
                auto s_to_apply = xsimd::load_aligned(&S[vec_idx]);

                auto update_potential = [&s_to_apply](auto... local) { return std::make_tuple(s_to_apply * local...); };

                auto simd_locals = simd::load_splat_value_from_it<simd_type>(locals_iterator);

                auto to_update_particles = std::apply(update_potential, simd_locals);

                meta::tuple_sum_update(simd_outputs, to_update_particles);

                meta::repeat([](auto& it) { ++it; }, locals_iterator);
                vec_idx += inc;
            }

            simd::store_tuple<simd_type>(outputs_iterator, simd_outputs, alignment_tag);

            position_iterator += inc;
            outputs_iterator += inc;
        }

        // mapping operator in scalar
        const interpolation::map_glob_loc<position_type> mapping_part_position_scal(target_leaf.center(),
                                                                                    position_type(full_width));

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
                poly_of_part_scal[o] = simd::apply_f<position_type::dimension>(call_polynomials, local_position, o);
            }

            // Assembling S
            std::size_t idx = 0;

            auto construct_s = [&S_scal, &poly_of_part_scal, &idx](auto&... current_indices)
            {
                S_scal[idx] = utils::generate_s<position_type::dimension>(poly_of_part_scal, {{current_indices...}});
                idx++;
            };

            meta::looper<position_type::dimension>()(construct_s, stops);

            auto locals_iterator = source_cell.clocals_begin();

            for(std::size_t m = 0; m < nnodes; ++m)
            {
                auto s_to_apply = S_scal[m];
                auto update_potential = [&s_to_apply](auto... local_it)
                { return std::make_tuple(s_to_apply * (*local_it)...); };

                auto to_update_particles = std::apply(update_potential, locals_iterator);

                meta::tuple_sum_update(*outputs_iterator, to_update_particles);
                meta::repeat([](auto& it) { ++it; }, locals_iterator);
            }

            ++position_iterator;
            ++outputs_iterator;
        }
    }

    /**
     * @brief apply l2p when the potential is evaluated and we derivate it to obtain the forces
     *
     * @tparam FarFieldType Far field type  format is FarFieldType<InterpolatorType, true>
     * @tparam InterpolatorType  Interpolator type used in far field
     * @tparam CellType Cell type
     * @tparam LeafType Leaf type
     * @todo L2P error in the optimized version (shift grad) if we have several locals (both in simd and non simd
     * version)
     * @warning the optimized version works only if the matrix kernel involved in the far-field has only
     * one input
     * @param[in] far_field the interpolation far-field operation
     * @param[in] source_cell the cell containing the local
     * @param[out] target_leaf the leaf containing the particles
     */
    template<template<typename, bool> typename FarFieldType, typename InterpolatorType, typename CellType,
             typename LeafType>
    inline auto apply_l2p(FarFieldType<InterpolatorType, true> const& far_field, CellType const& source_cell,
                          LeafType& target_leaf)
      -> std::enable_if_t<std::is_base_of_v<interpolation::impl::interpolator<InterpolatorType>, InterpolatorType>>
    {
        // Leaf type
        using leaf_type = LeafType;
        using particle_type = typename leaf_type::particle_type;
        using position_type = typename leaf_type::position_type;
        using outputs_type = typename particle_type::outputs_type;
        using value_type = typename leaf_type::value_type;
        using simd_outputs_type =
          decltype(meta::to_tuple(meta::replace_inner_tuple_type_t<xsimd::simd_type, outputs_type>{}));
        // position_type -> scalfmm::container::point<...>
        using position_type = typename leaf_type::position_type;
        static constexpr std::size_t dimension = position_type::dimension;
        // output_type -> std::tuple<...>
        // value_type -> inner position_type::value_type -> underlying
        // scalar_type
        using value_type = typename LeafType::value_type;
        // simd_type -> xsimd::batch<scalar_type>
        using simd_type = xsimd::simd_type<value_type>;
        // simd_position_type -> container::point<xsimd::batch<scalar_type>>
        using simd_position_type = container::point<simd_type, dimension>;
        // simd_vector size -> 8 in double for AVX512 for example
        const std::size_t inc = simd_type::size;
        // number of particles in the leaf
        const std::size_t leaf_size{target_leaf.size()};
        using polynomials_container_type =
          container::particle_container<container::particle<value_type, dimension, value_type, 0, value_type, 0>>;
        /////
        static constexpr auto alignment_tag = simd::unaligned{};
        /////
        auto const& interp = far_field.approximation();
        const auto order{interp.order()};
        static constexpr int number_of_output = particle_type::outputs_size;
        //// if true we compute the potential. If the gradient is false we always
        /// compute it otherwise
        /// it depends if the number of out put is > dimension
        static constexpr bool compute_pot = ((number_of_output > dimension) ? true : false);
        // vectorizable particles
        const std::size_t vec_size = leaf_size - leaf_size % inc;

        // width of the current leaf/cell (taking into account the extension)
        const value_type full_width = target_leaf.width() + interp.cell_width_extension();
        //
        simd_position_type simd_center{};
        meta::for_each(simd_center, target_leaf.center(), [](auto const& c) { return simd_type(c); });

        // mapping operator in simd
        const interpolation::map_glob_loc<simd_position_type> mapping_part_position(
          simd_center, simd_position_type(simd_type(full_width)));

        // iterator on the particle container -> std::tuple<iterators...>
        auto position_iterator = container::position_begin(target_leaf.particles());
        auto outputs_iterator = container::outputs_begin(target_leaf.particles());

        const std::size_t nnodes = math::pow(order, position_type::dimension);

        // result of polynomial function for the input particle, here in simd
        polynomials_container_type poly_of_part(order * inc);
        polynomials_container_type der_poly_of_part(order * inc);   /// HERE

        // Resulting S, also in simd
        using vector_type = std::vector<value_type, XTENSOR_DEFAULT_ALLOCATOR(value_type)>;
        vector_type S(nnodes * inc);
        container::get_variadic_adaptor_t<vector_type, dimension> S_der;
        typename container::get_variadic_adaptor_t<vector_type, dimension>::iterator S_der_it;

        meta::for_each(S_der, [nnodes, inc](auto& c) { c.resize(nnodes * inc); });

        auto call_polynomials = [&interp](auto x, std::size_t n) { return interp.polynomials(x, n); };
        auto call_derivative = [&interp](auto x, std::size_t n) { return interp.derivative(x, n); };

        for(std::size_t part = 0; part < vec_size; part += inc)
        {
            // simd load of particles
            simd_position_type part_position(simd::load_position<simd_position_type>(position_iterator, alignment_tag));
            simd_position_type local_position{};

            // mapping
            mapping_part_position(part_position, local_position);
            const auto jacobian{mapping_part_position.jacobian()};
            //
            auto poly_of_part_begin = poly_of_part.begin();
            auto der_poly_of_part_begin = der_poly_of_part.begin();
            for(std::size_t o = 0; o < order; ++o)
            {
                auto poly_of_part_tmp =
                  simd::apply_f<simd_position_type::dimension>(call_polynomials, local_position, o);
                simd::store_position(poly_of_part_begin, poly_of_part_tmp, alignment_tag);
                auto der_poly_of_part_tmp =
                  simd::apply_f<simd_position_type::dimension>(call_derivative, local_position, o);
                simd::store_position(der_poly_of_part_begin, der_poly_of_part_tmp, alignment_tag);
                poly_of_part_begin += inc;
                der_poly_of_part_begin += inc;
            }

            // Assembling S
            std::size_t idx = 0;

            // lambda function for assembling S
            auto construct_s = [&S, &poly_of_part, &idx](auto&... current_indices)
            {
                auto s_simd = utils::generate_s_simd<inc>(poly_of_part, {{current_indices...}});
                xsimd::store_aligned(&S[idx], s_simd);
                idx += inc;
            };

            // lambda function for assembling the derivate
            auto construct_der_s = [&S_der_it, &poly_of_part, &der_poly_of_part](auto&... current_indices)
            {
                auto s_simd = utils::generate_der_s_simd<inc>(poly_of_part, der_poly_of_part, {{current_indices...}});
                simd::store_position(S_der_it, s_simd);
                S_der_it += inc;
            };

            std::array<std::size_t, position_type::dimension> stops{};
            stops.fill(order);

            // expand N loop over ORDER
            meta::looper<position_type::dimension>()(construct_s, stops);

            S_der_it = S_der.begin();
            meta::looper<position_type::dimension>()(construct_der_s, stops);
            S_der_it = S_der.begin();

            std::size_t vec_idx = 0;
            auto locals_iterator = source_cell.clocals_begin();
            // simd registers of the outputs of the current particle
            simd_outputs_type simd_outputs{simd::load_tuple<simd_type>(outputs_iterator, alignment_tag)};

            using range_pot = meta::make_range_sequence<0, 1>;
            using range_force = std::conditional_t<compute_pot, meta::make_range_sequence<1, number_of_output>,
                                                   meta::make_range_sequence<0, number_of_output>>;
            // Get the reference on the force of the particle in the sime memory
            auto simd_outputs_force = meta::sub_tuple(simd_outputs, range_force{});
            auto simd_outputs_pot = meta::sub_tuple(simd_outputs, range_pot{});
            //
            //  the force contribution set to zero
            meta::replace_inner_tuple_type_t<std::remove_reference_t, std::decay_t<decltype(simd_outputs_force)>>
              simd_contrib_force{};
            meta::repeat([](auto& it) { it = 0; }, simd_contrib_force);
            // Loop on the nodes to construct the force contribution
            for(std::size_t m = 0; m < nnodes; ++m)
            {
                if constexpr(compute_pot == true)
                {
                    auto s_to_apply = xsimd::load_aligned(&S[vec_idx]);

                    auto update_potential = [&s_to_apply](auto... local)
                    { return std::make_tuple(s_to_apply * local...); };

                    auto simd_locals = simd::load_splat_value_from_it<simd_type>(locals_iterator);

                    auto to_update_potential = std::apply(update_potential, simd_locals);
                    // update the potential in the simd s of the outputs
                    meta::tuple_sum_update(simd_outputs_pot, to_update_potential);
                }
                // Force computation
                auto s_der_to_apply = simd::load_position<simd_position_type>(S_der_it, alignment_tag);
                /**
                 * @todo L2P error in the optimized version (shift grad) if we have several locals (both in simd and non
                 * simd version)
                 *
                 */
                auto simd_locals = meta::get<0>(simd::load_splat_value_from_it<simd_type>(locals_iterator));
                auto update_forces = [&simd_locals](auto s_der) { return simd_locals * s_der; };

                auto to_update_particles = meta::apply(s_der_to_apply, update_forces);
                // add the node contribution
                meta::tuple_sum_update(simd_contrib_force, to_update_particles);

                // update the iterators of the locals tensor and the derivative
                S_der_it += inc;
                meta::repeat([](auto& it) { ++it; }, locals_iterator);
                vec_idx += inc;
            }
            // update the simd memory of the outputs with the force contribution multiplied by the jacobian
            //               simd_outputs_force += simd_contrib_force*jacobian
            meta::repeat([](auto& f, auto const& contrib_f, auto const& j) { f += contrib_f * j; }, simd_outputs_force,
                         simd_contrib_force, jacobian);
            // store in memory the computation
            simd::store_tuple<simd_type>(outputs_iterator, simd_outputs, alignment_tag);

            // update the iterators
            position_iterator += inc;
            outputs_iterator += inc;
        }

        // mapping operator in scalar
        const interpolation::map_glob_loc<position_type> mapping_part_position_scal(target_leaf.center(),
                                                                                    position_type(full_width));

        // result of polynomial function for the input particle, here in simd
        std::vector<position_type> poly_of_part_scal(order);
        std::vector<position_type> der_poly_of_part_scal(order);   /// HERE
        // Resulting S, also in simd
        using vector_type = std::vector<value_type, XTENSOR_DEFAULT_ALLOCATOR(value_type)>;
        vector_type S_scal(nnodes);
        container::get_variadic_adaptor_t<vector_type, dimension> S_der_scal;
        typename container::get_variadic_adaptor_t<vector_type, dimension>::iterator S_der_scal_it;

        meta::for_each(S_der_scal, [nnodes](auto& c) { c.resize(nnodes); });

        // Here is the resulting scalar computation
        for(std::size_t part = vec_size; part < leaf_size; ++part)
        {
            // scalar load of particles
            position_type part_position(simd::load_position<position_type>(position_iterator));
            position_type local_position{};   // apres mapping
            // mapping
            mapping_part_position_scal(part_position, local_position);
            const auto jacobian{mapping_part_position_scal.jacobian()};

            // generate polynomials and derivative
            for(std::size_t o = 0; o < order; ++o)
            {
                poly_of_part_scal[o] = simd::apply_f<position_type::dimension>(call_polynomials, local_position, o);
                der_poly_of_part_scal[o] = simd::apply_f<position_type::dimension>(call_derivative, local_position, o);
            }
            // Assembling S
            std::size_t idx = 0;

            auto construct_s = [&S_scal, &poly_of_part_scal, &idx](auto&... current_indices)
            {
                S_scal[idx] = utils::generate_s<position_type::dimension>(poly_of_part_scal, {{current_indices...}});
                idx++;
            };
            auto construct_der_s =
              [&S_der_scal_it, &poly_of_part_scal, &der_poly_of_part_scal](auto&... current_indices)
            {
                auto s_simd = utils::generate_der_s<position_type::dimension>(poly_of_part_scal, der_poly_of_part_scal,
                                                                              {{current_indices...}});
                simd::store_position(S_der_scal_it, s_simd);
                S_der_scal_it++;
            };

            std::array<std::size_t, position_type::dimension> stops{};
            stops.fill(order);

            meta::looper<position_type::dimension>()(construct_s, stops);

            S_der_scal_it = S_der_scal.begin();
            meta::looper<position_type::dimension>()(construct_der_s, stops);
            S_der_scal_it = S_der_scal.begin();

            auto locals_iterator = source_cell.clocals_begin();

            using range_pot = meta::make_range_sequence<0, 1>;
            using range_force = std::conditional_t<compute_pot, meta::make_range_sequence<1, number_of_output>,
                                                   meta::make_range_sequence<0, number_of_output>>;

            auto update_forces = [&locals_iterator](auto s_der) { return *meta::get<0>(locals_iterator) * s_der; };
            decltype(meta::apply(*S_der_scal_it, update_forces)) contribution_force{};
            meta::repeat([](auto& it) { it = 0; }, contribution_force);

            // Loop on the points
            for(std::size_t m = 0; m < nnodes; ++m)
            {
                if constexpr(compute_pot == true)
                {
                    auto s_to_apply = S_scal[m];
                    auto update_potential = [&s_to_apply](auto... local_it)
                    { return std::make_tuple(s_to_apply * (*local_it)...); };

                    auto to_update_particles = std::apply(update_potential, locals_iterator);

                    meta::tuple_sum_update(meta::sub_tuple(*outputs_iterator, range_pot{}), to_update_particles);
                }

                // Compute the  contribution_force += L_k grad(S_k)
                meta::tuple_sum_update(contribution_force, meta::apply(*S_der_scal_it, update_forces));

                // update the iterators
                ++S_der_scal_it;
                meta::repeat([](auto& it) { ++it; }, locals_iterator);
            }
            // apply the jacobian on each component of the force contribution
            meta::repeat([](auto& f, auto const& j) { f *= j; }, contribution_force, jacobian);
            // update the final contribution to the current outputs of the particles
            meta::tuple_sum_update(meta::sub_tuple(*outputs_iterator, range_force{}), contribution_force);

            ++position_iterator;
            ++outputs_iterator;
        }
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_INTERPOLATION_L2P_HPP
