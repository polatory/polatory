// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/p2p.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_P2P_HPP
#define SCALFMM_OPERATORS_P2P_HPP

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/type_pack.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/simd/memory.hpp"
#include "scalfmm/tree/utils.hpp"

#include "xsimd/xsimd.hpp"

#include <array>
#include <numeric>

namespace scalfmm::operators
{
    /**
     * @brief Computes in a mutual way the particle-to-particle interactions between the target and the neighbors.
     *
     * For each particle, x, in target_leaf we compute for each leaf, neighbor_leaf,
     *    in neighbors the interactions between particle y in
     *     $\f[ x.outputs = matrix_kernel(x,y) * y.inputs \f]$
     *     $\f[ y.outputs = matrix_kernel(x,y) * x.inputs\f]$
     *
     * @tparam MatrixKernelType
     * @tparam LeafType
     * @tparam ContainerOfLeafIteratorType
     * @tparam ArrayType
     * @tparam ValueType
     * @param matrix_kernel the kernel used to compute the interactions.
     * @param target_leaf the current leaf where we compute the p2p
     * @param neighbors all neighbouring leaves whose Morton indexes are lower than the target leaf's one.
     * @param size
     * @param pbc
     * @param box_width
     */
    template<typename MatrixKernelType, typename LeafType, typename ContainerOfLeafIteratorType, typename ArrayType,
             typename ValueType>
    inline auto p2p_full_mutual(MatrixKernelType const& matrix_kernel, LeafType& target_leaf,
                                ContainerOfLeafIteratorType const& neighbors, const int& size,
                                [[maybe_unused]] ArrayType const& pbc, ValueType const& box_width) -> void
    {
        using leaf_type = LeafType;
        using matrix_kernel_type = MatrixKernelType;
        // leaf particle type
        using particle_type = typename leaf_type::particle_type;
        // position type
        using position_type = typename leaf_type::position_type;
        // outputs tuple type
        using outputs_type = typename particle_type::outputs_type;
        // value type
        using value_type = typename leaf_type::value_type;
        // simd batch type according to the value type
        using simd_type = xsimd::simd_type<value_type>;
        // simd point type
        using simd_position_type = container::point<simd_type, particle_type::dimension>;
        // outputs tuple type with simd vector
        using simd_tuple_outputs_type =
          decltype(meta::to_tuple(meta::replace_inner_tuple_type_t<xsimd::simd_type, outputs_type>{}));
        // simd array style outputs type.
        using simd_outputs_type = std::array<simd_type, particle_type::outputs_size>;
        static constexpr std::size_t kn = matrix_kernel_type::kn;
        static constexpr std::size_t km = matrix_kernel_type::km;

        // compute the center of the target_leaf
        auto const& target_center = target_leaf.center();

        auto apply_full_mutual = [&target_leaf, &matrix_kernel, &target_center, &pbc, &box_width](auto const& neighbor)
        {
            static constexpr auto alignment_tag = simd::unaligned{};
            const std::size_t inc = simd_type::size;
            // number of particle in the target leaf
            const std::size_t target_size{target_leaf.size()};
            // number of particle in the neighbor leaf
            const std::size_t neighbor_size{neighbor->size()};
            // number of particle to be proceed in simd
            const std::size_t vec_size = neighbor_size - neighbor_size % inc;

#ifdef scalfmm_BUILD_PBC
            auto const& neighbor_center = neighbor->center();
            const auto shift = index::get_shift(target_center, neighbor_center, pbc, box_width);
            simd_position_type simd_shift(shift);
#endif
            // getting iterators from the target particles vector
            auto target_position_iterator = container::position_begin(target_leaf.cparticles());
            auto target_inputs_iterator = container::inputs_begin(target_leaf.cparticles());
            auto target_outputs_iterator = container::outputs_begin(target_leaf.particles());
            // load mutual coeff in simd
            const auto mutual_coefficient = matrix_kernel.template mutual_coefficient<simd_type>();

            // loop on all the particle of the target leaf
            for(std::size_t target_idx = 0; target_idx < target_size; ++target_idx)
            {
                // getting iterators from the neighbor particles vector
                auto neighbor_position_iterator = container::position_begin(neighbor->cparticles());
                auto neighbor_inputs_iterator = container::inputs_begin(neighbor->cparticles());
                auto neighbor_outputs_iterator = container::outputs_begin(neighbor->particles());

                // simd computation
                {
                    // simd target particle position
                    auto target_position =
                      simd::load_position<simd_position_type>(target_position_iterator, simd::splated{});
                    // simd target input values, array style
                    auto target_inputs_values =
                      meta::to_array(simd::load_tuple<simd_type>(target_inputs_iterator, simd::splated{}));

                    // simd outputs
                    simd_outputs_type target_outputs{};
                    meta::for_each(target_outputs, [](auto& p) { p = simd_type(0.); });

                    // loop on the simd part of the neighbor particles
                    for(std::size_t neighbor_idx = 0; neighbor_idx < vec_size; neighbor_idx += inc)
                    {
                        // load simd position
                        auto neighbor_position =
                          simd::load_position<simd_position_type>(neighbor_position_iterator, alignment_tag);
                        // load simd input values
                        auto neighbor_inputs_values =
                          meta::to_array(simd::load_tuple<simd_type>(neighbor_inputs_iterator, alignment_tag));
#ifdef scalfmm_BUILD_PBC
                        neighbor_position += simd_shift;
#endif
                        // load current outputs of the neighbors
                        simd_tuple_outputs_type neighbor_outputs =
                          simd::load_tuple<simd_type>(neighbor_outputs_iterator, alignment_tag);

                        // initialise temporary outputs
                        simd_outputs_type to_update_neighbor_outputs{};
                        meta::for_each(to_update_neighbor_outputs, [](auto& p) { p = simd_type(0.); });

                        // get the matrix kernel evaluation from the particles positions
                        auto kxy_ = matrix_kernel.evaluate(target_position, neighbor_position);

                        // loop on the n:m in/out matrix_kernel
                        for(std::size_t m = 0; m < km; ++m)
                        {
                            for(std::size_t n = 0; n < kn; ++n)
                            {
                                auto kxy_nm = kxy_.at(n * km + m);

                                // apply mutual K
                                // target_out(n) = K(n,m)*neighbor_in(m)
                                target_outputs.at(n) += kxy_nm * neighbor_inputs_values.at(m);
                                // neighbor_out(n) = K(n,m)*target_in(m)
                                to_update_neighbor_outputs.at(n) +=
                                  mutual_coefficient.at(n) * kxy_nm * target_inputs_values.at(m);
                            }
                        }

                        // += on the neighbor_outputs
                        meta::tuple_sum_update(neighbor_outputs, to_update_neighbor_outputs);
                        //// store simd outputs back to the particles vector
                        simd::store_tuple<simd_type>(neighbor_outputs_iterator, neighbor_outputs, alignment_tag);
                        // increment iterators to the next simd sized values
                        neighbor_position_iterator += inc;
                        neighbor_outputs_iterator += inc;
                        neighbor_inputs_iterator += inc;
                    }

                    // reduce the target simd outputs vector
                    auto reduced_target_outputs =
                      meta::apply(target_outputs, [](auto const& f) { return xsimd::reduce_add(f); });
                    // update scalar target output
                    meta::tuple_sum_update(*target_outputs_iterator, reduced_target_outputs);
                }

                // same steps as before but for the scalar end of the neighbors particles vector
                {
                    const auto mutual_coefficient = matrix_kernel.template mutual_coefficient<value_type>();
                    auto target_position = simd::load_position<position_type>(target_position_iterator);
                    auto target_inputs_values = meta::to_array(simd::load_tuple<value_type>(target_inputs_iterator));
                    outputs_type target_outputs{};
                    meta::for_each(target_outputs, [](auto& p) { p = value_type(0.); });

                    for(std::size_t scal_idx = vec_size; scal_idx < neighbor_size; ++scal_idx)
                    {
                        auto neighbor_position = simd::load_position<position_type>(neighbor_position_iterator);
                        auto neighbor_inputs_values =
                          meta::to_array(simd::load_tuple<value_type>(neighbor_inputs_iterator));
#ifdef scalfmm_BUILD_PBC
                        neighbor_position += shift;
#endif
                        outputs_type to_update_neighbor_outputs{};
                        meta::for_each(to_update_neighbor_outputs, [](auto& p) { p = value_type(0.); });

                        auto kxy_ = matrix_kernel.evaluate(target_position, neighbor_position);
                        for(std::size_t m = 0; m < km; ++m)
                        {
                            for(std::size_t n = 0; n < kn; ++n)
                            {
                                auto kxy_nm = kxy_.at(n * km + m);
                                auto tmp_ = kxy_nm * neighbor_inputs_values.at(m);
                                auto tmp__ = kxy_nm * mutual_coefficient.at(n) * target_inputs_values.at(m);
                                target_outputs.at(n) += tmp_;
                                to_update_neighbor_outputs.at(n) += tmp__;
                            }
                        }

                        meta::tuple_sum_update(*neighbor_outputs_iterator, to_update_neighbor_outputs);

                        ++neighbor_position_iterator;
                        ++neighbor_outputs_iterator;
                        ++neighbor_inputs_iterator;
                    }

                    meta::tuple_sum_update(*target_outputs_iterator, target_outputs);
                }
                ++target_position_iterator;
                ++target_outputs_iterator;
                ++target_inputs_iterator;
            }
        };

        // Apply the previous lambda on all the interaction list of the target leaf
        std::for_each(std::begin(neighbors), std::begin(neighbors) + size, apply_full_mutual);
    }

    /**
     * @brief Computes the field due to the  particles the source container on the target particles
     *
     * compute the field due to the  particles the source container on the target particles by
     *
     *   \f$ out(pt) = \sum_{ps\in source} { matrix\_kernel(pt, ps) input(ps) }  \f$
     *
     * We don't check if | pt - ps| =0
     *
     * @tparam MatrixKernelType
     * @tparam TargetLeafType
     * @tparam SourceLeafType
     * @tparam ArrayType
     * @param matrix_kernel the kernel used to compute the interaction
     * @param[inout] target_leaf the container of particles where the field is evaluated
     * @param[in] source_leaf the container of particles which generate the field
     * @param[in] shift the shift to apply on the particle in periodic system.
     */
    template<typename MatrixKernelType, typename TargetLeafType, typename SourceLeafType, typename ArrayType>
    inline auto p2p_outer(MatrixKernelType const& matrix_kernel, TargetLeafType& target_leaf,
                          SourceLeafType const& source_leaf, [[maybe_unused]] ArrayType const& shift) -> void
    {
#ifndef NDEBUG
        if constexpr(std::is_same_v<TargetLeafType, SourceLeafType>)
        {
            if(&target_leaf.symbolics() == &source_leaf.symbolics())
            {
                throw std::runtime_error("In p2p_outer the two containers must be different.");
            }
        }
#endif
        using target_leaf_type = TargetLeafType;
        using source_leaf_type = SourceLeafType;
        using matrix_kernel_type = MatrixKernelType;
        using particle_type = typename target_leaf_type::particle_type;
        using position_type = typename particle_type::position_type;
        using outputs_type = typename particle_type::outputs_type;
        using inputs_value_type = typename source_leaf_type::particle_type::inputs_value_type;

        using value_type = inputs_value_type;
        using simd_type = xsimd::simd_type<value_type>;
        using simd_position_type = container::point<simd_type, position_type::dimension>;
        using simd_outputs_type = std::array<simd_type, particle_type::outputs_size>;
        static constexpr std::size_t kn = matrix_kernel_type::kn;
        static constexpr std::size_t km = matrix_kernel_type::km;

        static constexpr std::size_t inc = simd_type::size;
        const std::size_t target_size{target_leaf.size()};
        const std::size_t neighbor_size{source_leaf.size()};
        const std::size_t vec_size = neighbor_size - neighbor_size % inc;

        auto target_position_iterator = container::position_begin(target_leaf.cparticles());
        auto target_outputs_iterator = container::outputs_begin(target_leaf.particles());

#ifdef scalfmm_BUILD_PBC
        simd_position_type simd_shift(shift);
#endif

        for(std::size_t target_idx = 0; target_idx < target_size; ++target_idx)
        {
            auto neighbor_position_iterator = container::position_begin(source_leaf.cparticles());
            auto neighbor_inputs_iterator = container::inputs_begin(source_leaf.cparticles());

            // Just need the position and the output for the target
            auto target_position = simd::load_position<simd_position_type>(target_position_iterator, simd::splated{});

            simd_outputs_type target_outputs{};
            meta::for_each(target_outputs, [](auto& v) { v = value_type(0.); });

            {
                // loop on the simd part of the neighbor particles
                for(std::size_t neighbor_idx = 0; neighbor_idx < vec_size; neighbor_idx += inc)
                {
                    // load simd position
                    auto neighbor_position =
                      simd::load_position<simd_position_type>(neighbor_position_iterator, simd::unaligned{});
                    // load simd input values
                    auto neighbor_inputs_values =
                      meta::to_array(simd::load_tuple<simd_type>(neighbor_inputs_iterator, simd::unaligned{}));

#ifdef scalfmm_BUILD_PBC
                    neighbor_position += simd_shift;
#endif
                    // get the matrix kernel evaluation from the particles positions
                    auto kxy_ = matrix_kernel.evaluate(target_position, neighbor_position);

                    // loop on the n:m in/out matrix_kernel
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            auto kxy_nm = kxy_.at(n * km + m);
                            // target_out(n) = K(n,m)*neighbor_in(m)
                            target_outputs.at(n) += kxy_nm * neighbor_inputs_values.at(m);
                        }
                    }
                    // increment iterators to the next simd sized values
                    neighbor_position_iterator += inc;
                    neighbor_inputs_iterator += inc;
                }

                // reduce the target simd outputs vector
                auto reduced_target_outputs = meta::apply(target_outputs, [](auto f) { return xsimd::reduce_add(f); });
                // update scalar target output
                meta::tuple_sum_update(*target_outputs_iterator, reduced_target_outputs);
            }
            // same steps as before but for the scalar end of the neighbors particles vector
            {
                auto target_position = simd::load_position<position_type>(target_position_iterator);
                outputs_type target_outputs{};
                meta::for_each(target_outputs, [](auto& p) { p = value_type(0.); });

                for(std::size_t scal_idx = vec_size; scal_idx < neighbor_size; ++scal_idx)
                {
                    auto neighbor_position = simd::load_position<position_type>(neighbor_position_iterator);
                    auto neighbor_inputs_values =
                      meta::to_array(simd::load_tuple<value_type>(neighbor_inputs_iterator));

#ifdef scalfmm_BUILD_PBC
                    neighbor_position += shift;
#endif
                    auto kxy_ = matrix_kernel.evaluate(target_position, neighbor_position);
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            auto kxy_nm = kxy_.at(n * km + m);
                            auto tmp_ = kxy_nm * neighbor_inputs_values.at(m);
                            target_outputs.at(n) += tmp_;
                        }
                    }

                    ++neighbor_position_iterator;
                    ++neighbor_inputs_iterator;
                }

                meta::tuple_sum_update(*target_outputs_iterator, target_outputs);
            }
            ++target_position_iterator;
            ++target_outputs_iterator;
        }
    }

    /**
     * @brief p2p_outer compute the particle-to-particle interactions between the target leaf
     *  and its neighbors.
     *
     * For each particle, x, in target_leaf we compute for each leaf, neighbor_leaf,
     *    in neighbors the interactions between particle y in
     *     $\f[ x.outputs = matrix_kernel(x,y) * y.inputs \f]$
     *
     * @warning{The neighbors are not modified.}
     * @todo{To be implemented}
     *
     * @tparam MatrixKernelType
     * @tparam LeafType
     * @tparam ContainerOfLeafIteratorType
     * @tparam ArrayType
     * @tparam ValueType
     * @param matrix_kernel the kernel used to compute the interactions.
     * @param target_leaf the current leaf where we compute the p2p.
     * @param neighbors all neighboring leaves.
     * @param number_of_neighbors number of neighbors.
     * @param pbc
     * @param box_width
     */
    template<typename MatrixKernelType, typename LeafType, typename ContainerOfLeafIteratorType, typename ArrayType,
             typename ValueType>
    inline auto p2p_outer(MatrixKernelType const& matrix_kernel, LeafType& target_leaf,
                          ContainerOfLeafIteratorType const& neighbors, const int number_of_neighbors,
                          [[maybe_unused]] ArrayType const& pbc, ValueType const& box_width) -> void
    {
        using leaf_type = LeafType;
        using position_type = typename leaf_type::position_type;
        // // value type
        using value_type = typename leaf_type::value_type;
        //
        // compute the center of the target_leaf
        auto const& target_center = target_leaf.center();

        auto apply_outer = [&target_leaf, &matrix_kernel, &target_center, &pbc, &box_width](auto const& neighbor)
        {
#ifdef scalfmm_BUILD_PBC
            auto const& neighbor_center = neighbor->center();
            auto shift = index::get_shift(target_center, neighbor_center, pbc, box_width);
#else
            position_type shift(value_type(0.0));
#endif

            p2p_outer(matrix_kernel, target_leaf, *neighbor, shift);
        };
        std::for_each(std::begin(neighbors), std::begin(neighbors) + number_of_neighbors, apply_outer);
    }

    /**
     * @brief P2P function between the target leaf to update and the list of neighbors.
     *
     * @tparam MatrixKernelType
     * @tparam LeafType
     * @tparam ContainerOfLeafIteratorType
     * @tparam ArrayType
     * @tparam ValueType
     * @param[in] matrix_kernel The kernel used to compute the interactions.
     * @param[inout] target_leaf the target leaf (the output field will be updated by the source leaf).
     * @param[in] neighbors the array of neighbors.
     * @param[in] pbc the array of periodic boundary condition.
     * @param[in] box_width the width of the simulation box.
     */
    template<typename MatrixKernelType, typename LeafType, typename ContainerOfLeafIteratorType, typename ArrayType,
             typename ValueType>
    inline auto p2p_outer(MatrixKernelType const& matrix_kernel, LeafType& target_leaf,
                          ContainerOfLeafIteratorType const& neighbors, [[maybe_unused]] ArrayType const& pbc,
                          ValueType const& box_width) -> void
    {
        p2p_outer(matrix_kernel, target_leaf, neighbors, target_leaf.csymbolics().number_of_neighbors, pbc, box_width);
    }

    /**
     * @brief Compute the particle-to-particle interactions inside of a target leaf with mutual computation.
     *
     * For each particle, x, in target_leaf we compute for each particle y (!=x)
     *     \f[ x.outputs = matrix_kernel(x,y) * y.inputs \f] for y !=x
     *     \f[ y.outputs = matrix_kernel(x,y) * x.inputs \f] for y !=x
     *
     * Most the computations are done using SIMD instructions. Assuming that there are N particles in the target leaf,
     * the total number of interactions to be computed is proportional to N x (N - 1) / 2.
     *
     * @tparam MatrixKernelType The type of the matrix kernel that is used in the p2p computation.
     * @tparam LeafType The type of the leaf on which the operation will be performed.
     *
     * @param matrix_kernel a constant reference to the kernel object used to compute the interactions.
     * @param target_leaf the current leaf where we compute the particle-to-particle interactions.
     */
    template<typename MatrixKernelType, typename LeafType>
    inline auto p2p_inner_mutual(MatrixKernelType const& matrix_kernel, LeafType& target_leaf) -> void
    {
        using leaf_type = LeafType;
        using matrix_kernel_type = MatrixKernelType;
        using particle_type = typename leaf_type::particle_type;
        using position_type = typename leaf_type::position_type;
        using outputs_type = typename particle_type::outputs_type;
        using value_type = typename leaf_type::value_type;
        using simd_type = xsimd::simd_type<value_type>;

        static constexpr std::size_t dimension = particle_type::dimension;
        using simd_position_type = container::point<simd_type, dimension>;
        using simd_outputs_type = std::array<simd_type, particle_type::outputs_size>;

        static constexpr std::size_t kn = matrix_kernel_type::kn;
        static constexpr std::size_t km = matrix_kernel_type::km;

        constexpr std::size_t inc = simd_type::size;
        const std::size_t target_size{target_leaf.size()};
        const std::size_t neighbor_size{target_leaf.size()};

        auto target_position_iterator = container::position_begin(target_leaf.cparticles());
        auto target_inputs_iterator = container::inputs_begin(target_leaf.cparticles());
        auto target_outputs_iterator = container::outputs_begin(target_leaf.particles());

        for(std::size_t target_idx = 0; target_idx < target_size; ++target_idx)
        {
            auto neighbor_idx = target_idx + 1;
            auto neighbor_position_iterator = container::position_begin(target_leaf.cparticles()) + neighbor_idx;
            auto neighbor_inputs_iterator = container::inputs_begin(target_leaf.cparticles()) + neighbor_idx;
            auto neighbor_outputs_iterator = container::outputs_begin(target_leaf.particles()) + neighbor_idx;
            const auto mutual_coefficient = matrix_kernel.template mutual_coefficient<simd_type>();

            // first part: simd computation
            {
                auto target_position =
                  simd::load_position<simd_position_type>(target_position_iterator, simd::splated{});
                auto target_inputs_values =
                  meta::to_array(simd::load_tuple<simd_type>(target_inputs_iterator, simd::splated{}));

                simd_outputs_type target_outputs{};
                meta::for_each(target_outputs, [](auto& p) { p = value_type(0.); });

                const std::size_t vec_size = ((neighbor_size - neighbor_idx) / inc) * inc + neighbor_idx;

                for(; neighbor_idx < vec_size; neighbor_idx += inc)
                {
                    auto neighbor_position =
                      simd::load_position<simd_position_type>(neighbor_position_iterator, simd::unaligned{});
                    auto neighbor_inputs_values =
                      meta::to_array(simd::load_tuple<simd_type>(neighbor_inputs_iterator, simd::unaligned{}));
                    auto neighbor_outputs = simd::load_tuple<simd_type>(neighbor_outputs_iterator, simd::unaligned{});

                    simd_outputs_type to_update_neighbor_outputs{};
                    meta::for_each(to_update_neighbor_outputs, [](auto& p) { p = simd_type(0.); });

                    auto kxy_ = matrix_kernel.evaluate(target_position, neighbor_position);
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            auto kxy_nm = kxy_.at(n * km + m);
                            // auto tmp = to_update_target_outputs.at(n);
                            // to_update_target_outputs.at(n) = xsimd::fma(kxy_nm, neighbor_inputs_values.at(m),
                            // tmp);
                            target_outputs.at(n) += kxy_nm * neighbor_inputs_values.at(m);
                            to_update_neighbor_outputs.at(n) +=
                              mutual_coefficient.at(n) * kxy_nm * target_inputs_values.at(m);
                        }
                    }

                    meta::tuple_sum_update(neighbor_outputs, to_update_neighbor_outputs);
                    simd::store_tuple<simd_type>(neighbor_outputs_iterator, neighbor_outputs, simd::unaligned{});

                    neighbor_position_iterator += inc;
                    neighbor_inputs_iterator += inc;
                    neighbor_outputs_iterator += inc;
                }

                auto reduced_target_outputs = meta::apply(target_outputs, [](auto f) { return xsimd::reduce_add(f); });

                meta::tuple_sum_update(*target_outputs_iterator, reduced_target_outputs);
            }

            {
                const auto mutual_coefficient = matrix_kernel.template mutual_coefficient<value_type>();
                auto target_position = simd::load_position<position_type>(target_position_iterator, simd::splated{});
                auto target_inputs_values =
                  meta::to_array(simd::load_tuple<value_type>(target_inputs_iterator, simd::splated{}));

                outputs_type target_outputs{};
                meta::for_each(target_outputs, [](auto& p) { p = value_type(0.); });

                for(; neighbor_idx < neighbor_size; ++neighbor_idx)
                {
                    auto neighbor_position = simd::load_position<position_type>(neighbor_position_iterator);

                    auto neighbor_inputs_values =
                      meta::to_array(simd::load_tuple<value_type>(neighbor_inputs_iterator));

                    outputs_type neighbor_outputs =
                      meta::to_array(simd::load_tuple<value_type>(neighbor_outputs_iterator, simd::splated{}));

                    outputs_type to_update_neighbor_outputs{};
                    meta::for_each(to_update_neighbor_outputs, [](auto& p) { p = value_type(0.); });

                    auto kxy_ = matrix_kernel.evaluate(target_position, neighbor_position);
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            auto kxy_nm = kxy_.at(n * km + m);
                            auto tmp = kxy_nm * neighbor_inputs_values.at(m);
                            auto tmp_ = mutual_coefficient.at(n) * kxy_nm * target_inputs_values.at(m);
                            target_outputs.at(n) += tmp;   // kxy_nm * neighbor_inputs_values.at(m);
                            to_update_neighbor_outputs.at(n) += tmp_;
                        }
                    }

                    meta::tuple_sum_update(neighbor_outputs, to_update_neighbor_outputs);
                    simd::store_tuple<value_type>(neighbor_outputs_iterator, neighbor_outputs);

                    ++neighbor_position_iterator;
                    ++neighbor_inputs_iterator;
                    ++neighbor_outputs_iterator;
                }

                meta::tuple_sum_update(*target_outputs_iterator, target_outputs);
            }

            // third part: self-interaction (only if the kernel allows it)
            {
                if constexpr(meta::has_self_interaction_v<matrix_kernel_type, position_type>)
                {
                    // load scalar target position
                    auto target_position =
                      simd::load_position<position_type>(target_position_iterator, simd::splated{});
                    outputs_type target_outputs{};
                    meta::for_each(target_outputs, [](auto& p) { p = value_type(0.); });

                    // load scalar target input
                    auto target_inputs = meta::to_array(
                      simd::load_tuple<value_type>(container::inputs_begin(target_leaf.cparticles()) + target_idx));

                    // get the matrix kernel evaluation from the particles' positions
                    auto kx_ = matrix_kernel.evaluate(target_position);

                    for(std::size_t m = 0; m < km; ++m)
                    {
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            auto kx_nm = kx_.at(n * km + m);
                            target_outputs.at(n) += kx_nm * target_inputs.at(m);
                        }
                    }

                    // update scalar target output
                    meta::tuple_sum_update(*target_outputs_iterator, target_outputs);
                }
            }

            ++target_position_iterator;
            ++target_inputs_iterator;
            ++target_outputs_iterator;
        }
    }

    /**
     * @brief Compute the particle-to-particle interactions inside of a target leaf without mutual computation.
     *
     * For each particle, x, in target_leaf we compute for each particle y (!=x)
     *     \f[ x.outputs = matrix_kernel(x,y) * y.inputs \f] for y !=x
     *
     * Most the computations are done using SIMD instructions. Assuming that there are N particles in the target leaf,
     * the total number of interactions to be computed is proportional to N x (N - 1).
     *
     * @tparam MatrixKernelType The type of the matrix kernel that is used in the p2p computation.
     * @tparam LeafType The type of the leaf on which the operation will be performed.
     *
     * @param matrix_kernel A constant reference to the kernel object used to compute the interactions.
     * @param target_leaf The current leaf where we compute the particle-to-particle interactions.
     */
    template<typename MatrixKernelType, typename LeafType>
    inline auto p2p_inner_non_mutual(MatrixKernelType const& matrix_kernel, LeafType& target_leaf) -> void
    {
        using leaf_type = LeafType;
        using matrix_kernel_type = MatrixKernelType;
        using particle_type = typename leaf_type::particle_type;
        using position_type = typename leaf_type::position_type;
        using outputs_type = typename particle_type::outputs_type;
        using value_type = typename leaf_type::value_type;
        using simd_type = xsimd::simd_type<value_type>;
        static constexpr std::size_t dimension = particle_type::dimension;
        using simd_position_type = container::point<simd_type, dimension>;
        using simd_outputs_type = std::array<simd_type, particle_type::outputs_size>;
        static constexpr std::size_t kn = matrix_kernel_type::kn;
        static constexpr std::size_t km = matrix_kernel_type::km;

        constexpr std::size_t inc = simd_type::size;
        const std::size_t size{target_leaf.size()};
        const std::size_t vec_size = size - size % inc;

        auto const local_tolerance{std::numeric_limits<value_type>::min()};

        auto target_position_iterator = container::position_begin(target_leaf.cparticles());
        auto target_outputs_iterator = container::outputs_begin(target_leaf.particles());

        // initialize mask to filer the case where target_idx == source_idx
        std::array<value_type, inc> next_source_indices;
        std::iota(next_source_indices.begin(), next_source_indices.end(), value_type(0.));
        // loop on the target indices
        for(std::size_t target_idx = 0; target_idx < size; ++target_idx)
        {
            // batch for source indices (starting with 0,1,2,... and later incremented with 'inc')
            auto next_source_indices_batch_mask = xsimd::load_unaligned(next_source_indices.data());

            auto source_position_iterator = container::position_begin(target_leaf.cparticles());
            auto source_inputs_iterator = container::inputs_begin(target_leaf.cparticles());

            // need the position and the output for the target
            auto target_position = simd::load_position<simd_position_type>(target_position_iterator, simd::splated{});

            simd_outputs_type target_outputs{};
            meta::for_each(target_outputs, [](auto& v) { v = value_type(0.); });

            // first part: simd computation
            {
                // loop on the simd part of the particles
                for(std::size_t source_idx = 0; source_idx < vec_size;
                    source_idx += inc, next_source_indices_batch_mask += inc)
                {
                    // generate a mask for the current slice based on the condition target_idx != source_idx
                    auto batch_mask = xsimd::abs(next_source_indices_batch_mask - target_idx) > local_tolerance;

                    // load simd position
                    auto source_position =
                      simd::load_position<simd_position_type>(source_position_iterator, simd::unaligned{});
                    // load simd input
                    auto source_inputs =
                      meta::to_array(simd::load_tuple<simd_type>(source_inputs_iterator, simd::unaligned{}));
                    // get the matrix kernel evaluation from the particles' positions
                    auto kxy_ = matrix_kernel.evaluate(target_position, source_position);

                    // loop on the n:m in/out matrix_kernel
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            auto kxy_nm = kxy_.at(n * km + m);
                            // target_out(n) = K(n,m) * input_in(m)
                            // filter the case where target_idx == source_idx
                            target_outputs.at(n) +=
                              xsimd::select(batch_mask, kxy_nm * source_inputs.at(m), simd_type(0.));
                        }
                    }

                    // increment the source iterators
                    source_position_iterator += inc;
                    source_inputs_iterator += inc;
                }
                // reduce the target simd outputs vector
                auto reduced_target_outputs = meta::apply(target_outputs, [](auto f) { return xsimd::reduce_add(f); });
                // update scalar target output
                meta::tuple_sum_update(*target_outputs_iterator, reduced_target_outputs);
            }

            // second part: scalar computation (end of the vector)
            // we make sure that the iteration target_idx == source_idx is not treated
            {
                auto target_position = simd::load_position<position_type>(target_position_iterator, simd::splated{});
                outputs_type target_outputs{};
                meta::for_each(target_outputs, [](auto& p) { p = value_type(0.); });

                for(std::size_t source_idx = vec_size; source_idx < size; ++source_idx)
                {
                    if(target_idx != source_idx)
                    {
                        // load scalar position
                        auto source_position = simd::load_position<position_type>(source_position_iterator);
                        // load scalar input
                        auto source_inputs = meta::to_array(simd::load_tuple<value_type>(source_inputs_iterator));
                        // get the matrix kernel evaluation from the particles' positions
                        auto kxy_ = matrix_kernel.evaluate(target_position, source_position);

                        for(std::size_t m = 0; m < km; ++m)
                        {
                            for(std::size_t n = 0; n < kn; ++n)
                            {
                                auto kxy_nm = kxy_.at(n * km + m);
                                // target_out(n) = K(n,m) * input_in(m)
                                target_outputs.at(n) += kxy_nm * source_inputs.at(m);
                            }
                        }
                    }

                    // increment the source iterators
                    ++source_position_iterator;
                    ++source_inputs_iterator;
                }

                // update scalar target output
                meta::tuple_sum_update(*target_outputs_iterator, target_outputs);
            }

            // third part: self-interaction (only if the kernel allows it)
            {
                if constexpr(meta::has_self_interaction_v<matrix_kernel_type, position_type>)
                {
                    // load scalar target position
                    auto target_position =
                      simd::load_position<position_type>(target_position_iterator, simd::splated{});
                    outputs_type target_outputs{};
                    meta::for_each(target_outputs, [](auto& p) { p = value_type(0.); });

                    // load scalar target input
                    auto target_inputs = meta::to_array(
                      simd::load_tuple<value_type>(container::inputs_begin(target_leaf.cparticles()) + target_idx));

                    // get the matrix kernel evaluation from the particles' positions
                    auto kx_ = matrix_kernel.evaluate(target_position);

                    for(std::size_t m = 0; m < km; ++m)
                    {
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            auto kx_nm = kx_.at(n * km + m);
                            target_outputs.at(n) += kx_nm * target_inputs.at(m);
                        }
                    }

                    // update scalar target output
                    meta::tuple_sum_update(*target_outputs_iterator, target_outputs);
                }
            }

            ++target_position_iterator;
            ++target_outputs_iterator;
        }
    }

    /**
     * @brief Compute the particle-to-particle interactions inside of a target leaf.
     *
     * This function acts as an interface that either calls the mutual p2p inner operator or the non-mutual p2P inner
     * operator depending on the boolean parameter 'mutual'.
     *
     * @tparam MatrixKernelType The type of the matrix kernel that is used in the p2p computation.
     * @tparam LeafType The type of the leaf on which the operation will be performed.
     *
     * @param matrix_kernel A constant reference to the kernel object used to compute the interactions.
     * @param target_leaf The current leaf where we compute the particle-to-particle interactions.
     * @param mutual An optional boolean flag that determines whether the computation is mutual or not.
     */
    template<typename MatrixKernelType, typename LeafType>
    inline auto p2p_inner(MatrixKernelType const& matrix_kernel, LeafType& target_leaf, bool mutual = true) -> void
    {
        using matrix_kernel_type = MatrixKernelType;
        using leaf_type = LeafType;

        static constexpr std::size_t dimension = leaf_type::dimension;
        using array_type = std::array<bool, dimension>;

        if(mutual)
        {
            p2p_inner_mutual(matrix_kernel, target_leaf);
        }
        else
        {
            p2p_inner_non_mutual(matrix_kernel, target_leaf);
        }
    }

}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_P2P_HPP
