// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/interpolation/m2m_l2l.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_INTERPOLATION_M2M_L2L_HPP
#define SCALFMM_OPERATORS_INTERPOLATION_M2M_L2L_HPP

#include <xtensor-blas/xblas.hpp>
#include <xtensor-blas/xblas_utils.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/views/xview.hpp>

#include "scalfmm/interpolation/interpolator.hpp"
#include "scalfmm/memory/storage.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"

#include <array>
#include <type_traits>
#include <utility>

namespace scalfmm::operators::impl
{
    /**
     * @brief Computes whether it needs to transpose or not according to the operator
     *
     * @tparam Tag
     */
    template<typename Tag>
    constexpr auto transpose(Tag /*tag*/)
    {
        if constexpr(std::is_same_v<Tag, tag_m2m>)
        {
            return cxxblas::Trans;
        }
        else if constexpr(std::is_same_v<Tag, tag_l2l>)
        {
            return cxxblas::NoTrans;
        }
        else
        {
            return cxxblas::NoTrans;
        }
    }
}   // namespace scalfmm::operators::impl

namespace scalfmm::operators
{
    namespace impl
    {
        // -------------------------------------------------------------
        // Compute : core optimized function for m2m and l2l
        // -------------------------------------------------------------

        /**
         * @brief
         *
         * @tparam I
         * @tparam Interp
         * @tparam TensorMultipole
         * @tparam Tag
         */
        template<std::size_t I, typename Interp, typename TensorMultipole, typename Tag>
        inline void compute_opt(Interp const& interp, TensorMultipole const& child_expansion, std::size_t child_index,
                                xt::xarray<typename TensorMultipole::value_type>& expansion_buffer,
                                xt::xarray<typename TensorMultipole::value_type>& perm_expansion_buffer,
                                std::size_t order, std::size_t tree_level, Tag /* tag */)
        {
            using value_type = typename TensorMultipole::value_type;
            std::size_t multipoles_dim = Interp::dimension;
            std::size_t nnodes{interp.nnodes()};

            auto const& interp_tensor{interp.cinterpolator_tensor()};
            auto const& grid_permutations{interp.grid_permutations()};
            auto size{static_cast<int>(math::pow(order, multipoles_dim - 1))};
            auto order_int{static_cast<int>(order)};
            // Call to blas
            if constexpr(I == 0)
            {
                cxxblas::gemm(cxxblas::StorageOrder::RowMajor, transpose(Tag{}), cxxblas::NoTrans, order_int, size,
                              order_int, value_type(1.), &interp_tensor.at(tree_level, child_index, 0, 0), order_int,
                              child_expansion.data(), size, value_type(0.), perm_expansion_buffer.data(), size);
            }
            else
            {
                const auto perm_data = grid_permutations.data();
                const auto exp_buff_data = expansion_buffer.data();
                const auto perm_exp_buff_data = perm_expansion_buffer.data();
                //auto perm_1 = &grid_permutations.at(I-1,0);
                auto perm_1 = perm_data + (I - 1) * nnodes;
                //auto perm_2 = &grid_permutations.at(I,0);
                auto perm_2 = perm_data + I * nnodes;
                for(std::size_t n{0}; n < nnodes; ++n)
                {
                    exp_buff_data[perm_1[n]] = perm_exp_buff_data[perm_2[n]];
                    //expansion_buffer.data()[perm_1[n]] = perm_expansion_buffer.data()[perm_2[n]];
                }

                cxxblas::gemm(xt::get_blas_storage_order(expansion_buffer), transpose(Tag{}), cxxblas::NoTrans,
                              order_int, size, order_int, value_type(1.),
                              &interp_tensor.at(tree_level, child_index, I, 0), order_int, expansion_buffer.data(),
                              size, value_type(0.), perm_expansion_buffer.data(), size);
            }
        }

        // -------------------------------------------------------------
        // Compute : core generic function for m2m and l2l
        // -------------------------------------------------------------

        /**
         * @brief
         *
         * @tparam Interp
         * @tparam TensorMultipole
         * @tparam Tag
         */
        template<typename Interp, typename TensorMultipole, typename Tag>
        inline void compute(Interp const& interp, TensorMultipole const& child_expansion, std::size_t child_index,
                            xt::xarray<typename TensorMultipole::value_type>& expansion_buffer, std::size_t order,
                            std::size_t tree_level, std::size_t dimension, Tag /* tag */)
        {
            using value_type = typename TensorMultipole::value_type;
            std::size_t multipoles_dim = expansion_buffer.dimension();
            std::array<std::size_t, 2> matrix_shape = {order, math::pow(order, multipoles_dim - 1)};
            auto buffer = xt::xarray<value_type>::from_shape(matrix_shape);
            auto const& interp_tensor{interp.cinterpolator_tensor()};

            // Retrieve interpolator matrix according to tree level, child index and dimension
            auto interpolator_view = xt::view(interp_tensor, tree_level, child_index, dimension, xt::all());
            // Reshape it for blas operation
            auto reshape_interpolator_view = xt::eval(xt::reshape_view(interpolator_view, std::array{order, order}));
            // Unfolding child tensor expansion
            auto b_mat = xt::eval(tensor::unfold(child_expansion, dimension));
            // Call to blas
            xt::blas::gemm(reshape_interpolator_view, b_mat, buffer, transpose(Tag{}));
            // Folding back the result into a tensor
            xt::eval(expansion_buffer = tensor::fold(buffer, dimension, expansion_buffer.shape()));
        }

        /**
         * @brief
         *
         * @tparam Interp
         * @tparam TensorMultipole
         * @tparam Tag
         * @tparam Is
         */
        template<typename Interp, typename TensorMultipole, typename Tag, std::size_t... Is>
        inline void expand(Interp const& interp, TensorMultipole const& child_expansion, std::size_t child_index,
                           TensorMultipole& parent_expansion,
                           xt::xarray<typename TensorMultipole::value_type>& expansion_buffer,
                           xt::xarray<typename TensorMultipole::value_type>& perm_expansion_buffer, std::size_t order,
                           std::size_t tree_level, Tag /* tag */, std::index_sequence<Is...> /*s*/)
        {
            static constexpr auto dim_minus_one{sizeof...(Is)};

            if constexpr(dim_minus_one < 3 && dim_minus_one > 0)
            {
                // For compute il directly made form the child tensor expansion
                compute_opt<0>(interp, child_expansion, child_index, expansion_buffer, perm_expansion_buffer, order,
                               tree_level, Tag{});

                if constexpr(dim_minus_one > 0)
                {
                    // Others are expand using the buffered result (dimension -1 calls)
                    meta::noop_t{((compute_opt<Is + 1>(interp, expansion_buffer, child_index, expansion_buffer,
                                                       perm_expansion_buffer, order, tree_level, Tag{})),
                                  0)...};

                    auto const& grid_permutations{interp.grid_permutations()};
                    const auto perm = &grid_permutations.at(dim_minus_one, 0);
                    const auto exp_buff_data = expansion_buffer.data();
                    const auto perm_exp_buff_data = perm_expansion_buffer.data();
                    for(std::size_t n{0}; n < interp.nnodes(); ++n)
                    {
                        exp_buff_data[perm[n]] = perm_exp_buff_data[n];
                        //expansion_buffer.data()[perm[n]] = perm_expansion_buffer.data()[n];
                    }
                }
            }
            else
            {
                // For compute il directly made form the child tensor expansion
                compute(interp, child_expansion, child_index, expansion_buffer, order, tree_level, 0, Tag{});
                // Others are expand using the buffered result (dimension -1 calls)
                meta::noop_t{
                  ((compute(interp, expansion_buffer, child_index, expansion_buffer, order, tree_level, Is + 1, Tag{})),
                   0)...};
            }
            // Updating parent tensorial expansion
            using value_type = typename TensorMultipole::value_type;
            cxxblas::axpy(static_cast<int>(expansion_buffer.size()), value_type(1.), expansion_buffer.data(),
                          static_cast<int>(1), parent_expansion.data(), static_cast<int>(1));
        }

        /**
         * @brief
         *
         * @tparam Interp
         * @tparam TensorMultipole
         * @tparam Tag
         */
        template<typename Interp, typename TensorMultipole, typename Tag>
        inline void expand(Interp const& interp, TensorMultipole const& child_expansion, std::size_t child_index,
                           TensorMultipole& parent_expansion,
                           xt::xarray<typename TensorMultipole::value_type>& expansion_buffer,
                           xt::xarray<typename TensorMultipole::value_type>& perm_expansion_buffer, std::size_t order,
                           std::size_t tree_level, Tag /* tag */)
        {
            using tensor_multipole_shape = typename std::decay_t<TensorMultipole>::shape_type;

            // Expand computation (compute function) on dimension - 1 as the first call directly
            // use the child tensor, see expand() call.
            return expand(interp, child_expansion, child_index, parent_expansion, expansion_buffer,
                          perm_expansion_buffer, order, tree_level, Tag{},
                          std::make_index_sequence<std::tuple_size<tensor_multipole_shape>::value - 1>{});
        }
    }   // namespace impl

    // -------------------------------------------------------------
    // M2M operator
    // -------------------------------------------------------------

    /**
     * @brief
     *
     * @tparam D
     * @tparam Cell
     * @param interp
     * @param child_cell
     * @param child_index
     * @param parent_cell
     * @param tree_level
     */
    template<typename D, typename Cell>
    inline auto apply_m2m(interpolation::impl::interpolator<D> const& interp, Cell const& child_cell,
                          std::size_t child_index, Cell& parent_cell, std::size_t tree_level) -> void
    {
        using tensor_multipole_type =
          typename memory::storage_traits<typename Cell::storage_type::multipoles_storage_type>::inner_type;

        typename tensor_multipole_type::shape_type order_shape;
        auto order{interp.order()};
        order_shape.fill(order);

        // get multipoles
        auto& parent_expansion = parent_cell.multipoles();
        auto const& child_expansion = child_cell.cmultipoles();

        // Creating buffer for blas computation
        auto expansion_buffer = xt::xarray<typename tensor_multipole_type::value_type>::from_shape(order_shape);
        // TODO : empty perm_expansion_buffer when dimension is > 3
        auto perm_expansion_buffer = xt::xarray<typename tensor_multipole_type::value_type>::from_shape(order_shape);

        // Expanding computation according to the number of multipole values
        auto tensor_size = child_expansion.size();

        for(std::size_t km_{0}; km_ < tensor_size; ++km_)
        {
            impl::expand(interp, child_expansion.at(km_), child_index, parent_expansion.at(km_), expansion_buffer,
                         perm_expansion_buffer, order, tree_level, impl::tag_m2m{});
        }
    }

    // -------------------------------------------------------------
    // L2L operator
    // -------------------------------------------------------------

    /**
     * @brief
     *
     * @tparam D
     * @tparam Cell
     * @param interp
     * @param parent_cell
     * @param child_index
     * @param child_cell
     * @param tree_level
     */
    template<typename D, typename Cell>
    inline auto apply_l2l(interpolation::impl::interpolator<D> const& interp, Cell const& parent_cell,
                          std::size_t child_index, Cell& child_cell, std::size_t tree_level) -> void
    {
        using tensor_multipole_type =
          typename memory::storage_traits<typename Cell::storage_type::multipoles_storage_type>::inner_type;

        typename tensor_multipole_type::shape_type order_shape;
        auto order{interp.order()};
        order_shape.fill(order);

        // Creating buffer for blas computation
        auto expansion_buffer = xt::xarray<typename tensor_multipole_type::value_type>::from_shape(order_shape);
        // TODO : empty perm_expansion_buffer when dimension is > 3
        auto perm_expansion_buffer = xt::xarray<typename tensor_multipole_type::value_type>::from_shape(order_shape);

        // get multipoles
        auto const& parent_expansion = parent_cell.clocals();
        auto& child_expansion = child_cell.locals();

        // Expanding computation according to the number of multipole values
        auto tensor_size = child_expansion.size();

        for(std::size_t kn_{0}; kn_ < tensor_size; ++kn_)
        {
            impl::expand(interp, parent_expansion.at(kn_), child_index, child_expansion.at(kn_), expansion_buffer,
                         perm_expansion_buffer, order, tree_level, impl::tag_l2l{});
        }
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_INTERPOLATION_M2M_L2L_HPP
