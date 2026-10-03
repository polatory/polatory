// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/generate_circulent.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_GENERATE_CIRCULENT_HPP
#define SCALFMM_INTERPOLATION_GENERATE_CIRCULENT_HPP

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/utils/tensor.hpp"

#include "xtensor/core/xtensor_forward.hpp"
#include "xtensor/views/xview.hpp"

#include <cstddef>
#include <utility>
#include <vector>
#include <xtensor/views/xslice.hpp>

namespace scalfmm::interpolation
{
    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam SavedDimension
     */
    template<typename ValueType, std::size_t Dimension, std::size_t SavedDimension>
    struct build
    {
        using value_type = ValueType;

        /**
         * @brief
         *
         * @tparam TensorViewX
         * @tparam TensorViewY
         * @tparam FarField
         * @param X
         * @param Y
         * @param order
         * @param far_field
         * @param n
         * @param m
         * @return auto
         */
        template<typename TensorViewX, typename TensorViewY, typename FarField>
        [[nodiscard]] inline auto operator()(TensorViewX&& X, TensorViewY&& Y, std::size_t order,
                                             FarField const& far_field, std::size_t n, std::size_t m)
        {
            static constexpr std::size_t current_dimension = Dimension;
            static constexpr std::size_t saved_dimension = SavedDimension;
            static constexpr std::size_t dimension_decrease = current_dimension - 1;
            std::size_t ntilde = 2 * order - 1;
            xt::xarray<value_type> c(std::vector(current_dimension, std::size_t(ntilde)));
            build<value_type, dimension_decrease, saved_dimension> build_c1{};

            for(std::size_t i = 0; i < order; ++i)
            {
                auto range = xt::all();
                auto c1 = tensor::get_view<dimension_decrease>(c, i, range, tensor::row{});
                auto X_view =
                  tensor::get_view<dimension_decrease>(std::forward<TensorViewX>(X), i, range, tensor::row{});
                auto Y_view =
                  tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), 0, range, tensor::row{});
                c1 = build_c1(std::forward<decltype(X_view)>(X_view), std::forward<decltype(Y_view)>(Y_view), order,
                              far_field, n, m);
            }

            for(std::size_t i = 1; i < order; ++i)
            {
                auto range = xt::all();
                auto c1 = tensor::get_view<dimension_decrease>(c, order - 1 + i, range, tensor::row{});
                auto X_view =
                  tensor::get_view<dimension_decrease>(std::forward<TensorViewX>(X), 0, range, tensor::row{});
                auto Y_view =
                  tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), order - i, range, tensor::row{});
                c1 = build_c1(std::forward<decltype(X_view)>(X_view), std::forward<decltype(Y_view)>(Y_view), order,
                              far_field, n, m);
            }
            return c;
        }
    };

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam SavedDimension
     */
    template<typename ValueType, std::size_t SavedDimension>
    struct build<ValueType, 1, SavedDimension>
    {
        using value_type = ValueType;

        /**
         * @brief
         *
         * @tparam TensorViewX
         * @tparam TensorViewY
         * @tparam FarField
         * @param X
         * @param Y
         * @param order
         * @param far_field
         * @param n
         * @param m
         * @return auto
         */
        template<typename TensorViewX, typename TensorViewY, typename FarField>
        [[nodiscard]] inline auto operator()(TensorViewX&& X, TensorViewY&& Y, std::size_t order,
                                             FarField const& far_field, std::size_t n, std::size_t m)
        {
            static constexpr std::size_t km = FarField::km;
            std::size_t ntilde = 2 * order - 1;
            xt::xarray<value_type> c(std::vector(1, std::size_t(ntilde)));

            auto c1_column = xt::view(c, xt::range(0, order));
            for(std::size_t i = 0; i < order; ++i)
            {
                c1_column(i) =
                  far_field.evaluate(std::forward<TensorViewX>(X)(i), std::forward<TensorViewY>(Y)(0)).at(n * km + m);
            }

            auto c1_row = xt::view(c, xt::range(order, ntilde));
            for(std::size_t i = 1; i < order; ++i)
            {
                c1_row(i - 1) =
                  far_field.evaluate(std::forward<TensorViewX>(X)(0), std::forward<TensorViewY>(Y)(order - i))
                    .at(n * km + m);
            }
            return c;
        }
    };
}   // namespace scalfmm::interpolation

#endif   // SCALFMM_INTERPOLATION_GENERATE_CIRCULENT_HPP
