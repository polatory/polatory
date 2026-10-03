// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/builders.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_BUILDERS_HPP
#define SCALFMM_INTERPOLATION_BUILDERS_HPP

#include "scalfmm/meta/const_functions.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/containers/xcontainer.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xtensor_forward.hpp"

#include <cstddef>
#include <tuple>
#include <utility>

namespace scalfmm::interpolation
{

    /**
     * @brief
     *
     * @tparam td::size_t
     * @tparam T
     * @param t
     * @return constexpr auto
     */
    template<std::size_t, typename T>
    constexpr auto id(T t)
    {
        return std::forward<T>(t);
    }

    /**
     * @brief
     *
     * @tparam Gen
     * @tparam I
     */
    template<typename Gen, std::size_t... I>
    constexpr auto get_generator(Gen&& gen, std::index_sequence<I...> /*unused*/)
    {
        return std::move(xt::stack(std::make_tuple(id<I>(gen)...)));
    }

    /**
     * @brief
     *
     * @tparam Dim
     * @param order
     * @return constexpr auto
     */
    template<std::size_t Dim>
    constexpr auto get(const std::size_t order)
    {
        return get_generator(xt::linspace(std::size_t{0}, order - 1, order), std::make_index_sequence<Dim>{});
    }

    // TODO for nodes
    // auto lin = xt::stack(xt::xtuple(xt::linspace(std::size_t{0},order-1,order)));
    // xt::xarray<std::size_t> saved;
    // for(std::size_t i=0; i<dimension; ++i)
    //{
    //  auto r = xt::stack(xt::xtuple(xt::ones<std::size_t>({meta::pow(order,(dimension-1-i))})));
    //  auto tmp = xt::linalg::kron(lin,r);
    //  auto l = xt::stack(xt::xtuple(xt::ones<std::size_t>({meta::pow(order,i)})));
    //  auto reshaped = xt::linalg::kron(l,tmp);
    //  reshaped.reshape({1,-1});
    //  auto res = xt::stack(xt::xtuple(reshaped,saved));
    //  saved = res;
    //}
    //  std::cout << "ids: " << saved << '\n';

    /**
     * @brief
     *
     * @tparam Dim
     */
    template<std::size_t Dim>
    struct nodes
    {
        static_assert(Dim < 4, "Dimension for interpolation node not supported.");
    };

    /**
     * @brief
     *
     * @tparam
     */
    template<>
    struct nodes<1>
    {
        /**
         * @brief
         *
         * @param order
         * @return auto
         */
        inline static auto get(std::size_t order) { return xt::linspace(std::size_t{0}, order - 1, order); }
    };

    /**
     * @brief
     *
     * @tparam
     */
    template<>
    struct nodes<2>
    {
        /**
         * @brief
         *
         * @param order
         * @return auto
         */
        inline static auto get(std::size_t order)
        {
            const std::size_t s{meta::pow(order, std::size_t(2))};
            const xt::xarray<std::size_t>::shape_type shape{std::size_t(2), std::size_t(s)};
            xt::xarray<std::size_t> ns(shape);
            for(std::size_t i = 0; i < s; ++i)
            {
                ns(0, i) = i % order;
                ns(1, i) = (i / order) % order;
            }
            return ns;
        }
    };

    /**
     * @brief
     *
     * @tparam
     */
    template<>
    struct nodes<3>
    {
        /**
         * @brief
         *
         * @param order
         * @return auto
         */
        inline static auto get(std::size_t order)
        {
            const auto s{meta::pow(order, 3)};
            const xt::xarray<std::size_t>::shape_type shape{3, s};
            xt::xarray<std::size_t> ns(shape);
            for(std::size_t i = 0; i < s; ++i)
            {
                ns(0, i) = i % order;
                ns(1, i) = (i / order) % order;
                ns(2, i) = i / (order * order);
            }
            return ns;
        }
    };

    /**
     * @brief
     *
     * @tparam
     */
    template<>
    struct nodes<4>
    {
        /**
         * @brief
         *
         * @param order
         * @return auto
         */
        inline static auto get(std::size_t order)
        {
            const auto s{meta::pow(order, 4)};
            const xt::xarray<std::size_t>::shape_type shape{4, s};
            xt::xarray<std::size_t> ns(shape);
            for(std::size_t i = 0; i < s; ++i)
            {
                ns(0, i) = i % order;
                ns(1, i) = (i / (order)) % order;
                ns(2, i) = (i / (order * order)) % order;
                ns(3, i) = i / (order * order * order);
            }
            return ns;
        }
    };

    // Builders for relative center

}   // namespace scalfmm::interpolation

#endif   // SCALFMM_INTERPOLATION_BUILDERS_HPP
