// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/spherical/utils.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_INTERPOLATION_UTILS_HPP
#define SCALFMM_OPERATORS_INTERPOLATION_UTILS_HPP

#include "scalfmm/meta/utils.hpp"

#include <array>
#include <utility>

namespace scalfmm::operators::utils
{
    namespace impl
    {
        /**
         * @brief
         *
         * @tparam Container
         * @tparam ISs
         * @param s
         * @param cont
         * @param index
         * @return constexpr auto
         */
        template<typename Container, std::size_t... ISs>
        constexpr inline auto generate_s(std::index_sequence<ISs...> s, Container const& cont,
                                         std::array<std::size_t, sizeof...(ISs)> const& index)
        {
            return meta::multiply(std::get<ISs>(cont[index[ISs]])...);
        }

        /**
         * @brief
         *
         */
        constexpr auto erase = [](auto pol, auto der, std::size_t i) constexpr
        {
            pol.at(i) = der.at(i);
            return pol;
        };

        /**
         * @brief
         *
         * @tparam T
         * @tparam Is
         * @param der
         * @param s
         * @return constexpr auto
         */
        template<typename T, std::size_t... Is>
        inline constexpr auto multiply_der(T&& der, std::index_sequence<Is...> s)
        {
            return meta::multiply(meta::get<Is>(der)...);
        }

        /**
         * @brief
         *
         * @tparam T
         * @param der
         * @return constexpr auto
         */
        template<typename T>
        inline constexpr auto multiply_der(T&& der)
        {
            return impl::multiply_der(std::forward<T>(der),
                                      std::make_index_sequence<meta::tuple_size<std::decay_t<T>>::value>{});
        }

        /**
         * @brief
         *
         * @tparam Container
         * @tparam ISs
         * @param s
         * @param pol
         * @param der
         * @param index
         * @return constexpr auto
         */
        template<typename Container, std::size_t... ISs>
        constexpr inline auto generate_der_s(std::index_sequence<ISs...> s, Container const& pol, Container const& der,
                                             std::array<std::size_t, sizeof...(ISs)> const& index)
        {
            using value_type = typename Container::value_type;
            constexpr auto dimension = value_type::dimension;
            // the polynome and the derivative value at the current loop index
            value_type pol_val{meta::get<ISs>(pol.at(index.at(ISs)))...};
            value_type der_val{meta::get<ISs>(der.at(index.at(ISs)))...};
            value_type res{};
            // we need an array of size dimension to calculate px_d
            //  const FReal PX = dL_of_x[index.at(0)][0] * L_of_x[index.at(1)][1] * L_of_x[index.at(2)][2];
            //  const FReal PY = L_of_x[index.at(0)][0] *  dL_of_x[index.at(1)][1] * L_of_x[index.at(2)][2];
            //  const FReal PZ = L_of_x[index.at(0)][0] *  L_of_x[index.at(1)][1] * dL_of_x[index.at(2)][2];
            //  each element of the array corespond to PX, PY, PZ
            //  then we fold it with the multiply operator.
            std::array<value_type, sizeof...(ISs)> to_folds{};

            // Here we erase the L_of_x[] with the dL_of_x value
            meta::noop_t{(meta::get<ISs>(to_folds) = erase(pol_val, der_val, ISs), 0)...};
            //Then we fold to obtain PX, PY, PZ
            meta::noop_t{(meta::get<ISs>(res) = multiply_der(to_folds.at(ISs)), 0)...};
            return res;
        }

    }   // namespace impl

    /**
     * @brief
     *
     * @tparam N
     * @tparam Container
     * @param cont
     * @param index
     * @return constexpr auto
     */
    template<std::size_t N, typename Container>
    constexpr inline auto generate_s(Container const& cont, std::array<std::size_t, N> const& index)
    {
        return impl::generate_s(std::make_index_sequence<N>{}, cont, index);
    }

    /**
     * @brief
     *
     * @tparam N
     * @tparam Container
     * @param cont
     * @param der
     * @param index
     * @return constexpr auto
     */
    template<std::size_t N, typename Container>
    constexpr inline auto generate_der_s(Container const& cont, Container const& der,
                                         std::array<std::size_t, N> const& index)
    {
        return impl::generate_der_s(std::make_index_sequence<N>{}, cont, der, index);
    }
}   // namespace scalfmm::operators::utils

#endif   // SCALFMM_OPERATORS_INTERPOLATION_UTILS_HPP
