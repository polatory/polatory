// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/interpolation/utils.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_INTERPOLATION_UTILS_HPP
#define SCALFMM_OPERATORS_INTERPOLATION_UTILS_HPP

#include "scalfmm/container/point.hpp"
#include "scalfmm/meta/utils.hpp"

#include "xsimd/xsimd.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
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
         */
        template<typename Container, std::size_t... ISs>
        constexpr inline auto generate_s(std::index_sequence<ISs...> /*s*/, Container const& cont,
                                         std::array<std::size_t, sizeof...(ISs)> const& index)
        {
            return meta::multiply(std::get<ISs>(cont[index[ISs]])...);
        }

        /**
         * @brief
         *
         * @tparam vec_size
         * @tparam Container
         * @tparam ISs
         */
        template<std::size_t vec_size, typename Container, std::size_t... ISs>
        constexpr inline auto generate_s_simd(std::index_sequence<ISs...> /*s*/, Container const& cont,
                                              std::array<std::size_t, sizeof...(ISs)> const& index)
        {
            return meta::multiply(xsimd::load_aligned(&std::get<ISs>(cont)[index[ISs] * vec_size])...);
        }

        /**
         * @brief
         *
         * @tparam T
         * @tparam Is
         */
        template<typename T, std::size_t... Is>
        inline constexpr auto multiply_der(T der, std::index_sequence<Is...> /*s*/)
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
        inline constexpr auto multiply_der(T der)
        {
            return multiply_der(der, std::make_index_sequence<meta::tuple_size<std::decay_t<T>>::value>{});
        }

        /**
         * @brief
         *
         * @tparam Container
         * @tparam ISs
         */
        template<typename Container, std::size_t... ISs>
        constexpr inline auto generate_der_s(std::index_sequence<ISs...> /*s*/, Container const& pol,
                                             Container const& der, std::array<std::size_t, sizeof...(ISs)> const& index)
        {
            auto erase = [](auto pol, auto der, std::size_t i)
            {
                pol[i] = der[i];
                return pol;
            };
            using value_type = typename Container::value_type;
            // the polynome and the derivative value at the current loop index
            value_type pol_val{meta::get<ISs>(pol[index[ISs]])...};
            value_type der_val{meta::get<ISs>(der[index[ISs]])...};
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
            // Then we fold to obtain PX, PY, PZ
            meta::noop_t{(meta::get<ISs>(res) = multiply_der(to_folds[ISs]), 0)...};
            return res;
        }

        /**
         * @brief
         *
         * @tparam vec_size
         * @tparam Container
         * @tparam ISs
         */
        template<std::size_t vec_size, typename Container, std::size_t... ISs>
        constexpr inline auto generate_der_s_simd(std::index_sequence<ISs...> /*s*/, Container const& pol,
                                                  Container const& der,
                                                  std::array<std::size_t, sizeof...(ISs)> const& index)
        {
            using particle_type = typename Container::value_type;
            using value_type = typename particle_type::position_value_type;
            constexpr auto dimension = particle_type::dimension;
            using position_type = container::point<xsimd::simd_type<value_type>, dimension>;

            constexpr auto erase = [](auto& res, auto const& pol, auto const& der) constexpr
            {
                if constexpr(dimension == 3)
                {
                    res[0] = der[0] * pol[1] * pol[2];
                    res[1] = pol[0] * der[1] * pol[2];
                    res[2] = pol[0] * pol[1] * der[2];
                }
                else if constexpr(dimension == 2)
                {
                    res[0] = der[0] * pol[1];
                    res[1] = pol[0] * der[1];
                }
                else if constexpr(dimension == 1)
                {
                    res[0] = der[0];
                }
                else
                {
                    static_assert(dimension > 0 && dimension < 4, "Please specifiy the derivative pattern for S !");
                }
            };
            // the polynome and the derivative value at the current loop index
            position_type pol_val{xsimd::load_aligned(&meta::get<ISs>(pol)[index[ISs] * vec_size])...};
            position_type der_val{xsimd::load_aligned(&meta::get<ISs>(der)[index[ISs] * vec_size])...};
            position_type res{};

            // we need an array of size dimension to calculate px_d
            //  const FReal PX = dL_of_x[index.at(0)][0] * L_of_x[index.at(1)][1] * L_of_x[index.at(2)][2];
            //  const FReal PY = L_of_x[index.at(0)][0] *  dL_of_x[index.at(1)][1] * L_of_x[index.at(2)][2];
            //  const FReal PZ = L_of_x[index.at(0)][0] *  L_of_x[index.at(1)][1] * dL_of_x[index.at(2)][2];
            //  each element of the array corespond to PX, PY, PZ
            //  then we fold it with the multiply operator.
            // Here we erase the L_of_x[] with the dL_of_x value
            erase(res, pol_val, der_val);
            return res;
        }

        /**
         * @brief
         *
         * @tparam Container
         * @tparam ISs
         */
        template<typename Container, std::size_t... ISs>
        constexpr inline auto generate_root(std::index_sequence<ISs...>, Container const& cont,
                                            std::array<std::size_t, sizeof...(ISs)> const& index)
        {
            return std::array<typename Container::value_type, sizeof...(ISs)>{cont[index[ISs]]...};
        }

        /**
         * @brief
         *
         * @tparam Container
         * @tparam ISs
         */
        template<typename Container, std::size_t... ISs>
        constexpr inline auto multiply_components_at_indices(std::index_sequence<ISs...> /*s*/, Container const& cont,
                                                             std::array<std::size_t, sizeof...(ISs)> const& index)
        {
            return meta::multiply((cont[index[ISs]])...);
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
     * @tparam vec_size
     * @tparam Container
     * @param cont
     * @param index
     * @return constexpr auto
     */
    template<std::size_t vec_size, typename Container>
    constexpr inline auto
    generate_s_simd(Container const& cont,
                    std::array<std::size_t, meta::tuple_size_v<typename Container::value_type>> const& index)
    {
        return impl::generate_s_simd<vec_size>(
          std::make_index_sequence<meta::tuple_size_v<typename Container::value_type>>{}, cont, index);
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

    /**
     * @brief
     *
     * @tparam vec_size
     * @tparam Container
     * @param cont
     * @param der
     * @param index
     * @return constexpr auto
     */
    template<std::size_t vec_size, typename Container>
    constexpr inline auto
    generate_der_s_simd(Container const& cont, Container const& der,
                        std::array<std::size_t, meta::tuple_size_v<typename Container::value_type>> const& index)
    {
        return impl::generate_der_s_simd<vec_size>(
          std::make_index_sequence<meta::tuple_size_v<typename Container::value_type>>{}, cont, der, index);
    }

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
    constexpr inline auto generate_root(Container const& cont, std::array<std::size_t, N> const& index)
    {
        return impl::generate_root(std::make_index_sequence<N>{}, cont, index);
    }

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
    constexpr inline auto multiply_components_at_indices(Container const& cont, std::array<std::size_t, N> const& index)
    {
        return impl::multiply_components_at_indices(std::make_index_sequence<N>{}, cont, index);
    }
}   // namespace scalfmm::operators::utils

#endif   // SCALFMM_OPERATORS_INTERPOLATION_UTILS_HPP
