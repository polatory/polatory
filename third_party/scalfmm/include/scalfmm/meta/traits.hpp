//==============================================================================
// This file is a test for a clean scalfmm code.
// It's a attempt to a cleaner and more expressive code for the user.
// Author : Pierre Esterie
//==============================================================================
// --------------------------------
// See LICENCE file at project root
// File : scalfmm/meta/traits.hpp
// -------------------------------
#ifndef SCALFMM_META_TRAITS_HPP
#define SCALFMM_META_TRAITS_HPP

#include "scalfmm/meta/forward.hpp"
#include "scalfmm/meta/is_valid.hpp"

#include "xsimd/xsimd.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace scalfmm
{
    namespace container
    {
        /**
         * @brief Forward declaration for 'particle'.
         *
         * @tparam PositionType
         * @tparam PositionDim
         * @tparam InputsType
         * @tparam NInputs
         * @tparam OutputsType
         * @tparam MOutputs
         * @tparam Variables
         */
        template<typename PositionType, std::size_t PositionDim, typename InputsType, std::size_t NInputs,
                 typename OutputsType, std::size_t MOutputs, typename... Variables>
        struct particle;
    }   // namespace container
}   // namespace scalfmm

namespace xsimd
{
    /**
     * @brief Forward declaration for 'batch'.
     *
     * @tparam T
     * @tparam A
     */
    template<class T, class A>
    class batch;
}   // namespace xsimd

// Traits
namespace scalfmm::meta
{
    /**
     * @brief
     *
     * @tparam Whatever
     * @tparam T
     * @param t
     * @return constexpr auto
     */
    template<std::size_t Whatever, typename T>
    inline constexpr auto id(T&& t)
    {
        return std::forward<T>(t);
    }

    /**
     * @brief
     *
     * @tparam T
     * @tparam Trait
     * @return constexpr auto
     */
    template<typename T, template<typename> class Trait>
    inline constexpr auto delayed_trait(std::true_type)
    {
        return typename Trait<T>::type{};
    }

    /**
     * @brief
     *
     */
    struct foo
    {
    };

    /**
     * @brief
     *
     * @tparam T
     * @tparam Trait
     * @param b
     * @return constexpr auto
     */
    template<typename T, template<typename> class Trait>
    inline constexpr auto delayed_trait(std::false_type b)
    {
        return foo{};
    }

    /**
     * @brief
     *
     */
    inline constexpr auto is_equality_comparable_f = meta::is_valid([](auto&& a, auto&& b) -> decltype(a == b) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_begin_f = meta::is_valid([](auto&& a) -> decltype(a.begin()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_cbegin_f = meta::is_valid([](auto&& a) -> decltype(a.cbegin()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_rbegin_f = meta::is_valid([](auto&& a) -> decltype(a.rbegin()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_crbegin_f = meta::is_valid([](auto&& a) -> decltype(a.crbegin()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_end_f = meta::is_valid([](auto&& a) -> decltype(a.end()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_cend_f = meta::is_valid([](auto&& a) -> decltype(a.cend()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_rend_f = meta::is_valid([](auto&& a) -> decltype(a.rend()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_crend_f = meta::is_valid([](auto&& a) -> decltype(a.crend()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_plus_f = meta::is_valid([](auto&& a, auto&& b) -> decltype(a + b) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_size_func_f = meta::is_valid([](auto t) -> decltype(t.size()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_max_size_func_f = meta::is_valid([](auto t) -> decltype(t.max_size()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_empty_func_f = meta::is_valid([](auto t) -> decltype(t.empty()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_resize_func_f = meta::is_valid([](auto t, auto count) -> decltype(t.resize(count)) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_resize_valued_func_f =
      meta::is_valid([](auto t, auto count, auto value) -> decltype(t.resize(count, value)) {});

    /**
     * @brief
     *
     */
    inline constexpr auto has_clear_func_f = meta::is_valid([](auto t) -> decltype(t.clear()) {});

    // Containers related traits

    /**
     * @brief
     *
     * @tparam T
     */
    template<class T>
    struct has_begin
    {
        using type = decltype(std::declval<std::decay_t<T>>().begin());
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<class T>
    struct has_cbegin
    {
        using type = decltype(std::declval<std::decay_t<T>>().cbegin());
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<class T>
    struct has_rbegin
    {
        using type = decltype(std::declval<std::decay_t<T>>().rbegin());
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<class T>
    struct has_crbegin
    {
        using type = decltype(std::declval<std::decay_t<T>>().crbegin());
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_end
    {
        using type = decltype(std::declval<std::decay_t<T>>().end());
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_cend
    {
        using type = decltype(std::declval<std::decay_t<T>>().cend());
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_rend
    {
        using type = decltype(std::declval<std::decay_t<T>>().rend());
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_crend
    {
        using type = decltype(std::declval<std::decay_t<T>>().crend());
    };

    /**
    * @brief
    *
    * @tparam T
    */
    template<class T>
    struct has_range_interface
    {
        static const bool value = has_begin<T>::value && has_end<T>::value;
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_size_func
    {
        using type = decltype(std::declval<std::decay_t<T>>().size());
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_empty_func
    {
        using type = decltype(std::declval<std::decay_t<T>>().empty());
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_max_size_func
    {
        using type = decltype(std::declval<std::decay_t<T>>().max_size());
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_resize_func
    {
        using type = decltype(std::declval<std::decay_t<T>>().resize(typename T::size_type{}));
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_resize_valued_func
    {
        using type =
          decltype(std::declval<std::decay_t<T>>().resize(typename T::size_type{}, typename T::value_type{}));
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_clear_func
    {
        using type = decltype(std::declval<std::decay_t<T>>().clear());
    };

    // Type version

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_begin_t = decltype(std::declval<std::decay_t<T>>().begin());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_cbegin_t = decltype(std::declval<std::decay_t<T>>().cbegin());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_rbegin_t = decltype(std::declval<std::decay_t<T>>().rbegin());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_crbegin_t = decltype(std::declval<std::decay_t<T>>().crbegin());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_end_t = decltype(std::declval<std::decay_t<T>>().end());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_cend_t = decltype(std::declval<std::decay_t<T>>().cend());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_rend_t = decltype(std::declval<std::decay_t<T>>().rend());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_crend_t = decltype(std::declval<std::decay_t<T>>().crend());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_size_func_t = decltype(std::declval<std::decay_t<T>>().size());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_empty_func_t = decltype(std::declval<std::decay_t<T>>().empty());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_max_size_func_t = decltype(std::declval<std::decay_t<T>>().max_size());

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_resize_func_t = decltype(std::declval<std::decay_t<T>>().resize(typename T::size_type{}));

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_resize_valued_func_t =
      decltype(std::declval<std::decay_t<T>>().resize(typename T::size_type{}, typename T::value_type{}));

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_clear_func_t = decltype(std::declval<std::decay_t<T>>().clear());

    // Value version

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_begin_v = decltype(has_begin_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_cbegin_v = decltype(has_cbegin_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_rbegin_v = decltype(has_rbegin_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_crbegin_v = decltype(has_crbegin_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_end_v = decltype(has_end_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_cend_v = decltype(has_cend_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_rend_v = decltype(has_rend_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_crend_v = decltype(has_crend_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_range_interface_v = has_range_interface<T>::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_size_func_v = decltype(has_size_func_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_empty_func_v = decltype(has_empty_func_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_max_size_func_v = decltype(has_max_size_func_f(T{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_resize_func_v = decltype(has_resize_func_f(T{}, typename T::size_type{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_resize_valued_func_v =
      decltype(has_resize_valued_func_f(T{}, typename T::size_type{}, typename T::value_type{}))::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_clear_func_v = decltype(has_clear_func_f(T{}))::value;

    // Other traits

    /**
     * @brief
     *
     * @tparam T
     */
    template<class T>
    struct is_equality_comparable
    {
        static const bool value = decltype(is_equality_comparable_f(T{}, T{}))::value;
    };

    /**
     * @brief check if the type is float (std::complex) or not
     *
     * @tparam T
     */
    template<class, class = std::void_t<>>
    struct is_float : std::false_type
    {
    };

    /**
     * @brief specialization recognizes types that do have a nested ::type member:
     *
     * @tparam T
     */
    template<class T>
    struct is_float<T, std::void_t<float>> : std::true_type
    {
    };

    /**
     * @brief check if the type is double (std::complex) or not
     *
     * @tparam T
     */
    template<class, class = std::void_t<>>
    struct is_double : std::false_type
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<class T>
    struct is_double<T, std::void_t<double>> : std::true_type
    {
    };

    /**
     * @brief  check if the type is complex (std::complex) or not
     *
     * @tparam T
     */
    template<typename, typename = std::void_t<>>
    struct is_complex : std::false_type
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_complex<std::complex<T>, std::void_t<std::complex<T>>> : std::true_type
    {
    };

    /**
     * @brief return true if the type is complex
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool is_complex_v = is_complex<T>::value;

    /**
     * @brief Tuple cat
     *
     * @tparam T
     * @tparam U
     */
    template<typename T, typename U>
    struct tuple_cat
    {
        using type = decltype(std::tuple_cat(T{}, U{}));
    };

    /**
     * @brief Check if the type T has a member type value_type
     *
     * \code
     *
     * \endcode
     * @tparam T
     */
    template<typename T, typename = std::void_t<>>
    struct has_value_type : std::false_type
    {
        using type = T;
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct has_value_type<T, std::void_t<typename T::value_type>> : std::true_type
    {
        using type = typename T::value_type;
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline constexpr bool has_value_type_v = has_value_type<T>::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using has_value_type_t = typename has_value_type<T>::type;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_addable
    {
        static const bool value = decltype(has_plus_f(T{}, T{}))::value;
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_simd : xsimd::is_batch<T>
    {
    };

    /**
     * @brief Tree related type traits.
     *
     * @tparam ypename
     * @tparam typename
     */
    template<typename, typename = std::void_t<>>
    struct is_leaf_group_symbolics : std::false_type
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_leaf_group_symbolics<T, std::void_t<typename T::group_leaf_type>> : std::true_type
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_arithmetic : std::is_arithmetic<T>
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_integral : std::is_integral<T>
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_arithmetic<xsimd::batch<T>> : is_arithmetic<T>
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_integral<xsimd::batch<T>> : is_integral<T>
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_tuple : std::false_type
    {
    };

    /**
     * @brief
     *
     * @tparam Ts
     */
    template<typename... Ts>
    struct is_tuple<std::tuple<Ts...>> : std::true_type
    {
    };

    /**
     * @brief
     *
     * @tparam Ts
     */
    template<typename... Ts>
    inline constexpr bool is_tuple_v = is_tuple<Ts...>::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_array : std::false_type
    {
    };

    /**
     * @brief
     *
     * @tparam T
     * @tparam N
     */
    template<typename T, std::size_t N>
    struct is_array<std::array<T, N>> : std::true_type
    {
    };

    /**
     * @brief
     *
     * @tparam Ts
     */
    template<typename... Ts>
    inline constexpr bool is_array_v = is_array<Ts...>::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct is_particle : std::false_type
    {
    };

    /**
     * @brief
     *
     * @tparam PositionType
     * @tparam PositionDim
     * @tparam InputsType
     * @tparam NInputs
     * @tparam OutputsType
     * @tparam MOutputs
     * @tparam Variables
     */
    template<typename PositionType, std::size_t PositionDim, typename InputsType, std::size_t NInputs,
             typename OutputsType, std::size_t MOutputs, typename... Variables>
    struct is_particle<
      container::particle<PositionType, PositionDim, InputsType, NInputs, OutputsType, MOutputs, Variables...>>
      : std::true_type
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    inline static constexpr bool is_particle_v = is_particle<T>::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename MatrixKernelType, typename = void>
    struct is_smooth : std::false_type
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename MatrixKernelType>
    struct is_smooth<MatrixKernelType, std::void_t<decltype(MatrixKernelType::is_smooth)>> : std::true_type
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename MatrixKernelType>
    inline static constexpr bool is_smooth_v = is_smooth<MatrixKernelType>::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename MatrixKernelType, typename PointType, typename = void>
    struct has_self_interaction : std::false_type
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename MatrixKernelType, typename PointType>
    struct has_self_interaction<
      MatrixKernelType, PointType,
      std::void_t<decltype(std::declval<MatrixKernelType>().evaluate(std::decay_t<PointType>{}))>> : std::true_type
    {
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename MatrixKernelType, typename PointType>
    inline static constexpr bool has_self_interaction_v = has_self_interaction<MatrixKernelType, PointType>::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct td;

    // interpolator

    /**
     * @brief
     *
     */
    inline constexpr auto sig_preprocess_f = meta::is_valid(
      [](auto&& inter, auto& cell, std::size_t th) -> decltype(inter.apply_multipoles_preprocessing_impl(cell, th)) {});

    /**
     * @brief
     *
     */
    inline constexpr auto sig_postprocess_f =
      meta::is_valid([](auto&& inter, auto& cell, auto& buffer,
                        std::size_t th) -> decltype(inter.apply_multipoles_postprocessing_impl(cell, buffer, th)) {});

    /**
     * @brief
     *
     */
    inline constexpr auto sig_buffer_init_f =
      meta::is_valid([](auto&& inter) -> decltype(inter.buffer_initialization_impl()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto sig_buffer_shape_f =
      meta::is_valid([](auto&& inter) -> decltype(inter.buffer_shape_impl()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto sig_buffer_reset_f =
      meta::is_valid([](auto&& inter, auto& buf) -> decltype(inter.buffer_reset_impl(buf)) {});

    /**
     * @brief
     *
     */
    inline constexpr auto sig_init_k_f = meta::is_valid([](auto&& inter) -> decltype(inter.initialize_k_impl()) {});

    /**
     * @brief
     *
     */
    inline constexpr auto sig_gen_k_f =
      meta::is_valid([](auto&& inter, auto&& X, auto&& Y, std::size_t n, std::size_t m,
                        std::size_t th) -> decltype(inter.generate_matrix_k_impl(X, Y, n, m, th)) {});

    /**
     * @brief
     *
     */
    inline constexpr auto sig_gen_w_f =
      meta::is_valid([](auto&& inter, std::size_t order) -> decltype(inter.generate_weights_impl(order)) {});

    /**
     * @brief Internal structure used for interaction list in source target simulation
     *
     * @tparam T
     */
    template<typename T>
    struct inject
    {
    };

    /**
     * @brief
     *
     * @tparam T
     * @tparam typename
     */
    template<typename T, typename = void>
    struct exist : std::false_type
    {
        using type = void;
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct exist<T, std::void_t<typename T::type>> : std::true_type
    {
        using type = typename T::type;
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    static constexpr bool exist_v = exist<T>::value;

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    using exist_t = typename exist<T>::type;

}   // namespace scalfmm::meta

#endif
