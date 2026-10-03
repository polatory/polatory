// --------------------------------
// See LICENCE file at project root
// File : options/options.hpp
// --------------------------------
#ifndef SCALFMM_OPTIONS_OPTIONS_HPP
#define SCALFMM_OPTIONS_OPTIONS_HPP

#include <sstream>
#include <string>
#include <string_view>
#include <tuple>

#include "scalfmm/meta/utils.hpp"

namespace scalfmm::options
{
    /**
     * @brief Base template for defining a setting.
     *
     * This template provides a `value()` method to retrieve the string representation
     * of the derived setting. It uses CRTP (Curiously Recurring Template Pattern) to
     * delegate to the derived class's implementation.
     *
     * @tparam D The derived setting type.
     */
    template<typename D>
    struct setting
    {
        using type = setting<D>;
        using derived_type = D;

        /**
         * @brief Retrieves the string representation of the setting.
         *
         * Delegates to the `value()` method of the derived class.
         *
         * @return A string view representing the setting.
         */
        constexpr auto value() const noexcept -> std::string_view { return static_cast<derived_type>(this)->value(); }
    };

    /**
     * @brief Represents a collection of multiple settings.
     *
     * This struct uses variadic inheritance to combine multiple `setting` instances
     * into a single `settings` object.
     *
     * @tparam S Variadic template representing the types of the settings.
     */
    template<typename... S>
    struct settings : S...
    {
        using type = settings<S...>;

        /**
         * @brief Constructs a settings object.
         *
         * @param s Variadic arguments representing the settings.
         */
        template<typename... Setting>
        constexpr settings(Setting... s)
        {
        }
    };

    /**
     * @brief Deduction guide for `settings`.
     *
     * Automatically deduces the types of the settings when `settings` is instantiated.
     */
    template<typename... S>
    settings(S... s) -> settings<typename S::type...>;

    /**
     * @brief A specific option for the 'dense' case (the M2L interaction matrices are fully stored).
     */
    struct dense_ : setting<dense_>
    {
        using type = dense_;
        static constexpr auto value() noexcept -> std::string_view { return "dense"; };
    };

    /**
     * @brief A specific option for the 'fft' case (this enable FFT-based optimization for the M2L pass).
     */
    struct fft_ : setting<fft_>
    {
        using type = fft_;
        static constexpr auto value() noexcept -> std::string_view { return "fft"; };
    };

    /**
     * @brief A specific option for the 'low-rank' case (the M2L interaction matrices are compressed).
     */
    struct low_rank_ : setting<low_rank_>
    {
        using type = low_rank_;
        static constexpr auto value() noexcept -> std::string_view { return "low_rank"; };
    };

    /**
     * @brief A specific compound option for the 'sequential' case that combines multiple settings.
     *
     * This option is formed by variadic inheritance of the specified settings.
     *
     * @tparam S Variadic template representing the types of the settings.
     */
    template<typename... S>
    struct seq_
      : setting<seq_<S...>>
      , S...
    {
        using type = seq_<S...>;
        using inner_settings = settings<S...>;
        static constexpr auto value() noexcept -> std::string_view { return "seq"; };
    };
#ifdef _OPENMP

    /**
     * @brief A specific compound option for the 'OpenMP-based' case that combines multiple settings.
     *
     * This option is formed by variadic inheritance of the specified settings.
     *
     * @tparam S Variadic template representing the types of the settings.
     */
    template<typename... S>
    struct omp_
      : setting<omp_<S...>>
      , S...
    {
        using type = omp_<S...>;
        using inner_settings = settings<S...>;
        static constexpr auto value() noexcept -> std::string_view { return "omp"; };
    };
#endif
    /**
     * @brief A specific option to time each pass of the FMM algorithm.
     */
    struct timit_ : setting<timit_>
    {
        using type = timit_;
        static constexpr auto value() noexcept -> std::string_view { return "timit"; };
    };

    /**
     * @brief A specific compound option for the 'Uniform interpolation' that extends another setting.
     *
     * This option is a combination of its own identity (Uniform interpolation) and another setting
     * (e.g., dense_, low_rank_, fft_). Examples of possible combinations are 'Uniform + dense',
     * 'Uniform + low-rank', 'Uniform + fft'.
     *
     * @tparam S The base setting type (default is `fft_`).
     */
    template<typename S = fft_>
    struct uniform_
      : setting<uniform_<S>>
      , S
    {
        using type = uniform_<S>;
        static constexpr auto value() noexcept -> std::string_view { return "uniform_"; };
    };

    /**
     * @brief A specific compound option for the 'Chebyshev interpolation' that extends another setting.
     *
     * This option is a combination of its own identity (Chebyshev interpolation) and another setting
     * (e.g., dense_, low_rank_, fft_). Examples of possible combinations are 'Chebyshev + dense',
     * 'Chebyshev + low-rank'.
     *
     * @tparam S The base setting type (default is `low_rank_`).
     */
    template<typename S = low_rank_>
    struct chebyshev_
      : setting<chebyshev_<S>>
      , S
    {
        using type = chebyshev_<S>;
        static constexpr auto value() noexcept -> std::string_view { return "chebyshev_"; };
    };

    /**
     * @brief A specific compound option for the 'Barycentric interpolation' that extends another setting.
     *
     * This option is a combination of its own identity (Barycentric interpolation) and another setting
     * (e.g., low_rank_). Examples of possible combinations are 'Barycentric + dense',
     * 'Barycentric + low-rank'.
     *
     * @tparam S The base setting type (default is `low_rank_`).
     */
    template<typename S = low_rank_>
    struct barycentric_
      : setting<barycentric_<S>>
      , S
    {
        using type = barycentric_<S>;
        static constexpr auto value() noexcept -> std::string_view { return "barycentric_"; };
    };

    struct modified_uniform_ : setting<modified_uniform_>
    {
        using type = modified_uniform_;
        static constexpr auto value() noexcept -> std::string_view { return "modified_uniform_"; };
    };

    /**
    * @brief Predefined static option for 'Uniform interpolation' with fully stored M2L interaction matrices.
    */
    static constexpr auto uniform_dense = uniform_<dense_>{};

    /**
    * @brief Predefined static option for 'Uniform interpolation' with compressed M2L interaction matrices.
    */
    static constexpr auto uniform_low_rank = uniform_<low_rank_>{};

    /**
    * @brief Predefined static option for 'Uniform interpolation' with FFT-based optimization.
    */
    static constexpr auto uniform_fft = uniform_<fft_>{};

    /**
    * @brief Predefined static option for 'Barycentric interpolation' with fully stored M2L interaction matrices.
    */
    static constexpr auto barycentric_dense = barycentric_<dense_>{};

    /**
    * @brief Predefined static option for 'Barycentric interpolation' with compressed M2L interaction matrices.
    */
    static constexpr auto barycentric_low_rank = barycentric_<low_rank_>{};

    /**
    * @brief Predefined static option for 'Chebyshev-based interpolation' with fully stored M2L interaction matrices.
    */
    static constexpr auto chebyshev_dense = chebyshev_<dense_>{};

    /**
    * @brief Predefined static option for 'Chebyshev-based interpolation' with compressed M2L interaction matrices.
    */
    static constexpr auto chebyshev_low_rank = chebyshev_<low_rank_>{};

    static constexpr auto modified_uniform = modified_uniform_{};

    /**
    * @brief Predefined static option to fully store the M2L interaction matrices.
    */
    static constexpr auto dense = dense_{};

    /**
    * @brief Predefined static option to enable FFT-based optimization for the M2L pass.
    */
    static constexpr auto fft = fft_{};

    /**
    * @brief Predefined static option to compress the M2L interaction matrices.
    */
    static constexpr auto low_rank = low_rank_{};
#ifdef _OPENMP

    /**
    * @brief Predefined static option to use the OpenMP-based algorithm.
    */
    static constexpr auto omp = omp_{};

    /**
    * @brief Predefined static option to use the OpenMP-based algorithm with each pass is timed.
    */
    static constexpr auto omp_timit = omp_<timit_>{};
#endif
    /**
    * @brief Predefined static option to use the sequential algorithm.
    */
    static constexpr auto seq = seq_{};

    /**
    * @brief Predefined static option to use the sequential algorithm with each pass is timed.
    */
    static constexpr auto seq_timit = seq_<timit_>{};

    /**
    * @brief Predefined static option to time each pass of the algorithm.
    */
    static constexpr auto timit = timit_{};

    /**
     * @brief Check if S2 is inside S1
     *
     * Here we check if option op used for the algorithm contains omp.
     *  In other word, we check if the algorithm is the OpenMP one
     * \code {.c++}
     * has(scalfmm::options::_s(op),scalfmm::options::_s(scalfmm::options::omp))
     * \endcode
     *
     * @tparam S1
     * @tparam S2
     *
     * @param s1 the first setting
     * @param s2 the second one
     *
     * @return
     */
    template<typename... S1, typename... S2>
    static constexpr auto has(settings<S1...> s1, settings<S2...> s2) -> bool
    {
        return (... || std::is_base_of_v<S2, decltype(s1)>);
    }

    /**
     * @brief
     *
     * @tparam S1
     * @tparam S
     *
     * @param s1
     * @param s
     *
     * @return
     */
    template<typename... S1, typename... S>
    static constexpr auto has(settings<S1...> s1, S... s) -> bool
    {
        return (... || std::is_base_of_v<S, decltype(s1)>);
    }

    /**
     * @brief
     *
     * @tparam S1
     * @tparam S2
     *
     * @param s1
     * @param s2
     *
     * @return
     */
    template<typename... S1, typename... S2>
    static constexpr auto match(settings<S1...> s1, settings<S2...> s2) -> bool
    {
        return (sizeof...(S1) == sizeof...(S2)) && (... && std::is_base_of_v<S2, decltype(s1)>);
    }

    /**
     * @brief
     *
     * @tparam S1
     * @tparam S
     *
     * @param s1
     * @param s
     *
     * @return
     */
    template<typename... S1, typename... S>
    static constexpr auto match(settings<S1...> s1, S... s) -> bool
    {
        return ((sizeof...(S1) == sizeof...(S)) && (... && std::is_base_of_v<S, decltype(s1)>));
    }

    /**
     * @brief
     *
     * @tparam S1
     * @tparam S2
     *
     * @param s1
     * @param s2
     *
     * @return
     */
    template<typename... S1, typename... S2>
    static constexpr auto support(settings<S1...> s1, settings<S2...> s2) -> bool
    {
        return (... && std::is_base_of_v<S1, decltype(s2)>);
    }

    /**
     * @brief
     *
     * @tparam T
     *
     * @param t
     *
     * @return
     */
    template<typename... S>
    static constexpr auto is_settings(settings<S...> s) -> bool
    {
        return true;
    }

    /**
     * @brief
     *
     * @tparam T
     *
     * @param t
     *
     * @return
     */
    template<typename T>
    static constexpr auto is_settings(T t) -> bool
    {
        return false;
    }

    /**
     * @brief
     *
     * @tparam S
     *
     * @param s
     *
     * @return
     */
    template<typename... S>
    static constexpr auto _s(S... s)
    {
        return settings<S...>{};
    }

    /**
     * @brief
     *
     * @tparam S
     *
     * @param s
     *
     * @return
     */
    template<typename... S>
    static constexpr auto _s(settings<S...> s)
    {
        return s;
    }
}   // namespace scalfmm::options

/**
 * @brief Macro to define an "optioned callee" struct with a functional interface.
 *
 * This macro generates a struct and a corresponding global instance that allows
 * flexible invocation of functions in the `impl` namespace. The generated struct
 * provides three interfaces:
 *
 * 1. **Static Call (`call`)**:
 *    A static method that forwards arguments to the corresponding `impl::<NAME>` function.
 *
 * 2. **Subscript Operator (`operator[]`)**:
 *    Allows binding of settings (`options::settings`) to create a callable lambda.
 *
 * 3. **Function Call Operator (`operator()`)**:
 *    Enables direct invocation of the `impl::<NAME>` function without explicitly binding settings.
 *
 * @param NAME The name of the callee, corresponding to a function in the `impl` namespace.
 *
 * ### Generated Components
 *
 * The macro expands into:
 * - A struct named `<NAME>_` that encapsulates the interfaces for invocation.
 * - A global constant instance named `<NAME>` for easy access.
 *
 * ### Example
 * @code{.cpp}
 * // Declare an optioned callee for the `process` function
 * DECLARE_OPTIONED_CALLEE(fmm)
 *
 * // Usage
 * fmm(tree, fmm_operator); // Direct invocation
 * fmm[options::_s(options::seq)](tree, fmm_operator); // Invocation with settings
 * @endcode
 */
#define DECLARE_OPTIONED_CALLEE(NAME)                                                                                  \
    struct NAME_                                                                                                       \
    {                                                                                                                  \
        template<typename Arg, typename... Args>                                                                       \
        inline static constexpr auto call(Arg&& arg, Args&&... args) noexcept                                          \
        {                                                                                                              \
            return impl::NAME(std::forward<Arg>(arg), std::forward<Args>(args)...);                                    \
        }                                                                                                              \
        template<typename... S>                                                                                        \
        inline auto constexpr operator[](scalfmm::options::settings<S...> s) const noexcept                            \
        {                                                                                                              \
            return [s](auto&&... args) { return NAME_::call(s, std::forward<decltype(args)>(args)...); };              \
        }                                                                                                              \
        template<typename... Args>                                                                                     \
        inline auto constexpr operator()(Args&&... args) const noexcept                                                \
        {                                                                                                              \
            return NAME_::call(std::forward<Args>(args)...);                                                           \
        }                                                                                                              \
    };                                                                                                                 \
    inline const NAME_ NAME = {};

#endif   // SCALFMM_OPTIONS_OPTIONS_HPP
