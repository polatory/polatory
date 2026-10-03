#ifndef _DETECTION_IDIOM_HPP_
#define _DETECTION_IDIOM_HPP_

#include <type_traits>

/**
 * \file
 * \brief Implementation of the `Library fundamentals TS v2` detection idiom
 */

namespace inria {

    namespace detail {

        /**
         * \brief SFINAE helper
         */
        template<class...>
        using void_t = void;

        /**
         * \brief Implementation of the detection idiom, fallback case
         */
        template<class Default, class VoidIfOK, template<class...> class Op, class... Args>
        struct detected_or_impl {
            using type = Default;
            using value_t = std::false_type;
        };

        /**
         * \brief Implementation of the detection idiom, detected case
         */
        template<class Default, template<class...> class Op, class... Args>
        struct detected_or_impl<Default, void_t<Op<Args...>>, Op, Args...> {
            using type = Op<Args...>;
            using value_t = std::true_type;
        };
    }

    /**
     * \brief Class used by detected_t to indicate failure
     */
    struct nonesuch {
        nonesuch() = delete;
        ~nonesuch() = delete;
        nonesuch(const nonesuch&) = delete;
        void operator=(const nonesuch&) = delete;
    };

    /**
     * \brief Main detection class
     *
     * This class is used to check whether a template `Op` can be instanciated with
     * the given arguments `Args...`. It holds two type definitions: `type` and `value_t`.
     *
     * When Op<Args...> exists:
     *    - `type` is an alias for `Op<Args...>`,
     *    - `value_t` is an alias for std::true_type.
     *
     * When Op<Args...> does not exist:
     *    - `type` is an alias for `Default`,
     *    - `value_t` is an alias for std::false_type.
     *
     * \tparam Default Fallback type in case the detection fails
     * \tparam Op Template which instaciation must be detected
     * \tparam Args `Op` instanciation parameters
     */
    template<class Default, template<class...> class Op, class... Args>
    using detected_or = detail::detected_or_impl<Default, void, Op, Args...>;

    /**
     * \brief Alias for `typename detected_or<nonesuch, Op, Args...>::value_t`
     *
     * \tparam Op Template which instaciation must be detected
     * \tparam Args `Op` instanciation parameters
     */
    template<template<class...> class Op, class... Args>
    using is_detected = typename detected_or<nonesuch, Op, Args...>::value_t;

    /**
     * \brief Alias for `typename detected_or<nonesuch, Op, Args...>::type`
     *
     * \tparam Op Template which instaciation must be detected
     * \tparam Args `Op` instanciation parameters
     */
    template<template<class...> class Op, class... Args>
    using detected_t = typename detected_or<nonesuch, Op, Args...>::type;


    /**
     * \brief Alias for `typename detected_or<Default, Op, Args...>::type`
     *
     * \tparam Default Fallback type in case the detection fails
     * \tparam Op Template which instaciation must be detected
     * \tparam Args `Op` instanciation parameters
     */
    template< class Default, template<class...> class Op, class... Args >
    using detected_or_t = typename detected_or<Default, Op, Args...>::type;

    /**
     * \brief Checks whether `detected_t<Op<Args>>` is `Expected`
     *
     * \tparam Expected Type that is expected to be detected
     * \tparam Op Template which instaciation must be detected
     * \tparam Args `Op` instanciation parameters
     */
    template <class Expected, template<class...> class Op, class... Args>
    using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

    /**
     * \brief Checks whether `detected_t<Op<Args>>` can be converted to `To`
     *
     * \tparam To Type that `detected_t<Op<Args>>` should be convertible to
     * \tparam Op Template which instaciation must be detected
     * \tparam Args `Op` instanciation parameters
     */
    template <class To, template<class...> class Op, class... Args>
    using is_detected_convertible = std::is_convertible<detected_t<Op, Args...>, To>;

    // Add shorthand template variables if C++14 or newer is used
    #if __cplusplus >= 201402L

    /**
     * \brief Shorthand for `is_detected<Op, Args...>::value`
     *
     * \tparam Op Template which instaciation must be detected
     * \tparam Args `Op` instanciation parameters
     */
    template< template<class...> class Op, class... Args >
    constexpr bool is_detected_v = is_detected<Op, Args...>::value;

    /**
     * \brief Shorthand for `is_detected_exact<Expected, Op, Args...>::value`
     *
     * \tparam Expected Type that is expected to be detected
     * \tparam Op Template which instaciation must be detected
     * \tparam Args `Op` instanciation parameters
     */
    template <class Expected, template<class...> class Op, class... Args>
    constexpr bool is_detected_exact_v = is_detected_exact<Expected, Op, Args...>::value;

    /**
     * \brief Shorthand for `is_detected_convertible<To, Op, Args...>::value`
     *
     * \tparam To Type that `detected_t<Op<Args>>` should be convertible to
     * \tparam Op Template which instaciation must be detected
     * \tparam Args `Op` instanciation parameters
     */
    template <class To, template<class...> class Op, class... Args>
    constexpr bool is_detected_convertible_v = is_detected_convertible<To, Op, Args...>::value;

    #endif
}


#endif /* _DETECTION_IDIOM_HPP_ */
