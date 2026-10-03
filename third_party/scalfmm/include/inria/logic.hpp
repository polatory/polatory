#ifndef _LOGIC_HPP_
#define _LOGIC_HPP_

#include <type_traits>

namespace inria {

    /**
     * \brief Negate the compile time value of the given type
     *
     * \tparam B Type which value to negate
     */
    template<class B>
    struct negation {
        /// not B::value
        enum {value = !B::value};
    };

    namespace details {
        template<bool...> struct bool_t{};
    }

    /**
     * \brief Conjunction of compile time boolean values
     *
     * \tparam Bs Types which values to multiply
     *
     * This implementation relies on the fact that the conjunction will be true
     * if and only if all values are true. Two types are generated from the
     * values: `bool_t<true, values...>` and `bool_t<values..., true>`. The
     * value is the result of the comparison of these types.
     *
     * This method is a lot faster and size resilient than recursive or
     * constexpr functions.
     */
    template<class... Bs>
    struct conjunction {
        enum {
            value = std::is_same<details::bool_t<true, Bs::value...>,
                                 details::bool_t<Bs::value..., true>>::value
        };
    };

    /**
     * \brief Disjunction of compile time boolean values
     */
    template<class... Bs>
    using disjunction = negation<conjunction<negation<Bs...>>>;


    template<class...Ts>
    using require = typename std::enable_if<inria::conjunction<Ts...>::value>::type;

}

#endif /* _LOGIC_HPP_ */
