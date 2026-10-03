#ifndef INTEGER_SEQUENCE_POLYFILL_HPP__
#define INTEGER_SEQUENCE_POLYFILL_HPP__

/**
 * \file
 * \brief Integer sequence type
 *
 * \author Quentin Khan
 *
 * This file implements the c++14 std::integer_sequence class for C++11
 * projects. It also defines a formatted output operator for the sequences.
 */


#if __cplusplus < 201103L
//#error "The definition of integer_sequence needs at least c++11."
#endif


#if __cplusplus >= 201402L

#include <utility>

namespace inria {
    using std::integer_sequence;
    using std::make_integer_sequence;
    using std::index_sequence;
    using std::make_index_sequence;
    using std::index_sequence_for;
}

#else

#include <cstddef> // for std::size_t

namespace inria {

    /**
     * \brief Recursive definition of the integer_sequence type
     *
     * \author Quentin Khan
     *
     * This structure allows using template parameter deduction to expand variadic
     * template parameter packs using indexes. One such use is to apply a function
     * to every element in a std::tuple. Another use case is to extract all elements
     * from a tuple to a function parameters.
     *
     * Use \link make_integer_sequence \endlink to create an object instance.
     *
     * Example:
     * ~~~~{cpp}
     * // The f function takes a variadic count of arguments.
     * template<class... Args>
     * double f(const Args&... args);
     *
     * // The helper calls the max function with the expanded tuple
     * template<class... Args, std::size_t... Indexes>
     * double apply_f_helper(const std::tuple<Args...>& tuple, inria::integer_sequence<Indexes...>) {
     *      // std::get<Indexes>(tuple)... is expanded into 'std::get<0>(tuple), std::get<1>(tuple), ...'
     *      return f( std::get<Indexes>(tuple)... );
     * }
     *
     * template<class... Args>
     * double apply_f(const std::tuple<Args...>& tuple) {
     *     apply_f_helper(tuple, inria::index_sequence_for<Args>());
     * }
     * ~~~~
     *
     * \tparam T Underlying type for the sequence
     * \tparam Ints Sequence of integers of type T
     */
    template<typename T, T... Ints>
    class integer_sequence {
        /**
         * \brief integer_sequence length
         * \return The interger count int the integer_sequence
         */
        static constexpr std::size_t size() noexcept {
            return sizeof...(Ints);
        }
    };

    /**
     * \brief Shorthand for integer_sequence of type std::size_t
     */
    template<std::size_t... Ints>
    using index_sequence = integer_sequence<std::size_t, Ints...>;

    namespace detail {

        /**
         * \brief Recursive definition of the integer_sequence type
         * \copydetails integer_sequence
         */
        template<typename T, std::size_t I, T... Is>
        struct integer_sequence_impl {
            using type = typename integer_sequence_impl<T, I-1, static_cast<T>(I-1), Is...>::type;
        };

        /**
         * \brief Recursive definition end of the integer_sequence type
         * \copydetails integer_sequence
         */
        template<typename T, T... Is>
        struct integer_sequence_impl<T, 0, Is...> {
            using type = integer_sequence<T, Is...>;
        };

        /**
         * \brief Implementation of make_integer_sequence
         */
        template<typename T, T N>
        struct _make_integer_sequence {
            static_assert(N >= 0, "Cannot create integer sequence of negative length.");
            using type = typename integer_sequence_impl<T, static_cast<T>(N)>::type;
        };

    }

    /**
     * \brief integer_sequence<T, 0, ..., N-1> type.
     *
     * \tparam T Base integer type.
     * \tparam N Suquence upper bound.
     *
     * Example:
     * ~~~{cpp}
     * auto is = inria::make_integer_sequence<int, 10>();
     * // `is` is of type `inria::index_sequence<int,0,1,2,3,4,5,6,7,8,9>`
     * ~~~
     */
    template<class T, T N>
    using make_integer_sequence = typename detail::_make_integer_sequence<T,N>::type;

    /**
     * \brief index_sequence with std::size_t as a base type.
     *
     * \tparam N Sequence upper bound.
     *
     * Example:
     * ~~~{cpp}
     * auto is = inria::make_index_sequence<10>();
     * // `is` is of type `inria::index_sequence<0,1,2,3,4,5,6,7,8,9>`
     * ~~~
     */
    template<std::size_t N>
    using make_index_sequence = make_integer_sequence<std::size_t, N>;

    /**
     * \brief index_sequence type corresponding to a type pack.
     *
     * \tparam Ts Parameter pack to index
     *
     * Example:
     * ~~~{cpp}
     * auto is = inria::index_sequence_for<int, double, double>();
     * // `is` is of type `inria::index_sequence<0,1,2>`
     * ~~~
     */
    template<class... Ts>
    using index_sequence_for = make_index_sequence<sizeof...(Ts)>;


}

#endif

#include <ostream>

namespace inria {
    /**
     * \brief Output an index sequence
     *
     * \tparam T Index sequence base integer type
     * \tparam Ints Index sequence
     *
     * \param os Output stream
     * \param unnamed The index sequence object does not store information
     *
     * \return A reference to the output stream
     */
    template<typename T, T... Ints>
    std::ostream& operator<<(std::ostream& os, integer_sequence<T, Ints...>) {
        auto l = {0, ((os << Ints << (Ints != sizeof...(Ints)-1 ? " " : "")),0) ...};
        (void)l;
        return os;
    }
}




#endif // INTEGER_SEQUENCE_POLYFILL_HPP__
