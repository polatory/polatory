#ifndef _META_HPP_
#define _META_HPP_

#include <iterator>
#include <type_traits>

namespace inria {

template<class...>
using void_t = void;

template<bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template<bool B, class T = void>
using disable_if_t = typename std::enable_if<!B, T>::type;


// Helper template to choose utilities overloads ///////////////////////////////
namespace details {

template<bool B, class... Ts>
struct first_true_impl {
    using type = void;
};

template<class T, class... Args>
struct first_true_impl<true, T, Args...> {
    using type = T;
};

template<class T, class U, class... Args>
struct first_true_impl<false, T, U, Args...>
    : first_true_impl<U::value, U, Args...>
{};

} // close namespace [inria]::details

/**
 * \brief Hold the first trait which value is true
 *
 * Each type is expected to hold a compile time static `value`. This structure
 * holds a type definition `type` which is an alias to the first type trait
 * which value is true.
 *
 * If no trait holds true, an error is generated.
 *
 * \tparam T    First type trait to test
 * \tparam Args Next type traits
 *
 * Example:
 *
 * ~~~{.cpp}
 * // Implementation when trait 1 is satisfied
 * template<class T>
 * void foo_impl(const T&, trait_1) {
 *     // impl...
 * }
 *
 * // Implementation when trait 2 is satisfied
 * template<class T>
 * void foo_impl(const T&, trait_2) {
 *     // impl...
 * }
 *
 * // Main function implementation.
 *
 * // If trait_1<T>::value is true, the first implementation is chosen,
 * // Else if trait_2<T>::value is true, the second implementation is chosen.
 * // Otherwise, a compilation error occurs.
 *
 * template<class T>
 * void foo(const T& arg) {
 *    foo_impl(arg, first_true_t<trait_1<T>, trait_2<T> >{});
 * }
 * ~~~
 */
template<class T, class... Args>
struct first_true {
    /// Alias to the first of T, Args... which value is true
    using type = typename details::first_true_impl<T::value, T, Args...>::type;
};

/**
 * \brief Alias for `typename first_true<Args...>::type`
 */
template<class... Args>
using first_true_t = typename first_true<Args...>::type;

////////////////////////////////////////////////////////////////////////////////


// Static conversion type traits ///////////////////////////////////////////////
template <class From, class To, class = void>
struct has_conversion: std::false_type { };

template <class From, class To>
struct has_conversion<From, To, void_t<decltype((&From::operator To))>>: std::true_type { };

template <class From, class To, class = void>
struct has_explicit_conversion: std::false_type {};

template <class From, class To>
struct has_explicit_conversion<From, To, void_t<decltype(std::declval<To&>() = (To)std::declval<From&>())>>: std::true_type { };

template <class From, class To, class = void>
struct has_implicit_conversion: std::false_type {};

template <class From, class To>
struct has_implicit_conversion<From, To, void_t<decltype(std::declval<To&>() = std::declval<From&>())>>: std::true_type { };

////////////////////////////////////////////////////////////////////////////////

// Range type traits

namespace details {
namespace is_range {

using std::begin;
using std::end;

template<class T>
struct has_begin_impl {
    template<class U>
    static constexpr auto check(U* u)
        -> decltype(begin(*u), true)
    {return (void)u, true;}
    static constexpr bool check(...) {return false;}
    enum {value = check((T*)0)};
};

template<class T>
struct has_end_impl {
    template<class U>
    static constexpr auto check(U* u)
        -> decltype(end(*u), true)
    {return (void)u, true;}
    static constexpr bool check(...) {return false;}
    enum {value = check((T*)0)};
};

template<class T>
struct range_extremities {
    using begin_t = decltype(begin(std::declval<T&>()));
    using end_t = decltype(end(std::declval<T&>()));
};

}} // close namespace details::is_range

template<class T> struct has_begin : details::is_range::has_begin_impl<T> {};
template<class T> struct has_end : details::is_range::has_end_impl<T> {};


namespace details {
namespace is_range {
template<class T>
struct is_range_impl {

    template<class U>
    static constexpr auto check(U* u)
        -> decltype(
            begin(*u) != end(*u), void(),
            ++std::declval<decltype(begin(*u))&>(), void(),
            *begin(*u), void(),
            std::declval<typename std::iterator_traits<decltype(begin(*u))>::value_type>(), void(),
            true)
    {return (void)u, true;}

    static constexpr bool check(...) {return false;}
    enum {value = check((T*)0)};
};

template<class T>
struct range_element_impl {
    using type = typename std::iterator_traits<decltype(begin(std::declval<typename std::remove_reference<T>::type>()))>::value_type;
};

template<class T, std::size_t N>
struct range_element_impl<T[N]> {
    using type = T;
};

}} // close namespace details::is_range

template<class T> struct is_range : details::is_range::is_range_impl<T> {};


template<class T>
struct range_traits {
    static_assert(is_range<T>::value, "Type must satisfy inria::is_range");

    using value_type = typename details::is_range::range_element_impl<T>::type;
    using begin_t = typename details::is_range::range_extremities<T>::begin_t;
    using end_t = typename details::is_range::range_extremities<T>::end_t;
};

template<class T>
using range_element_t = typename range_traits<T>::value_type;

// Assignable type trait

template<class T, class It1, class It2>
struct is_assignable {
    template<class U>
    static constexpr auto check(U* u)
        -> decltype(
            (*u).assign(std::declval<It1>(), std::declval<It2>()),
            true)
    {return (void)u, true;}

    static constexpr bool check(...) {return false;}
    enum {value = check((T*)0)};
};


template<class W, class E>
struct is_weight {
    template<class U>
    static constexpr auto check(U* u, E* e = nullptr, std::uint64_t w = 0)
        -> decltype(
            w = (*u)(*e), void(),
            true)
    {return (void)u, (void)e, (void)w, true;}
    static constexpr bool check(...) {return false;}
    enum {value = check((W*)0)};
};


template<class T, class Elem>
struct is_algo_distribution {
    template<class U>
    constexpr static auto check(U* u, Elem* e = nullptr, std::uint64_t w = 0)
        -> decltype(
            w = (*u)(*e),
            true
            )
    {return (void)u, (void)e, (void)w, true;}
    static constexpr bool check(...) {return false;}
    enum {value = check((T*)0)};
};

namespace details {

/**
 * \brief Implementation of contiguous storage type trait
 *
 * Checks that given type implements `data` and `size` methods. Those are
 * expected (not checked) to return a pointer to an array of stored elements
 * and the array size.
 *
 * \tparam T Inpected type
 *
 * \warning There is no way to ensure that the interface respects the above
 * conditions.
 *
 * The following checks are done, with an object `t` of type `T`:
 *
 *   - t.data() exists and its type is an array or a pointer
 *   - t.size() exists its type is an integral value
 *
 * The types are decayed through std::decay.
 */
template<class T>
struct has_contiguous_storage_impl {
    template<class U>
    constexpr static auto check(U*u)
        -> decltype(u->data(), void(),
                    u->size(), void(),
                    true)
    {
        using data_t = typename std::decay<decltype(u->data())>::type;
        using size_t = typename std::decay<decltype(u->size())>::type;
        return (std::is_pointer<data_t>::value || std::is_array<data_t>::value)
            && std::is_integral<size_t>::value;
    }
    static constexpr bool check(...) {return false;}
    /// value is true if check is successful
    enum {value = check((T*)0)};
};

} // close namespace [inria]::details

/**
 * \brief Contiguous storage type trait
 *
 * \copydetails details::has_contiguous_storage_impl
 */
template<class T>
struct has_contiguous_storage :
        std::integral_constant<bool, details::has_contiguous_storage_impl<T>::value>
{};

} // close namespace inria

#endif /* _META_HPP_ */
