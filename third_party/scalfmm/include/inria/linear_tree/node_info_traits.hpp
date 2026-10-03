#ifndef _INRIA_NODE_INFO_TRAITS_HPP_
#define _INRIA_NODE_INFO_TRAITS_HPP_

#include "inria/meta.hpp"

namespace inria {
namespace linear_tree {
namespace node {

// Implementation of the `level(e)` free function, see its documentation below
namespace details {

/**
 * \brief `level()` member function detection type trait.
 *
 * Holds a `value` static value that is true if the feature is detected.
 *
 * \tparam T Inspected type
 */
template<class T>
struct has_level_member_function {
    template<class U, class = decltype(std::declval<U>().level())>
    constexpr static bool check(U*) {return true;}
    constexpr static bool check(...) {return false;}
    /// value is true if the feature is detected
    enum {value = check((T*)0)};
};

/**
 * \brief Retrieve level using member function.
 *
 * \param e   Object to retrieve level from
 * \param tag Tag dispatch parameter
 *
 * \tparam T Type of `e`
 */
template<class T>
constexpr auto get_level(const T& e, has_level_member_function<T> tag)
    -> decltype(e.level())
{
    return (void)tag, e.level();
}

/**
 * \brief `level` attribute detection type trait.
 *
 * Holds a `value` static value that is true if the feature is detected.
 *
 * \tparam T Inspected type
 */
template<class T>
struct has_level_attribute {
    template<class U, bool V = std::is_integral<decltype(U::level)>::value>
    constexpr static bool check(U*) {return V;}
    constexpr static bool check(...) {return false;}
    /// value is true if the feature is detected
    enum {value = check((T*)0)};
};

/**
 * \brief Retrieve level using attribute.
 *
 * \param e   Object to retrieve level from
 * \param tag Tag dispatch parameter
 *
 * \tparam T Type of `e`
 */
template<class T>
constexpr auto get_level(const T& e, has_level_attribute<T> tag)
    -> decltype(e.level)
{
    return (void)tag, e.level;
}


/**
 * \brief Checks T::getLevel() exitence
 *
 * \tparam T Inspected type
 */
template<class T, class = void>
struct has_getLevel : std::false_type {};

/**
 * \brief Specialisation for has_getLevel success case
 *
 * \tparam T Inspected type
 */
template<class T>
struct has_getLevel<T, inria::void_t<decltype(std::declval<T>().getLevel())>> : std::true_type {};

/**
 * \brief Retrieve level using `e.getLevel()`.
 *
 * \param e   Object to retrieve level from
 * \param tag Tag dispatch parameter
 *
 * \tparam T Type of `e`
 */
template<class T>
constexpr auto get_level(const T& e, has_getLevel<T> tag)
    -> decltype(e.getLevel())
{
    return (void)tag, e.getLevel();
}

} // close namespace [inria::linear_tree::node]::details

/**
 * \brief Get the level of an object
 *
 * This function will get the level of `e` if it implements one of the
 * following:
 *
 *   - `e.level()` member function
 *   - `e.level` attribute
 *
 * The implementation works using tag dispatching. Several traits are computed,
 * one for each feature. The first trait that detects its feature is used as a
 * tag to choose the details::get_level overload that is used.
 *
 * \param e Object to retrieve level from
 *
 * \tparam T   Type of e
 * \tparam Tag Tag type for the implementation tag dispatch
 *
 * Example
 *
 * ~~~{.cpp}
 * struct test { int level; };
 * test t {24};
 *
 * using inria::linear_tree::node::level;
 *
 * auto l = level(t);
 * // l == 24
 * ~~~
 *
 * Note: this will not prevent ADL from working.
 *
 * ~~~{.cpp}
 * struct test {
 *     int level;
 *     friend int level(const test& t) { return 38; }
 * };
 * test t {24};
 *
 * using inria::linear_tree::node::level;
 *
 * auto l = level(t); // will select the friend function
 * // l == 38
 * ~~~
 *
 */
template<class T,
         class Tag = first_true_t<
             details::has_getLevel<T>,
             details::has_level_member_function<T>,
             details::has_level_attribute<T>
             >>
constexpr
auto level(const T& e)
    -> decltype(details::get_level(e, Tag{}))
{
    return details::get_level(e, Tag{});
}


// Implementation of the `morton_index(e)` free function, see its documentation below
namespace details {

/**
 * \brief `morton_index` attribute detection type trait.
 *
 * Holds a `value` static value that is true if the feature is detected.
 *
 * \tparam T Inspected type
 */
template<class T>
struct has_morton_index_attribute {
    template<class U, bool V = std::is_integral<decltype(U::morton_index)>::value>
    constexpr static bool check(U*) {return true;}
    constexpr static bool check(...) {return false;}
    enum {value = check((T*)0)};
};

/**
 * \brief Retrieve morton index using attribute.
 *
 * \param e   Object to retrieve morton index from
 * \param tag Tag dispatch parameter
 *
 * \tparam T Type of `e`
 */
template<class T, class... Args>
constexpr auto get_morton_index(const T& e, has_morton_index_attribute<T> tag)
    -> decltype(e.morton_index)
{
    return (void)tag, e.morton_index;
}


/**
 * \brief `morton_index()` member function detection type trait.
 *
 * Holds a `value` static value that is true if the feature is detected.
 *
 * \tparam T Inspected type
 */
template<class T>
struct has_morton_index_member_function {
    template<class U, class V = decltype(std::declval<U>().morton_index())>
    constexpr static bool check(U*) {return true;}
    constexpr static bool check(...) {return false;}
    enum {value = check((T*)0)};
};

/**
 * \brief Retrieve morton index using member function.
 *
 * \param e   Object to retrieve morton index from
 * \param tag Tag dispatch parameter
 *
 * \tparam T Type of `e`
 */
template<class T, class... Args>
constexpr
auto get_morton_index(const T& e, has_morton_index_member_function<T>)
    -> decltype(e.morton_index())
{
    return e.morton_index();
}

/**
 * \brief Checks T::getLevel() exitence
 *
 * \tparam T Inspected type
 */
template<class T, class = void>
struct has_getMortonIndex : std::false_type {};

/**
 * \brief Specialisation for has_getMortonIndex success case
 *
 * \tparam T Inspected type
 */
template<class T>
struct has_getMortonIndex<T, inria::void_t<decltype(std::declval<T>().getMortonIndex())>> : std::true_type {};

/**
 * \brief Retrieve morton index using `e.getMortonIndex()`.
 *
 * \param e   Object to retrieve level from
 * \param tag Tag dispatch parameter
 *
 * \tparam T Type of `e`
 */
template<class T>
constexpr auto get_morton_index(const T& e, has_getMortonIndex<T> tag)
    -> decltype(e.getMortonIndex())
{
    return (void)tag, e.getMortonIndex();
}

} // close namespace [inria::linear_tree::node]::details

/**
 * \brief Get an object morton index
 *
 * This function will get the morton index of `e` if it implements one of the
 * following:
 *
 *   - `e.morton_index()` member function
 *   - `e.morton_index` attribute
 *
 * The implementation works using tag dispatching. Several traits are computed,
 * one for each feature. The first trait that detects its feature is used as a
 * tag to choose the details::get_level overload that is used.
 *
 * \param e Object to retrieve morton index from
 *
 * \tparam T   Type of e
 * \tparam Tag Tag type for the implementation tag dispatch
 *
 * Example
 *
 * ~~~{.cpp}
 * struct test { int morton_index; };
 * test t {24};
 *
 * using inria::linear_tree::node::morton_index;
 *
 * auto m = morton_index(t);
 * // m == 24
 * ~~~
 *
 * Note: this will not prevent ADL from working.
 *
 * ~~~{.cpp}
 * struct test {
 *     int morton_index;
 *     friend int morton_index(const test& t) { return 38; }
 * };
 * test t {24};
 *
 * using inria::linear_tree::node::morton_index;
 *
 * auto m = morton_index(t); // will select the friend function
 * // m == 38
 * ~~~
 *
 */
template<class T,
         class Tag = first_true_t<
             details::has_morton_index_member_function<T>,
             details::has_morton_index_attribute<T>
             >>
constexpr
auto morton_index(const T& e)
    -> decltype(details::get_morton_index(e, Tag{}))
{
    return details::get_morton_index(e, Tag{});
}


// Implementation of get_dimension
namespace details {

template<class T>
struct has_Dim_static_value {
    template<class U, decltype(U::Dim) = U::Dim>
    constexpr static bool check(U*) {return true;}
    constexpr static bool check(...) {return false;}
    enum {value = check((T*)0)};
};

template<class T>
struct has_position_Dim_static_value {
    template<class U, decltype(U::position_t::Dim) = U::position_t::Dim>
    constexpr static bool check(U*) {return true;}
    constexpr static bool check(...) {return false;}
    enum {value = check((T*)0)};
};

template<class T,
         bool = has_Dim_static_value<T>::value,
         bool = has_position_Dim_static_value<T>::value
         >
struct dimension;

template<class T, bool B>
struct dimension<T, true, B> {
    constexpr static auto get()
        -> decltype(T::Dim)
    {
        return T::Dim;
    }
};

template<class T>
struct dimension<T, false, true> {
    constexpr static auto get()
        -> decltype(T::position_t::Dim)
    {
        return T::position_t::Dim;
    }
};


} // close namespace [inria::linear_tree::node]::details

template<class T>
constexpr
auto dimension() noexcept
    -> decltype(details::dimension<T>::get())
{
    return details::dimension<T>::get();
}
;


// Implementation of node info traits
template<class T>
struct is_node_info {
    template<class U>
    constexpr static auto check(U* u)
        -> decltype(morton_index(*u), level(*u), true)
    {return true;}
    constexpr static bool check(...) {return false;}
    enum {value = check((T*)0)};
};


template<class T>
struct node_info_traits {
    static_assert(is_node_info<T>::value, "Type does not implement the node info concept");
    using morton_index_t = decltype(morton_index(std::declval<T>()));
    using level_t = decltype(level(std::declval<T>()));
    enum {Dim = dimension<T>()};
    enum {child_count = (1 << Dim)};
    // 8 bits in a byte, 1 bit per level per dimension
    enum {max_level = sizeof(morton_index_t) * 8 / Dim};

    template<class level_t>
    constexpr static morton_index_t max_idx(level_t l) {
        return (morton_index_t{1} << (Dim * l)) - 1;
    }
};



}}} // close namespace inria::linear_tree::node

#endif /* _INRIA_NODE_INFO_TRAITS_HPP_ */
