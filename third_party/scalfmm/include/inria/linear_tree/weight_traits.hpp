#ifndef _INRIA_WEIGHT_TRAITS_HPP_
#define _INRIA_WEIGHT_TRAITS_HPP_

#include "inria/meta.hpp"

namespace inria {
namespace meta {

template<class T>
struct has_weight_attribute {
    template<class U, bool B = std::is_arithmetic<decltype(U::weight)>::value>
    constexpr static bool check(U*) {return B;}
    constexpr static bool check(...) {return false;}
    enum {value = check((T*)0)};
};

template<class T>
struct has_weight_member_function {
    template<class U>
    constexpr static auto check(U* u)
        -> decltype(u->weight(), void(), true)
    {return true;}
    constexpr static bool check(...) {return false;}
    enum {value = check((T*)0)};
};

template<class T>
struct has_weight_free_function {
    template<class U, class V = decltype(weight(std::declval<U>()))>
    constexpr static bool check(U*) {return true;}
    constexpr static bool check(...) {return false;}
    enum {value = check((T*)0)};
};


namespace details {

template<class T>
constexpr
auto get_weight(const T& w, has_weight_member_function<T>) noexcept(noexcept(w.weight()))
    -> decltype(w.weight())
{
    return w.weight();
}

template<class T>
constexpr
auto get_weight(const T& w, has_weight_free_function<T>) noexcept(noexcept(weight(w)))
    -> decltype(weight(w))
{
    return weight(w);
}

template<class T>
constexpr
auto get_weight(const T& w, has_weight_attribute<T>) noexcept
    -> decltype(w.weight)
{
    return w.weight;
}

} // close namespace inria::meta::details

template<class T, class Tag = inria::first_true_t<
                      has_weight_member_function<T>,
                      has_weight_free_function<T>,
                      has_weight_attribute<T>
                      >>
constexpr
auto get_weight(const T& w) noexcept(noexcept(details::get_weight(w, Tag{})))
    -> decltype(details::get_weight(w, Tag{}))
{
    return details::get_weight(w, Tag{});
}



template<class T, class Weight>
struct has_set_weight_member_function {
    template<class U>
    constexpr static auto check(U* u, Weight* w = nullptr)
        -> decltype(u->set_weight(*w), true)
    {return true;}
    constexpr static bool check(...) {return false;}
    enum {value = check((T*)0)};
};

template<class T, class Weight>
struct has_set_weight_free_function {
    template<class U>
    constexpr static auto check(U* u, Weight* w = nullptr)
        -> decltype(set_weight(*u, *w), true)
    {return true;}
    constexpr static bool check(...) {return false;}
    enum {value = check((T*)0)};
};

namespace details {

template<class T, class W, class _>
void set_weight_impl(T& t, W&& w, has_set_weight_member_function<T,_>) {
    t.set_weight(std::forward<W>(w));
}

template<class T, class W, class _>
void set_weight_impl(T& t, W&& w, has_set_weight_free_function<T,_>) {
    set_weight(t, std::forward<W>(w));
}

template<class T, class W>
void set_weight_impl(T& t, W&& w, has_weight_attribute<T>) {
    t.weight = std::forward<W>(w);
}

} // close namespace inria::meta::details


template<class T, class W,
         class W_ = typename std::remove_reference<W>::type,
         class Tag = inria::first_true_t<
             has_set_weight_member_function<T, W_>,
             has_set_weight_free_function<T, W_>,
             has_weight_attribute<T>>
         >
void set_weight(T& t, W&& w) {
    details::set_weight_impl(t, w, Tag{});
}


template<class T>
struct weight_traits {
    using weight_t = typename std::remove_reference<decltype(inria::meta::get_weight(std::declval<T>()))>::type;
};

}} // close namespace inria::meta

#endif /* _INRIA_WEIGHT_TRAITS_HPP_ */
