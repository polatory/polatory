#ifndef _NODE_HPP_
#define _NODE_HPP_

#include "configuration.hpp"

#include "inria/integer_sequence.hpp"
#include "node_info_traits.hpp"

#include <algorithm>
#include <array>
#include <ostream>
#include <type_traits>
#include <vector>

namespace inria {
namespace linear_tree {
namespace node {

// Implementation of node info interface

namespace help {

template<class T>
constexpr static std::size_t get_m_idx(const T& n) {
    using node::morton_index;
    return morton_index(n);
}

template<class T>
std::size_t get_l(const T& n) {
    using node::level;
    return level(n);
}

}


template<std::size_t Dim_>
struct info {
    enum {Dim = Dim_};
    std::size_t morton_index;
    std::size_t level;

    info() = default;
    info(const info&) = default;
    info(info&&)      = default;
    info& operator=(const info&) = default;
    info& operator=(info&&)      = default;


    template<class T,
             class = inria::enable_if_t<is_node_info<T>::value>
             >
    constexpr
    info(const T& n) : info{help::get_m_idx(n), help::get_l(n)} {}

    template<class T,
             class = inria::enable_if_t<is_node_info<T>::value>
             >
    info& operator=(const T& n) {
        this->morton_index = help::get_m_idx(n);
        this->level = help::get_l(n);
        return *this;
    }

    constexpr info(std::size_t m_idx, std::size_t lvl)
        : morton_index(m_idx), level(lvl) {}

    friend constexpr
    info get_ancestor_if_lower_than(
        const info& n,
        std::size_t lvl)
        noexcept
    {
        return (n.level > lvl ? ancestor(n, n.level - lvl) : n);
    }

    friend constexpr
    bool operator<(const info& lhs, const info& rhs) noexcept {
        // C++11 forces use of a ternary for a constexpr
        return (get_ancestor_if_lower_than(lhs, rhs.level).morton_index
                < get_ancestor_if_lower_than(rhs, lhs.level).morton_index)
            || ((get_ancestor_if_lower_than(lhs, rhs.level).morton_index
                 == get_ancestor_if_lower_than(rhs, lhs.level).morton_index)
                && lhs.level < rhs.level);
    }

    friend constexpr
    bool operator==(const info& lhs, const info& rhs) noexcept {
        return (lhs.morton_index == rhs.morton_index) && (lhs.level == rhs.level);
    }

    friend constexpr
    bool operator!=(const info& lhs, const info& rhs) noexcept {
        return !(lhs == rhs);
    }

    friend constexpr
    bool operator>(const info& lhs, const info& rhs) noexcept {
        return rhs < lhs;
    }

    friend constexpr
    bool operator>=(const info& lhs, const info& rhs) noexcept {
        return !(lhs < rhs);
    }

    friend constexpr
    bool operator<=(const info& lhs, const info& rhs) noexcept {
        return !(lhs > rhs);
    }


    friend
    std::ostream& operator<<(std::ostream& os, const info& n) {
        return os << '(' << n.morton_index << ", " << n.level << ')';
    }

};

template<class T>
using info_t = info<node::dimension<T>()>;


template<class T,
         class = inria::enable_if_t<is_node_info<T>::value>,
         std::size_t Dim = node_info_traits<T>::Dim
         >
constexpr
info_t<T> ancestor(const T& n, std::size_t offset) {
    return (offset <= level(n))
        ? info_t<T>{morton_index(n) >> (Dim*offset), level(n) - offset}
    : info_t<T>{0,0};
}

template<class T,
         class = inria::enable_if_t<is_node_info<T>::value>
         >
constexpr
auto parent(const T& n)
    -> decltype(ancestor(n,1))
{
    return ancestor(n, 1);
}


template<class T, class U,
         class = inria::enable_if_t<is_node_info<T>::value>,
         class = inria::enable_if_t<is_node_info<U>::value>,
         std::size_t Dim = node_info_traits<T>::Dim
         >
constexpr
bool is_ancestor_of(const U& n1, const T& n2) {
    return (level(n1) < level(n2))
        && (morton_index(ancestor(n2, level(n2) - level(n1))) == morton_index(n1));
}

template<class T,
         class = inria::enable_if_t<is_node_info<T>::value>,
         std::size_t Dim = node_info_traits<T>::Dim
         >
constexpr
info_t<T> child(const T& n, std::size_t idx) {
    return info_t<T>{(morton_index(n) << Dim) + idx, level(n)+1};
}

template<class T,
         class = inria::enable_if_t<is_node_info<T>::value>,
         std::size_t Dim = node_info_traits<T>::Dim
         >
constexpr
info_t<T> first_descendant(const T& n, std::size_t offset) noexcept {
    return info_t<T>{(morton_index(n) << (Dim * offset)), level(n) + offset};
}

template<class T,
         class = inria::enable_if_t<is_node_info<T>::value>,
         std::size_t Dim = node_info_traits<T>::Dim
         >
constexpr
info_t<T> first_child(const T& n) noexcept {
    return first_descendant(n, 1);
}

template<class T,
         class = inria::enable_if_t<is_node_info<T>::value>,
         std::size_t Dim = node_info_traits<T>::Dim
         >
constexpr
info_t<T> last_descendant(const T& n, std::size_t offset) noexcept {
    return {((morton_index(n)+1) << (Dim * offset)) - 1, level(n) + offset};
}

template<class T,
         class = inria::enable_if_t<is_node_info<T>::value>,
         std::size_t Dim = node_info_traits<T>::Dim
         >
constexpr
info_t<T> last_child(const T& n) noexcept {
    return last_descendant(n, 1);
}

// Implementation of children function
namespace details {

template<class node_traits, class T, std::size_t... Is>
constexpr
std::array<info_t<T>, node_traits::child_count>
children_impl(const T& n, inria::index_sequence<Is...>) noexcept {
    static_assert(sizeof...(Is) == node_traits::child_count,
                  "The indices must be those of the children");
    return {{child(n,Is) ...}};
}


} // close namespace [inria::linear_tree::node]::details

template<class T,
         class = inria::enable_if_t<is_node_info<T>::value>,
         class IdxSeq = inria::make_index_sequence<node_info_traits<T>::child_count>
         >
constexpr
auto children(const T& n)
    -> decltype(details::children_impl<node_info_traits<T>>(n, IdxSeq{}))
{
    return details::children_impl<node_info_traits<T>>(n, IdxSeq{});
}

// Implementation of common_ancestor
namespace details {

template<class node_traits, class T>
constexpr
info<node_traits::Dim> get_ancestor_if_lower_than(
    const T& n,
    typename node_traits::level_t lvl)
    noexcept
{
    return (level(n) > lvl ? ancestor(n, level(n) - lvl) : info<node_traits::Dim>(n));
}

template<std::size_t Dim>
constexpr
info<Dim> common_ancestor_impl(const info<Dim>& lhs, const info<Dim>& rhs)
    noexcept
{
    return (lhs == rhs ? lhs
            : common_ancestor_impl(parent(lhs), parent(rhs)));
}

} // close namespace [inria::linear_tree::node]::details

template<class T, class U,
         class = inria::enable_if_t<is_node_info<T>::value>,
         class = inria::enable_if_t<is_node_info<U>::value>,
         class = inria::enable_if_t<node_info_traits<T>::Dim == node_info_traits<U>::Dim>
         >
constexpr
info_t<T> common_ancestor(const T& lhs, const U& rhs)
    noexcept
{
    return details::common_ancestor_impl(
        parent(details::get_ancestor_if_lower_than<node_info_traits<T>>(lhs, level(rhs))),
        parent(details::get_ancestor_if_lower_than<node_info_traits<U>>(rhs, level(lhs))));
}


template<class T, class It,
         class = inria::enable_if_t<is_node_info<T>::value>
         >
It ancestors(const T& n, It first, It last) {
    std::size_t i = 1;
    while(first != last && i < level(n)) {
        *first = ancestor(n, i);
        ++first;
        ++i;
    }
    return first;
}

template<class T,
         class = inria::enable_if_t<is_node_info<T>::value>
         >
std::vector<info_t<T>> ancestors(const T& n) {
    std::vector<info_t<T>> anc(level(n));
    ancestors(n, begin(anc), end(anc));
    return anc;
}


template<class T>
info_t<T> make_info(const T& n) {
    auto m = help::get_m_idx(n);
    auto l = help::get_l(n);
    return {m, l};
}



} // close namespace [inria::linear_tree]::node

}}




#endif /* _NODE_HPP_ */
