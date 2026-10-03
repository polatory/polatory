#ifndef _LINEARIZE_HPP_
#define _LINEARIZE_HPP_

#include <iostream>

#include <cassert>

#include <algorithm>
#include <utility>

#include "node.hpp"

namespace inria {
namespace linear_tree {

/**
 * \brief Removes overlaps from a sorted list of octants
 *
 * This function removes the internal nodes from a list of sorted octants. The
 * sort order is defined as follows. For two node a and b, a < b if:
 *
 *   + level(a) == level(b) and morton_index(a) < morton_index(b)
 *   + level(a) < level(b) and a < parent(b)
 *   + level(a) > level(b) and parent(a) < b
 *
 * In other words, the range of octants must be sorted in prefix order.
 *
 * The elements in the range are shifted to only keep the leafs. The other ones
 * are overwritten as per move-assignment operator. The elements after the new
 * end are still dereferenceable but have undefined value.
 *
 * \param first Range first element
 * \param last  Range end sentinel
 *
 * \tparam ForwardIt Iterator type satifying the [forward iterator](http://en.cppreference.com/w/cpp/concept/ForwardIterator) concept.
 *
 * \return An iterator to the new end of the range.
 */
template<class ForwardIt>
ForwardIt linearize(ForwardIt first, ForwardIt last) {
    using node_t = typename std::remove_reference<decltype(*std::declval<ForwardIt>())>::type;
    static_assert(node::is_node_info<node_t>::value,
                  "Range must hold elements of type "
                  "inria::linear_tree::node_info.");

    if(first == last) {
        return last;
    }

    assert(std::is_sorted(first, last));

    ForwardIt new_end = first, next = first;
    ++next;
    while(next != last) {
        if(! is_ancestor_of(*first, *next)) {
            *new_end = std::move(*first);
            ++new_end;
        }
        ++first; ++next;
    }
    *new_end = std::move(*first);
    ++new_end;
    return new_end;
}

}} // close namespace inria::linear_tree

#endif /* _LINEARIZE_HPP_ */
