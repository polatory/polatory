#ifndef _COMPLETE_REGION_HPP_
#define _COMPLETE_REGION_HPP_

#include <iostream>

#include "node.hpp"

namespace inria {
namespace linear_tree {


/**
 * \brief Create the coarsest linear tree between two octants
 *
 * \param a First node of the range
 * \param b Last node of the range
 *
 * \return A vector containing the leafs of tree (a and b excluded) as after a
 * left prefix traversal.
 */
template<std::size_t Dim, class OutRange>
void complete_region(
    node::info<Dim> a,
    node::info<Dim> b,
    OutRange& out_range
    )
{
    using std::swap;
    using std::end;
    using std::begin;
    using std::end;

    if(b < a) {
        swap(a,b);
    }

    if(a == b or is_ancestor_of(a, b)) {
        return;
    }

    std::vector<node::info<Dim>> W, region;
    {
        auto children = node::children(common_ancestor(a,b));
        W.assign(std::begin(children), std::end(children));
    }

    auto b_ans = ancestors(b);
    auto a_b_ans = ancestors(a);
    a_b_ans.insert(std::end(a_b_ans), std::begin(b_ans), std::end(b_ans));

    while(! W.empty()) {
        auto w = W.back();
        auto w_pos = W.end() - 1;
        if((a < w) && (w < b)
           && (std::find(std::begin(b_ans), std::end(b_ans), w) == std::end(b_ans)))
        {
            W.erase(w_pos);
            region.insert(end(region), w);
        } else if(std::find(begin(a_b_ans), end(a_b_ans), w) != end(a_b_ans)) {
            auto children = node::children(w);
            W.erase(w_pos);
            W.insert(end(W), begin(children), end(children));
        } else {
            W.erase(w_pos);
        }
    }

    std::sort(begin(region), end(region));
    out_range.insert(end(out_range), begin(region), end(region));
}

/**
 * \brief Regenerate region so that its octants are as coarse as possible
 */
template<class Range>
void coarsen_region(Range& region) {
    using node_t = node::info_t<typename Range::value_type>;

    if(region.empty()) {
        return;
    }

    auto is_last_descendant = [](const node_t& p, const node_t& d) {
        return last_descendant(p, d.level - p.level) == d;
    };

    node_t a = region.front(), b = region.back();
    node_t p_a = parent(a), p_b = parent(b);

    while(first_child(p_a) == a && (! is_ancestor_of(p_a, b)
                                    || is_last_descendant(p_a, b)))
    {
        a = p_a;
        p_a = parent(a);
    }

    while(last_child(p_b) == b && (! is_ancestor_of(p_b, a)
                                   || is_last_descendant(p_b, a)))
    {
        b = p_b;
        p_b = parent(b);
    }
    region.clear();
    region.insert(end(region), a);
    complete_region(a, b, region);
    if(a != b) {
        region.insert(end(region), b);
    }
}


}} // end namespace inria::linear_tree


#endif /* _COMPLETE_REGION_HPP_ */
