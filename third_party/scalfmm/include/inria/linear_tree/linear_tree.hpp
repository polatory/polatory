#ifndef _LINEAR_TREE_HPP_
#define _LINEAR_TREE_HPP_

#include <cassert>

#include <iterator>


namespace inria {
namespace linear_tree {


template<class BidirIt>
bool is_linear_tree(BidirIt first, BidirIt last) {
    BidirIt next = first +1;
    while(next != last) {
        if(first->level() > next->level()) {
            if(is_ancestor_of(first, next)) {
                return false;
            }
        }
        first = next;
        ++next;
    }
    return true;
}






}} // close inria::linear_tree




#endif /* _LINEAR_TREE_HPP_ */
