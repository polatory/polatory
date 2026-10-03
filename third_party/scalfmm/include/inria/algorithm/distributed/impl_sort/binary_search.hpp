#ifndef _BINARY_SEARCH_HPP_
#define _BINARY_SEARCH_HPP_

#include <functional>

namespace inria {

namespace details {

template<class Compare = std::less<void>, class T1, class T2>
bool less(const T1& lhs, const T2& rhs, Compare comp) {
    return comp(lhs, rhs);
}

template<class Compare, class T1, class T2>
bool greater(const T1& lhs, const T2& rhs, Compare comp) {
    return less(rhs, lhs, comp);
}

template<class Compare, class T1, class T2>
bool less_equal(const T1& lhs, const T2& rhs, Compare comp) {
    return !greater(lhs,rhs, comp);
}

template<class Compare, class T1, class T2>
bool greater_equal(const T1& lhs, const T2& rhs, Compare comp) {
    return !less(lhs, rhs, comp);
}

template<class Compare, class T1, class T2>
bool equal(const T1& lhs, const T2& rhs, Compare comp) {
    return !less(lhs,rhs,comp) && !greater(lhs,rhs,comp);
}

template<class Compare, class T1, class T2>
bool different(const T1& lhs, const T2& rhs, Compare comp) {
    return less(lhs,rhs,comp) || greater(lhs,rhs,comp);
}


}

namespace tag {
/// Find first element equal to target or the first element greater than target
/// in the range.
struct first_or_next {
    template<class Compare, class T1, class T2>
    static constexpr bool compare(const T1& lhs, const T2& rhs, Compare comp) {
        return details::less_equal(lhs, rhs, comp);
    }
};

/// Find element right after the target in the range.
struct next {
    template<class Compare, class T1, class T2>
    static constexpr bool compare(const T1& lhs, const T2& rhs, Compare comp) {
        return details::less(lhs, rhs, comp);
    }
};
} // namespace tag

/**
 * \brief Runs a binary search
 *
 * Performs a binary search over the sorted range \[first, last\) for the target
 * value. The ComparePolicy determines the returned iterator:
 *   - inria::tag::first_or_next, if the value is found, its first position is
 *     returned, ortherwise the position of the next value is returned;
 *   - inria::tag::next, the position of the next value is always returned.
 *
 * \param first  Iterator to the first element of the range.
 * \param last   Iterator to the last element of the range.
 * \param target Value to look for.
 *
 * \tparam ComparePolicy The comparison policy, either inria::tag::first_or_next
 *                       or inria::tag::next
 * \tparam RandomAccessIterator Iterator over a range of elements.
 * \tparam T Comparison target type
 * \tparam Compare  Comparison functor type. Must expose an operator() with the
 *                  signature `bool operator()(T target, U elem)` which returns
 *                  `true` if `target < elem`.
 *
 * \return The position of the first element to match the target. If no element
 * matches, the position of the first element greater to target is returned
 * (this may be the range past-the-end iterator).
 *
 * \warning \[first, last\) must be sorted.
 *
 * \note It came to my attention to the STL already implements this function (or
 * very close to) with the algorithms std::lower_bound and std::upper_bound.
 *
 * Example:
 *
 * ~~~{.cpp}
 * std::array<int, 20> data {};
 * int tmp = 0;
 * std::generate(begin(data), end(data), []{return tmp += 2;});
 *
 * binary_search();
 * ~~~
 */
template<class ComparePolicy, class RandomAccessIterator, class T, class Compare = std::less<T>>
[[gnu::deprecated("Use std::lower_bound (equivalent to tag::first_or_next) or std::upper_bound (equivalent to tag::next) instead")]]
RandomAccessIterator binary_search(
    ComparePolicy,
    RandomAccessIterator first,
    RandomAccessIterator last,
    const T& target,
    Compare comp = {})
{
    static_assert(std::is_same<ComparePolicy, tag::first_or_next>::value
                  || std::is_same<ComparePolicy, tag::next>::value,
                  "The ComparisonPolicy type must be one of "
                  "inria::tag::first_or_next, inria::tag::next");

    while(last - first > 2) {
        RandomAccessIterator median = first + ((last - first) / 2);
        if(ComparePolicy::compare(target, *median, comp)) {
            last = median + 1;
        } else {
            first = median;
        }
    }
    if(first < last && ComparePolicy::compare(target, *first, comp)) {
        return first;
    } else if(last - first == 2 && ComparePolicy::compare(target, *(first+1), comp)) {
        return first+1;
    } else {
        return last;
    }
}


}



#endif /* _BINARY_SEARCH_HPP_ */
