#ifndef _INRIA_UTILS_HPP_
#define _INRIA_UTILS_HPP_

/**
 * \brief Small utilities that don't find their place in a bigger file
 * \file
 *
 * \author Quentin Khan
 */

#include <memory>

namespace inria {

/**
 * \brief Contruct a dynamic array owned by a std::unique_ptr
 *
 * \tparam Array underlying type
 *
 * Example:
 *
 * ~~~{.cpp}
 * auto arr = inria::make_array<double>(128);
 * arr[0] = 37;
 * double* non_owning_ptr = arr.get();
 * ~~~
 */
template<class T>
std::unique_ptr<T[]> make_array(std::size_t count) {
    return std::unique_ptr<T[]>{new T[count]};
}

/**
 * \brief Hold two iterators to create a range
 *
 * \tparam It1 Type of the begin iterator
 * \tparam It2 Type of the end iterator
 *
 * \note Begin and end can have different types, for instance when the end is a
 * sentinel.
 */
template<class It1, class It2>
class it_range_proxy {
    It1 b; ///< begin iterator
    It2 e; ///< end iterator
public:
    /// Value type of the range
    using value_type = typename std::iterator_traits<It1>::value_type;
    /// Reference type of the range
    using reference = value_type&;
    /// Iterator type of the range
    using iterator = It1;
    /// Const iterator type of the range, same as iterator
    using const_iterator = It1;
    /// Iterator offset type
    using difference_type = typename std::iterator_traits<It1>::difference_type;

    /**
     * \brief Return an iterator to the range beginning
     */
    iterator begin() { return b; }
    /** \copydoc begin */
    const_iterator begin() const { return b; }
    /** \copydoc begin */
    const_iterator cbegin() const { return b; }
    /**
     * \brief Return an iterator to the range end
     */
    iterator end() { return e; }
    /** \copydoc end */
    const_iterator end() const { return e; }
    /** \copydoc end */
    const_iterator cend() const { return e; }

    /**
     * \brief Check whether the range is empty
     */
    bool empty() const { return b == e; }
};

/**
 * \brief Make a range-like object from two iterators
 *
 * \param b Range begin iterator
 * \param e Range end iterator
 *
 * \tparam It1 Type of the begin iterator
 * \tparam It2 Type of the end iterator
 *
 * \note Begin and end can have different types, for instance when the end is a
 * sentinel.
 */
template<class It1, class It2>
it_range_proxy<It1, It2> it_range(It1 b, It2 e) {
    return {b,e};
}


/**
 * \brief Get the size of a container
 *
 * \param c The container
 *
 * \tparam C Container type, must define a `size` member function
 */
template<class C>
constexpr auto size(const C& c) noexcept(noexcept(c.size()))
    -> decltype(c.size())
{
    return c.size();
}

/**
 * \brief Get the size of a static array
 *
 * \param a Array to get the size of
 *
 * \tparam T Array underlying type
 * \tparam N Static array size
 *
 * \note This allows passing static arrays to generic algorithms and consider
 * them as containers that declare their size.
 */
template<class T, std::size_t N>
constexpr std::size_t size(const T(&a)[N]) noexcept {
    return void(a), N;
}



} // end namespace inria



#endif /* _INRIA_UTILS_HPP_ */
