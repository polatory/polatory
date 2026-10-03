#ifndef _INRIA_SPAN_HPP_
#define _INRIA_SPAN_HPP_

/**
 * \file
 * \brief Non owning contiguous storage views
 *
 * \author Quentin Khan
 */

#include <iterator>

#include "utils.hpp"
#include "meta.hpp"

namespace inria {

namespace details {

/**
 * \brief Span base class
 *
 * Implements inria::span and inria::const_span common methods.
 *
 * \warning This class must be the first base class of the derived class.
 */
template<class D, class ValueType>
class span_base {
private:
    /**
     * \brief Easy cast to derived class
     */
    D* self() {
        return static_cast<D * const>(this);
    }
    /** \copydoc self */
    const D* self() const {
        return static_cast<const D * const>(this);
    }

protected:
    span_base() = default;

public:
    /// Span underlying type
    using value_type = ValueType;
    /// Span size storage type
    using size_type = std::size_t;
    /// Span iterator offset storage type
    using difference_type = std::ptrdiff_t;
    /// Span value_type const reference
    using const_reference = const ValueType&;
    /// Span const value_type pointer
    using const_pointer = const ValueType*;
    /// Span const iterator
    using const_iterator = const_pointer;

    /**
     * \brief Return an iterator to the span beginning
     */
    const_iterator begin() const noexcept { return self()->data_; }
    /** \copydoc begin */
    const_iterator cbegin() const noexcept { return self()->data_; }

    /**
     * \brief Return an iterator to the span end
     */
    const_iterator end() const noexcept { return self()->data_ + self()->size_; }
    /** \copydoc end */
    const_iterator cend() const noexcept { return self()->data_ + self()->size_; }

    /**
     * \brief Return pointer to underlying storage
     */
    const_pointer data() const noexcept { return self()->data_; }
    /**
     * \brief Return span size
     */
    size_type size() const noexcept { return self()->size_; }
    /**
     * \brief Return span capacity, alias for #size
     */
    size_type capacity() const noexcept { return self()->size_; }

    /**
     * \brief Check whether span size is 0
     */
    bool empty() const noexcept { return self()->size_ == 0; }

    /**
     * \brief Access span element
     *
     * \param i Element index
     */
    const_reference operator[](size_type i) const noexcept { return self()->data_[i]; }

    /**
     * \brief Access span element with bound checking
     *
     * \param i Element index
     *
     * \exception std::out_of_range Thrown when `i` is greater than #size
     */
    const_reference at(size_type i) const {
        if(i < size()) {
            return self()->data_[i];
        } else {
            throw std::out_of_range{""};
        }
    }

    /**
     * \brief Return first element is the span
     *
     * \warning Behaviour is undefined if the span is empty
     */
    const_reference front() const noexcept { return (*this)[0]; }

    /**
     * \brief Return last element is the span
     *
     * \warning Behaviour is undefined if the span is empty
     */
    const_reference back() const noexcept { return (*this)[size()-1]; }

    /**
     * \brief Swap two spans
     *
     * \param s Span to be swapped with
     */
    void swap(span_base& s) noexcept {
        using std::swap;
        swap(self()->data_, s.self()->data_);
        swap(self()->size_, s.self()->size_);
    }
};

} // close namespace [inria]::details

/**
 * \brief Non owning mutable contiguous storage range view
 *
 * The class template inria::span describes a contiguous sequence of objects. It
 * does not own its data.
 *
 * \tparam T Undelying value type
 *
 * Example:
 *
 * ~~~{.cpp}
 * // Allocate a buffer, life is managed by the std::unique_ptr
 * std::unique_ptr<int[]> buffer(new int[10]);
 *
 * {
 *     // Get a range view of the buffer
 *     span<int> s{buffer.get(), 5};
 *     std::iota(s.begin(), s.end(), 0);
 *     for(auto& i : s) {
 *         std::cout << i << ' ';
 *     }
 *     std::cout << '\n';
 * } // The buffer is not destroyed when the span life ends
 * ~~~
 *
 * Output:
 *
 * ~~~
 * 1 2 3 4 5
 * ~~~
 */
template<class T>
class span : public details::span_base<span<T>, T> {
    /// Base class
    using base_t = details::span_base<span<T>, T>;
    /// Declare the base class as a friend so it can access #data_ and #size_
    friend base_t;

    /// Pointer to underlying sequence
    T* data_;
    /// Span element count
    std::size_t size_;

public:
    /// Span underlying type
    using value_type = T;
    /// Span value_type reference
    using reference = value_type&;
    /// Span value_type pointer
    using pointer = value_type*;
    /// Span iterator
    using iterator = pointer;

    /**
     * \brief Construct a span
     *
     * \param d   Pointer to underlying storage
     * \param len `d` element count
     */
    span(T* d, std::size_t len) : data_(d), size_(len) {}

    /**
     * \brief Construct a span from a static array
     *
     * \param arr Static array
     *
     * \tparam Len Static array lenght
     */
    template<typename base_t::size_type Len>
    span(T(&arr)[Len]) : span(arr, Len) {}

    /**
     * \brief Construct a span from a constainer that has contiguous storage
     *
     * \param c Container to view
     *
     * \tparam C Container type
     */
    template<class C
             #ifndef DOXYGEN_DOCUMENTATION
             ,class IsContiguousStorage = inria::enable_if_t<inria::has_contiguous_storage<C>::value>,
             class MatchingValueType = inria::enable_if_t<std::is_same<T, typename C::value_type>::value>
             #endif
             >
    span(C& c) : span(c.data(), c.size()) {}

    /** \brief Default constructor, creates an empty span */
    span() = default;
    /** \brief Default copy constructor */
    span(const span&) = default;
    /** \brief Default move constructor */
    span(span&&) = default;
    /** \brief Default copy operator */
    span& operator=(const span&) = default;
    /** \brief Default move operator */
    span& operator=(span&&) = default;

    using base_t::begin;
    using base_t::end;
    using base_t::data;
    using base_t::at;
    using base_t::front;
    using base_t::back;
    using base_t::operator[];
    /**
     * \brief Return an iterator to the span beginning
     */
    iterator begin() noexcept { return data_; }

    /**
     * \brief Return an iterator to the span end
     */
    iterator end() noexcept { return data_ + size_; }

    /**
     * \brief Return pointer to underlying storage
     */
    pointer data() noexcept { return data_; }
    /**
     * \brief Access span element
     *
     * \param i Element index
     */
    reference operator[](typename base_t::size_type i) noexcept { return data_[i]; }

    /**
     * \brief Access span element with bound checking
     *
     * \param i Element index
     *
     * \exception std::out_of_range Thrown when `i` is greater than #size
     */
    reference at(typename base_t::size_type i) {
        if(i < this->size()) {
            return data_[i];
        } else {
            throw std::out_of_range{""};
        }
    }

    /**
     * \brief Return first element is the span
     *
     * \warning Behaviour is undefined if the span is empty
     */
    reference front() noexcept { return (*this)[0]; }

    /**
     * \brief Return last element is the span
     *
     * \warning Behaviour is undefined if the span is empty
     */
    reference back() noexcept { return (*this)[size_-1]; }
};


/**
 * \brief Non owning non mutable contiguous storage range view
 *
 * The class template inria::const_span describes a constant contiguous sequence
 * of objects. It does not own its data.
 *
 * \tparam T Undelying value type
 *
 * Example:
 *
 * ~~~{.cpp}
 * // Allocate a buffer, life is managed by the std::unique_ptr
 * std::unique_ptr<int[]> buffer(new int[10]);
 * std::iota(buffer.get(), buffer.get()+10, 0);
 *
 * {
 *     // Get a range view of the buffer
 *     const_span<int> s{buffer.get(), 5};
 *     // std::iota(s.begin(), s.end(), 0); // error, the span is constant
 *     for(auto& i : s) {
 *         std::cout << i << ' ';
 *     }
 *     std::cout << '\n';
 * } // The buffer is not destroyed when the span life ends
 * ~~~
 *
 * Output:
 *
 * ~~~
 * 1 2 3 4 5
 * ~~~
 */
template<class T>
class const_span : public details::span_base<const_span<T>,T>{
    /// Base class
    using base_t = details::span_base<const_span<T>, T>;
    /// Declare the base class as a friend so it can access #data_ and #size_
    friend base_t;
    /// Pointer to underlying data sequence
    const T* data_;
    /// Element count
    std::size_t size_;

public:

    /// Span underlying type
    using value_type = T;
    /// Span value_type reference
    using reference = const value_type&;
    /// Span value_type pointer
    using pointer = const value_type*;
    /// Span iterator
    using iterator = pointer;

    /**
     * \brief Construct a const_span
     *
     * \param d   Pointer to underlying storage
     * \param len `d` element count
     */
    const_span(const T* d, std::size_t len) : data_(d), size_(len) {}

    /**
     * \brief Construct a const_span from a static array
     *
     * \param arr Static array
     *
     * \tparam Len Static array lenght
     */
    template<typename base_t::size_type Len>
    const_span(const T(&d)[Len]) : const_span(d, Len) {}

    /**
     * \brief Construct a span from a constainer that has contiguous storage
     *
     * \param c Container to view
     *
     * \tparam C Container type
     */
    template<class C
             #ifndef DOXYGEN_DOCUMENTATION
             , class = inria::enable_if_t<inria::has_contiguous_storage<C>::value>,
             class MatchingValueType = inria::enable_if_t<std::is_same<T, typename C::value_type>::value>
             #endif
             >
    const_span(const C& c) : const_span(c.data(), c.size()) {}

    /** \brief Default constructor, creates an empty span */
    const_span() = default;
    /** \brief Default copy constructor */
    const_span(const const_span&) = default;
    /** \brief Default move constructor */
    const_span(const_span&&) = default;
    /** \brief Default copy operator */
    const_span& operator=(const const_span&) = default;
    /** \brief Default move operator */
    const_span& operator=(const_span&&) = default;
};

} // close namespace inria


#endif /* _INRIA_SPAN_HPP_ */
