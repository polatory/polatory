
#ifndef CT_ARRAY_HPP
#define CT_ARRAY_HPP

/**
 * \file
 * \brief Compile time array implementation
 *
 * Implementation of an array class that may be manipulated at compile time.
 */

#include <cstddef>
#include "integer_sequence.hpp"


namespace inria {


    namespace detail {
        /**
         * \brief Check that a start+index combinaison fits in given lenght
         *
         * \param idx Index
         * \param start Start position
         * \param len Intended length
         * \param N Maximum length
         */
        constexpr bool idx_is_in_range(std::size_t idx, std::size_t start, std::size_t len, std::size_t N) {
            return (idx < N) && (start + idx < N) && (idx < len);
        }
    }


    /**
     * \brief Compile time array
     */
    template<class T, std::size_t N>
    struct ct_array {
        /// Underlying array type
        using array_t = T[N];
        /// Array data
        array_t _data;

        constexpr ct_array() = default;
        constexpr ct_array(const ct_array&) = default;
        constexpr ct_array(ct_array&&) = default;

        /**
         * \brief Copy constructor from pointer
         *
         * \tparam T2 Pointer type
         */
        template<class T2>
        constexpr ct_array(const T2* other, const std::size_t len) noexcept :
            ct_array(indices, other, 0, len)
        {}

        /**
         * \brief Copy constructor from basic array
         *
         * \tparam T2 Array type
         * \tparam N2 Array length
         */
        template<class T2, std::size_t N2>
        constexpr ct_array(const T2 (&other)[N2]) noexcept :
            ct_array(indices, other, 0, N2)
        {}

        /**
         * \brief Copy constructor from another ct_array
         *
         * \tparam T2 Array type
         * \tparam N2 Array length
         */
        template<class T2, std::size_t N2>
        constexpr ct_array(const ct_array<T2, N2>& other) noexcept :
            ct_array(other.as_array())
        {}

        /**
         * \brief Extract a sub-array
         *
         * Creates a new array and copies the specified source array range.
         *
         * \tparam N2 Sub-array maximal-size
         *
         * \param start Start of the sub-array
         * \param len Length of the sub-array
         */
        template<std::size_t N2 = N>
        constexpr ct_array<T,N2> sub_array(std::size_t start, std::size_t len = N2)
            const noexcept
        {
            static_assert(N2 <= N, "A sub-array cannot be bigger than its source");
            return {make_index_sequence<N2>(),
                    this->data(), start, start + len > N ? N-start : len};
        }


        /**
         * \brief Access element
         *
         * \param idx Element index
         *
         * \return A reference to the requested element
         */
        constexpr const T& operator[](std::size_t idx) const noexcept {
            return this->_data[idx];
        }

        /**
         * \brief Access element
         *
         * \param idx Element index
         *
         * \return A reference to the requested element
         */
        T& operator[](std::size_t idx) {
            return this->_data[idx];
        }

        /**
         * \brief Get underlying array
         *
         * \return The array
         */
        constexpr const array_t& as_array() const && noexcept {
            return _data;
        }

        /**
         * \brief Get underlying array
         *
         * \return The array
         */
        constexpr const array_t& as_array() const & noexcept {
            return _data;
        }

        /**
         * \brief Get underlying array
         *
         * \return The array
         */
        array_t& as_array() & noexcept {
            return _data;
        }

        /**
         * \brief Get underlying data
         *
         * \return The array
         */
        constexpr const T* data() const noexcept {
            return _data;
        }

        /**
         * \brief Get underlying data
         *
         * \return The array
         */
        T* data() noexcept {
            return _data;
        }

        constexpr const T* begin() const noexcept {
            return _data;
        }

        T* begin() noexcept {
            return _data;
        }

        constexpr const T* end() const noexcept {
            return _data+N;
        }

        T* end() {
            return _data+N;
        }

    private:
        static constexpr const make_index_sequence<N> indices = {};

        /** Constructor implementation
         *
         * \tparam Is Index pack corresponding to make_index_sequence<N>
         * \tparam U  C-style array type
         *
         * \param str C-style array to copy
         * \param start index from where to start
         * \param len Array length
         */
        template<std::size_t... Is, class U>
        constexpr ct_array(index_sequence<Is...>,
                           const U* str,
                           std::size_t start = 0,
                           std::size_t len = N)
            noexcept :
            _data{(detail::idx_is_in_range(Is, start, len, N) ? str[start+Is] : '\0')...}
        {}
    };


    inline namespace string_literals {
        /**
         * \brief User defined string literal to create compile time char array
         *
         * \warning The string may not be longer than 64 bytes.
         */
        constexpr ct_array<char, 64> operator"" _cstr(const char* a, std::size_t N) {
            return {a,N};
        }
    }


    /**
     * \brief Find first occurence of a value in compile time array
     *
     * \tparam U Value type
     * \tparam T Compile-time array type
     * \tparam N Compile-time array size
     *
     * \param a Compile-time array
     * \param value Value to search for
     * \param start Index to start from
     *
     * \return The value index in the array if it is found, N otherwise
     */
    template<class U, class T, std::size_t N>
    constexpr std::size_t find(const ct_array<T,N>& a, const U& value, std::size_t idx = 0) noexcept {
        return idx >= N ? N : (a[idx] == value ? idx : find(a, value, idx + 1));
    }

    /**
     * \brief Find first occurence of a value in array
     *
     * \tparam U Value type
     * \tparam T Compile-time array type
     * \tparam N Compile-time array size
     *
     * \param a Array
     * \param value Value to search for
     * \param idx Index to start search from
     *
     * \return The value index in the array if it is found, N otherwise
     */
    template<class U, class T, std::size_t N>
    constexpr std::size_t find(const T (&a)[N], const U& value, std::size_t idx = 0) noexcept {
        return idx >= N ? N : (a[idx] == value ? idx : find(a, value, idx + 1));
    }

    /**
     * \brief Find first occurence of a value from an array pointer
     *
     * \tparam U Value type
     * \tparam T Compile-time array type
     *
     * \param a Array
     * \param N String size
     * \param value Value to search for
     * \param idx Index to start search from
     *
     * \return The value index in the array if it is found, N otherwise
     */
    template<class T, class U>
    constexpr std::size_t find(const T* a, std::size_t N, const U& value, std::size_t idx = 0) noexcept {
        return idx >= N ? N : (a[idx] == value ? idx : find(a, N, value, idx + 1));
    }

    /**
     * \brief Find length of C-string
     *
     * This is a constexpr version of the strlen method.
     *
     * \param str String to measure
     * \param ind Index to start from
     */
    template<class CharT>
    constexpr std::size_t strlen(const CharT* str, std::size_t idx = 0) noexcept {
        return str[idx] == '\0' ? idx : strlen(str, idx+1);
    }

    template<class T>
    struct ct_array_traits {};

    template<>
    struct ct_array_traits<char> {
        enum {sentinel = '\0'};
    };

    template<class T>
    struct has_sentinel {
        template<class U, decltype(U::sentinel)* = nullptr>
        static constexpr bool check(U*)  {return true;}
        static constexpr bool check(...) {return false;}
        enum {value = check(static_cast<T*>(nullptr))};
    };


    template<
        class Trait,
        class T, class U, std::size_t N1, std::size_t N2,
        typename std::enable_if<has_sentinel<Trait>::value>::type* = nullptr
        >
    constexpr bool operator_equal_impl(
        const ct_array<T, N1>& a,
        const ct_array<U, N2>& b,
        const std::size_t idx = 0)
        noexcept
    {
        static_assert(N1 <= N2, "a is supposed to be the shortest array");
        return
            (idx == N1 && (N1 == N2 || b[idx] == Trait::sentinel))
            ||
            (a[idx] == b[idx]
             &&
             ((a[idx] == Trait::sentinel)
              || operator_equal_impl<Trait>(a, b, idx + 1)))
            ;
    }

    template<
        class Trait,
        class T, class U, std::size_t N1, std::size_t N2,
        typename std::enable_if<!has_sentinel<Trait>::value>::type* = nullptr
        >
    constexpr bool operator_equal_impl(
        const ct_array<T, N1>& a,
        const ct_array<U, N2>& b,
        const std::size_t idx = 0)
        noexcept
    {
        return
            N1 != N2 ? false :
            ((idx == N1) || (a[idx] == b[idx] && operator_equal_impl<Trait>(a, b, idx + 1)));
    }

    template<class T, class U, std::size_t N1, std::size_t N2,
             typename std::enable_if<(N1 <= N2)>::type* = nullptr>
    constexpr bool operator==(const ct_array<T, N1>& a, const ct_array<U, N2>& b) noexcept {
        return operator_equal_impl<ct_array_traits<T>>(a,b);
    }

    template<class T, class U, std::size_t N1, std::size_t N2,
             typename std::enable_if<(N1 > N2)>::type* = nullptr>
    constexpr bool operator==(const ct_array<T, N1>& a, const ct_array<U, N2>& b) noexcept {
        return operator_equal_impl<ct_array_traits<T>>(b,a);
    }

    template<class T, class U, std::size_t N1, std::size_t N2>
    constexpr bool operator==(const ct_array<T, N1>& a, const U(&b)[N2]) noexcept {
        return a == ct_array<U, N2>(b);
    }

    template<class T, class U, std::size_t N1, std::size_t N2>
    constexpr bool operator==(const T (&a)[N1],  const ct_array<U, N2>& b) noexcept {
        return ct_array<T, N1>(a) == b;
    }

    template<class T, class U, std::size_t N1, std::size_t N2>
    constexpr bool operator!=(const ct_array<T, N1>& a, const ct_array<U, N2>& b) noexcept {
        return !(a == b);
    }

    template<class T, class U, std::size_t N1, std::size_t N2>
    constexpr bool operator!=(const ct_array<T, N1>& a, const U(&b)[N2]) noexcept {
        return a != ct_array<U, N2>(b);
    }

    template<class T, class U, std::size_t N1, std::size_t N2>
    constexpr bool operator!=(const T (&a)[N1],  const ct_array<U, N2>& b) noexcept {
        return ct_array<T, N1>(a) != b;
    }

    template<class Trait, class T, class U, std::size_t N1, std::size_t N2,
             typename std::enable_if<has_sentinel<Trait>::value>::type* = nullptr
             >
    constexpr bool operator_less_impl(
        const ct_array<T, N1>& a,
        const ct_array<U, N2>& b,
        std::size_t idx = 0)
        noexcept
    {
        return
            ((idx == N1) && (N1 < N2)) ? true
            : ( idx == N2 ? false
                : ((a[idx] == Trait::sentinel && b[idx] != Trait::sentinel) ? true
                   : (b[idx] == Trait::sentinel ? false
                      : ((a[idx] < b[idx])
                         || (a[idx] == b[idx] && operator_less_impl<Trait>(a, b, idx+1)))
                       )))
            ;
    }


    template<class Trait, class T, class U, std::size_t N1, std::size_t N2,
             typename std::enable_if<!has_sentinel<Trait>::value>::type* = nullptr
             >
    constexpr bool operator_less_impl(
        const ct_array<T, N1>& a,
        const ct_array<U, N2>& b,
        std::size_t idx = 0)
        noexcept
    {
        return
            ((idx == N1) && (N1 <= N2)) ? true
            : ( idx == N2 ? false
                : ((a[idx] < b[idx])
                   || (a[idx] == b[idx] && operator_less_impl<Trait>(a, b, idx+1)))
                )
            ;
    }


    template<class T, class U, std::size_t N1, std::size_t N2>
    constexpr bool operator<(const ct_array<T, N1>& a, const ct_array<U, N2>& b) noexcept {
        return operator_less_impl<ct_array_traits<T>>(a,b);
    }

    template<class T, class U, std::size_t N1, std::size_t N2>
    constexpr bool operator>(const ct_array<T, N1>& a, const ct_array<U, N2>& b) noexcept {
        return b < a;
    }

    template<class T, class U, std::size_t N1, std::size_t N2>
    constexpr bool operator>=(const ct_array<T, N1>& a, const ct_array<U, N2>& b) noexcept {
        return !(a < b);
    }

    template<class T, class U, std::size_t N1, std::size_t N2>
    constexpr bool operator<=(const ct_array<T, N1>& a, const ct_array<U, N2>& b) noexcept {
        return !(b < a);
    }

}



#endif /* CT_ARRAY_HPP */
