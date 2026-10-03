#include <array>
#include <cstring>
#include <ostream>
#include <istream>
#include <limits>
#include <type_traits>
#include <sstream>

#include "ct_array.hpp"

#pragma push_macro("versioning")
#undef major
#undef minor

/**
 * \file
 * \brief Constexpr implementation of semantic versioning.
 *
 * ```
 *
 * version 0.1.0:
 *   - extend constructor to \0 terminated C strings
 *   - add constructor from std::string
 * version 0.0.0:
 *   - initial code
 * ```
 *
 *
 *
 */


namespace inria {

namespace detail {

    struct parse_cursor {
        const char* str;
        std::size_t n;
    };

    /**
     * \brief Parse a integer in constant string
     *
     * \param c Cursor used to start parsing
     *
     * \return A cusor pointing after the end of the parsed number and
     * containing the parsed integer
     */
    constexpr parse_cursor parse_number(const parse_cursor c) {
        return (c.str[0] < '0' || c.str[0] > '9') ? parse_cursor{c.str+1, c.n} :
            parse_number({c.str+1, ((c.str[0] - '0') + 10 * c.n)});
    }

    /**
     * \brief Find nth element in a '\0' delimited array
     *
     * Looks for the nth required value until an element that compares equal to '\0'
     * is found.
     *
     * \tparam CharT Array type
     *
     * \param str Array to search into
     * \param c Value to look for
     * \param n Count of values to skip
     * \param idx Offset to start from
     *
     * \return The index of the nth value if found, otherwise the length of the array.
     */
    template<class CharT>
    constexpr std::size_t find_nth(const CharT* str, CharT c, std::size_t n, std::size_t idx = 0) {
        return (str[idx] == '\0') ?
            idx : ((str[idx] == c) ?
                 (n <= 1 ? idx : find_nth(str, c, n-1, idx+1)) : find_nth(str, c, n, idx+1));
    }

    /**
     * \brief Lexicographic comparison recursion end
     *
     * \return false
     */
    constexpr static bool compare_less() {
        return false;
    }
    /**
     * \brief Lexicographic comparison
     *
     * Compares successive pairs of values until a pair compares not equal.
     *
     * \tparam T Pair type
     * \tparam Ts Following arguments
     *
     * \param a First element of the current pair
     * \param b Second element of the current pair
     * \param ts Other elements to compare
     *
     * \return `true` if `a < b`, `false` if `b < a`. Otherwise, the result of `compare_less(ts...)`
     */
    template<class T, class...Ts>
    constexpr static bool compare_less(const T& a, const T& b, const Ts&... ts) {
        return (a < b) || ((a == b) && compare_less(ts...));
    }

}

/**
 * \brief `constexpr` semantic versioning implementation
 *
 * This class implements constexpr semantic versioning.
 */
struct version {
    std::size_t major; //< Major version number
    std::size_t minor; //< Minor version number
    std::size_t patch; //< Patch version number
    /** Pre-release specification string
     *
     * Limited to 32 characters
     */
    inria::ct_array<char, 32> pre_release;
    /** Build info string
     *
     * Limited to 32 characters
     *
     * \warning The build metadata is never taken into acount when comparing
     * versions.
     */
    inria::ct_array<char, 32> build;

    version() = default;
    version(const version&) = default;
    version& operator=(const version&) = default;
    version(version&&) = default;
    version& operator=(version&&) = default;


    /**
     * Constructor
     *
     * \param _major Major version number
     * \param _minor Minor version number
     */
    template<class CharT = char, std::size_t N1 = 1, std::size_t N2 = 1>
    constexpr version(std::size_t _major, std::size_t _minor = 0,
                      std::size_t _patch = 0,
                      const CharT (&_pre_release)[N1] = "", const CharT (&_build)[N2] = "")
        : major(_major), minor(_minor), patch(_patch),
          pre_release(_pre_release), build(_build)
    {}

    template<class CharT>
    constexpr version(const CharT* str, std::size_t len)
        : version(detail::parse_number({str,0}).n,
                  detail::parse_number({str+detail::find_nth(str, '.', 1)+1,0}).n,
                  detail::parse_number({str+detail::find_nth(str, '.', 2)+1,0}).n,
                  inria::ct_array<char, 32>(str,len).sub_array(
                      inria::find(str, len, '-')+1,
                      inria::find(str, len, '+') - inria::find(str,len, '-')).as_array(),
                  inria::ct_array<char, 32>(str, len).sub_array(
                      inria::find(str, len, '+')+1).as_array()
            )
    {}

    template<class CharT>
    constexpr version(const CharT* str) : version(str, inria::strlen(str)) {}

    version(const std::string& str) : version(str.c_str(), str.size()) {}

    operator std::string () const {
        std::stringstream sstr;
        sstr << *this;
        return sstr.str();
    }


    friend std::ostream& operator<<(std::ostream& os, const version& v) noexcept {
        os << v.major << '.' << v.minor << '.' << v.patch;
        if(std::strcmp(v.pre_release.data(), "")) {
            os << '-' << v.pre_release.data();
        }
        if(std::strcmp(v.build.data(), "")) {
            os << '+' << v.build.data();
        }
        return os;
    }

    friend std::istream& operator>>(std::istream& is, version& v) noexcept {
        auto check_alphanum = [](const char c) {
            return (c >= '0' && c <= '9')
            || (c >= 'a' && c <= 'z')
            || (c >= 'A' && c <= 'Z')
            || (c == '.')
            || (c == '-');
        };
        char c;
        is >> v.major >> c >> v.minor >> c >> v.patch;
        if(is && is.peek() == '-') {
            is.get();
            int i = 0;
            while(i < 31
                  && ((c = is.get()), true) // assign c to next character
                  && is
                  && check_alphanum(c) )
            {
                v.pre_release.as_array()[i] = c;
                ++i;
            }
            if(is) {
                is.unget();
            }
            v.pre_release.as_array()[i] = '\0';
        }
        if(is && is.peek() == '+') {
            is.get();
            int i = 0;
            while(i < 31
                  && ((c = is.get()), true) // assign c to next character
                  && is
                  && check_alphanum(c))
            {
                v.build.as_array()[i] = c;
                ++i;
            }
            v.build.as_array()[i] = '\0';
        }
        return is;
    }

    friend constexpr bool operator==(const version& v1, const version& v2) noexcept {
        return v1.major == v2.major
            && v1.minor == v2.minor
            && v1.patch == v2.patch
            && v1.pre_release == v2.pre_release
            ;
    }

    friend constexpr bool operator<(const version& v1, const version& v2) noexcept {
        return detail::compare_less(
            v1.major, v2.major,
            v1.minor, v2.minor,
            v1.patch, v2.patch)
            || (v1.pre_release != "" && v2.pre_release == "")
            || v1.pre_release < v2.pre_release
            ;
    }

    friend constexpr bool operator!=(const version& v1, const version& v2) noexcept {
        return !(v1 == v2);
    }

    friend constexpr bool operator>=(const version& v1, const version& v2) noexcept {
        return !(v1 < v2);
    }

    friend constexpr bool operator>(const version& v1, const version& v2) noexcept {
        return v2 < v1;
    }

    friend constexpr bool operator<=(const version& v1, const version& v2) noexcept {
        return !(v2 < v1);
    }
};

}
#pragma pop_macro("versioning")
