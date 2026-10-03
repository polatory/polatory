#ifndef CPP_TOOLS_CL_PARSER_TCLI_SFINAE_HPP
#define CPP_TOOLS_CL_PARSER_TCLI_SFINAE_HPP

#include <istream>
#include <string>
#include <type_traits>
#include <vector>

namespace cpp_tools::cl_parser::meta
{
    /**
     * \brief Useful type alias for metaprogramming
     */
    template<class...>
    using void_t = void;

    /**
     * \brief Checks that all boolean values are true
     */
    template<bool B, bool... Bs>
    struct all_true
    {
        /** \internal */
        template<bool... Cs>
        struct list
        {
        };
        enum
        {
            value = std::is_same<list<B, Bs...>, list<Bs..., B>>::value
        };
    };

    /**
     * \brief Alias to ease use of std::enable_if
     *
     * \tparam T Trait to check, value is extracted
     * \tparam U Type given to std::enable_if
     */
    template<class T, class U = char>
    using use_if = typename std::enable_if<T::value, U>::type;

    /**
     * \copydoc use_if
     */
    template<class T, class U = char>
    using not_use_if = typename std::enable_if<!T::value, U>::type;

    /**
     * \brief Checks for the compile time static value T::stackable
     *
     * \tparam T Type to inspect
     */
    template<class T>
    struct is_stackable
    {
        template<class U, int = U::stackable>
        static constexpr bool check(U*)
        {
            return true;
        }
        static constexpr bool check(...) { return false; }

        enum
        {
            value = check((T*)0)
        };
    };

    /**
     * \brief Checks for the compile time static value T::tcli_parser
     *
     * \tparam T Type to inspect
     */
    template<class T>
    struct is_parser
    {
        template<class U, int = U::tcli_parser>
        static constexpr bool check(U*)
        {
            return true;
        }
        static constexpr bool check(...) { return false; }

        enum
        {
            value = check((T*)0)
        };
    };

    /**
     * \brief Checks for the compile time static value T::flagged
     *
     * \tparam T Type to inspect
     */
    template<class T>
    struct is_flagged
    {
        template<class U, int = U::flagged>
        static constexpr bool check(U*)
        {
            return true;
        }
        static constexpr bool check(...) { return false; }

        enum
        {
            value = check((T*)0)
        };
    };

    /**
     * \brief Checks for the compile time static value T::required
     *
     * \tparam T Type to inspect
     */
    template<class T>
    struct is_required
    {
        template<class U, int = U::required>
        static constexpr bool check(U*)
        {
            return true;
        }
        static constexpr bool check(...) { return false; }

        enum
        {
            value = check((T*)0)
        };
    };

    namespace details
    {
        /**
         * \copydoc is_istream_settable
         */
        template<class T>
        struct has_formated_input_impl
        {
            template<class U>
            constexpr static auto check(U* u, std::istream* is = nullptr) -> decltype((*is >> *u), void(), true)
            {
                return true;
            }
            static constexpr bool check(...) { return false; }
            enum
            {
                value = check((T*)0)
            };
        };

    }   // namespace details

    /**
     * \brief Checks whether T has a formatted input operator
     *
     * \tparam T Type to inspect
     */
    template<class T>
    struct is_istream_settable : std::integral_constant<bool, details::has_formated_input_impl<T>::value>
    {
    };

    /**
     * \brief Checks whether T has a `type` type alias
     *
     * \tparam T Type to inspect
     */
    template<class T>
    struct has_type
    {
        template<typename U>
        static constexpr bool check(U*, typename U::type* = nullptr)
        {
            return true;
        }
        static constexpr bool check(...) { return false; }

        enum
        {
            value = check((T*)0)
        };
    };

    /**
     * \brief Checks whether T has a `parse` static method
     *
     * \tparam T    Type to inspect
     * \tparam Func Function signature
     *
     * Example:
     *
     * ~~~{.cpp}
     * struct S {
     *     static int parse(int, char**);
     * };
     *
     * std::cout << has_parse<S, int(int, char**)>::value; // true
     * ~~~
     */
    template<class T, class Func>
    struct has_parse;

    /**
     * \copydoc inria::tcli::meta::has_parse
     */
    template<class T, class Ret, class... Args>
    struct has_parse<T, Ret(Args...)>
    {
        template<class, class = void>
        struct check : std::false_type
        {
        };

        template<class U>
        struct check<U, void_t<decltype(U::parse(std::declval<Args>()...))>> : std::true_type
        {
        };

        enum
        {
            value = check<T>::value
        };
    };

    /**
     * \brief Checks whether T has a `visit` static method
     *
     * \tparam T    Type to inspect
     * \tparam Func Function signature
     *
     * Example:
     *
     * ~~~{.cpp}
     * struct S {
     *     static int visit(int, char**);
     * };
     *
     * std::cout << has_visit<S, int(int, char**)>::value; // true
     * ~~~
     */
    template<class T, class Func>
    struct has_visit;

    /**
     * \copydoc has_visit
     */
    template<class T, class Ret, class... Args>
    struct has_visit<T, Ret(Args...)>
    {
        template<class, class = void>
        struct check : std::false_type
        {
        };

        template<class U>
        struct check<U, void_t<decltype(std::declval<U>().visit(std::declval<Args>()...))>> : std::true_type
        {
        };

        enum
        {
            value = check<T>::value
        };
    };

    /**
     * \brief Checks attribute `T::def` existence
     *
     * \tparam T Type to inspect
     */
    template<class T>
    struct has_default
    {
        template<class U, decltype(std::declval<U>().def)* = nullptr>
        static constexpr std::true_type check(U*)
        {
            return {};
        }
        static constexpr std::false_type check(...) { return {}; }

        enum
        {
            value = decltype(check((T*)nullptr))::value
        };
    };

    /**
     * \brief Metaprogramming type list
     *
     * \tparam Elts Elements pack
     */
    template<class... Elts>
    struct list
    {
        /// List element count
        static constexpr int size = sizeof...(Elts);

        /**
         * \brief Find first occurence of an element in the list
         *
         * \tparam U  Element to find
         * \tparam Vs Remaining elements in the list
         */
        template<class U, class... Vs>
        struct find
        {
            /// Index of the element, default value is past the list end
            static constexpr int value = size + 1;
        };

        /**
         * \brief Find element in the list, success specialisation
         *
         * \copydetails find
         */
        template<class U, class... Vs>
        struct find<U, U, Vs...>
        {
            static constexpr int value = size - sizeof...(Vs) - 1;
        };

        /**
         * \brief Find element in the list, recursion specialisation
         *
         * \copydetails find
         * \tparam V First element, different from `U`
         */
        template<class U, class V, class... Vs>
        struct find<U, V, Vs...>
        {
            static constexpr int value = find<U, Vs...>::value;
        };

        /**
         * \brief Get element index in the list
         *
         * \tparam T Element to find
         */
        template<class T>
        struct index
        {
            /// Index of element, `list<Elts>::size+1` if the element does not exist
            static constexpr int value = find<T, Elts...>::value;
        };

        /**
         * \brief Check for type in list
         *
         * \tparam T Type to look for in the list
         */
        template<class T>
        struct exists
        {
            // Intel compiler requires the parentheses
            static constexpr int value = (find<T, Elts...>::value) < size;
        };
    };

}   // namespace cpptools::cl_parser::meta

#endif /* TCLI_SFINAE_HPP */
