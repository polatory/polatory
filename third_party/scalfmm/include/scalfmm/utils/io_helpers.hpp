// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/io_helpers.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_OSTREAM_TUPLE_HPP
#define SCALFMM_UTILS_OSTREAM_TUPLE_HPP

#include "inria/integer_sequence.hpp"
#include "inria/ostream_joiner.hpp"

#include "scalfmm/meta/utils.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>

namespace scalfmm
{
    namespace details
    {
        namespace tuple_helper
        {
            /**
             * @brief Helper for tuple formatted output
             *
             * @tparam Types Types contained in the tuple, automatically deduced
             * @tparam Indices Type indices, automatically deduced
             *
             * @param os Output stream
             * @param t Tuple to print
             */
            template<typename... Types, std::size_t... Indices>
            inline auto formated_output_old(std::ostream& os, const std::tuple<Types...>& t,
                                            inria::index_sequence<Indices...>) -> void
            {
                os << "(";   // opening brace
                             // initializer_list members are evaluated from left to right; this
                             // evaluates to an equivalent to:
                             //
                             //     os << std::get<0>(t) << ", ";
                             //     os << std::get<1>(t) << ", "
                             //     ...
                             //     std::get<N>(t) << "";
                //   auto l = {(os << std::get<Indices>(t) << (Indices != sizeof...(Types) - 1 ? ", " : ""), 0)...};
                //  (void)l;     // ignore fact that initializer_list is not used
                (..., (os << (Indices == 0 ? "" : ", ") << std::get<Indices>(t)));
                os << ")";   // closing brace
            }

            /**
             * @brief Print formated tuple
             *
             * @tparam Tuple
             * @tparam Indices
             * @param os the stream
             * @param t the tuple
             */
            template<typename Tuple, std::size_t... Indices>
            inline auto formated_output(std::ostream& os, const Tuple& t, inria::index_sequence<Indices...>) -> void
            {
                os << "[";
                (..., (os << (Indices == 0 ? "" : ", ") << std::get<Indices>(t)));
                os << "]";   // closing brace
            }

            /**
             * @brief
             *
             * @tparam Ts
             */
            template<class... Ts>
            struct tuple_wrapper
            {
                /**
                 * @brief
                 *
                 */
                const std::tuple<Ts...>& t;
                friend std::ostream& operator<<(std::ostream& os, const tuple_wrapper& w)
                {
                    using details::tuple_helper::formated_output;
                    formated_output(os, w.t, inria::index_sequence_for<Ts...>{});
                    return os;
                }
            };

        }   // namespace tuple_helper
    }   // namespace details

    namespace
    {
        /**
         * @brief
         *
         */
        constexpr struct
        {
            /**
             * @brief
             *
             * @tparam Ts
             * @param t
             * @return details::tuple_helper::tuple_wrapper<Ts...>
             */
            template<class... Ts>
            details::tuple_helper::tuple_wrapper<Ts...> operator()(const std::tuple<Ts...>& t) const
            {
                return {t};
            }
        } tuple_out{};

    }   // namespace

    namespace details
    {
        /**
         * @brief Silence the `unused variable` warnings about tuple_out
         *
         * @tparam T
         */
        template<class T>
        inline auto silence_tuple_out_warning() -> void
        {
            tuple_out(std::tuple<>{});
        }

    }   // namespace details

    namespace io
    {
        /**
         * @brief operator << for array std::array<T,N>
         *
         * using namespace io ;
         * std::array(int, 2) a;
         * std::cout << a << std::endl ;
         * print array [a_1, ..., a_N]
         *
         * @tparam T
         * @tparam N
         * @param os
         * @param array
         * @return std::ostream&
         */
        template<typename T, std::size_t N>
        inline auto operator<<(std::ostream& os, const std::array<T, N>& array) -> std::ostream&
        {
            os << "[";
            for(auto it = array.begin(); it != array.end() - 1; it++)
            {
                os << *it << ", ";
            }
            os << array.back() << "]";
            return os;
        }

        /**
         * @brief print a vector and its size
         *
         * The output is
         *  title (size) values
         *
         * The values are separated by a white space
         *
         * @tparam VectorType
         * @param title
         * @param v
         */
        template<typename VectorType>
        inline auto print(const std::string&& title, VectorType const& v) -> void
        {
            std::cout << title << " (" << v.size() << ") ";
            if(v.size() > 0)
            {
                for(auto& i: v)
                    std::cout << i << ' ';
            }
            std::cout << '\n';
        }

        /**
         * @brief print print a vector and its size starting at first and ending at end
         *
         * The output is
         *  title (size) [value1 value2 ...]
         *
         * The values are separated by a white space
         *
         * @tparam IteratorType
         * @param out
         * @param title string to print
         * @param first begin iterator
         * @param last  end iterator
         * @param sep
         */
        template<typename IteratorType>
        inline auto print(std::ostream& out, const std::string&& title, IteratorType first, IteratorType last,
                          std::string&& sep = " ") -> void
        {
            auto size = std::distance(first, last);

            out << title << " (" << size << ") [";
            if(size)
            {
                auto lastm1 = --last;
                IteratorType it;
                for(it = first; it != lastm1; ++it)
                    out << *it << sep;
                out << *it;
            }
            out << ']';
        }

        // template<typename IteratorType>
        // void print(std::ostream& out, const std::string&& title, IteratorType first, IteratorType last,
        //            std::string&& first_str = "[", std::string&& sep = " ", std::string&& last_str = "]")
        // {
        //     auto size = std::distance(first, last);

        //     out << title << " (" << size << ") " + first_str;
        //     if(size)
        //     {
        //         auto lastm1 = --last;
        //         IteratorType it;
        //         for(it = first; it != lastm1; ++it)
        //             out << *it << sep;
        //         out << *it;
        //     }
        //     out << last_str;
        // };

        /**
         * @brief
         *
         * @tparam VectorType
         * @param out
         * @param title
         * @param v
         * @param sep
         */
        template<typename VectorType>
        inline auto print(std::ostream& out, const std::string&& title, VectorType& v, std::string&& sep = " ") -> void
        {
            print(out, std::move(title), v.begin(), v.end(), std::move(sep));
            out << std::endl;
        }

        /**
         * @brief Print a part or a full array
         *
         * print the array array[0:size]
         *
         * @tparam ArrayType type of teh array
         * @param array the array to print
         * @param size  the list of elements to print
         */
        template<typename ArrayType>
        inline auto print_array(ArrayType const& array, const int& size) -> void
        {
            std::cout << "[ " << array[0];
            for(int i{1}; i < size; ++i)
            {
                std::cout << ", " << array[i];
            }
            std::cout << " ]";
        }

        /**
         * @brief
         *
         * @tparam T
         * @tparam N
         * @param out
         * @param a
         */
        template<typename T, std::size_t N>
        inline auto print(std::ostream& out, std::array<T, N> const& a) -> void
        {
            out << "[";
            for(std::size_t i{0}; i < N; ++i)
            {
                out << (i == 0 ? "" : ", ") << a[i];
            }
            out << "]";
        }

        /**
         * @brief print a tuple
         *
         * std::tuple<int, double, int> a{1,0.3,8};
         *  io::print(a)
         *  the output is [1, 0.3, 8]
         * 
         * @tparam T 
         * @param out the stream
         * @param tup the tuple
         */
        template<typename... T>
        inline auto print(std::ostream& out, const std::tuple<T...>& tup) -> void
        {
            scalfmm::details::tuple_helper::formated_output(out, tup, std::make_index_sequence<sizeof...(T)>());
        }

        /**
         * @brief print a sequence (tuple, array)
         *  tuple<int, double> t{3,0.5}
         *  std::cout << t << std::endl shows [3, 0.5, ]
         *
         * @tparam Seq
         * @param[inout] out the stream
         * @param[in] seq the sequence to print
         * @return auto
         */
        template<typename Seq>
        inline auto print_seq(std::ostream& out, Seq const& seq) -> void
        {
            out << "[";
            int i{0};
            meta::for_each(seq, [&out, &i](const auto& s) { out << (i++ == 0 ? "" : ", ") << s; });
            out << "]";
        }

        /**
         * @brief print the address ot the element of a sequence (tuple, array)
         *
         *  tuple<int, double> t{3,0.5}
         *  std::cout << t << std::endl
         *
         * @tparam Seq
         * @param[inout] out the stream
         * @param[in] seq the sequence to print
         */
        template<typename Seq>
        inline auto print_ptr_seq(std::ostream& out, const Seq& seq) -> void
        {
            out << "[";
            int i{0};
            meta::for_each(seq, [&out, &i](const auto& s) { out << (i++ == 0 ? "" : ", ") << &s; });
            out << "]";
        }

    }   // namespace io

}   // namespace scalfmm

#endif
