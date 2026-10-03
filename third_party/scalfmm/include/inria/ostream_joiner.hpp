#ifndef OSTREAM_JOINER_HPP
#define OSTREAM_JOINER_HPP

#include <string>
#include <ostream>

#include "integer_sequence.hpp"

namespace inria {

    /**
     * \brief Fake iterator to output range values using standard algorithms
     *
     * This iterator is used to output several values separated using a
     * delimiter.
     *
     * \tparam DelimT The delimiter type
     * \tparam CharT The std::basic_ostream character type
     * \tparam Traits The std::basic_ostream character traits type
     *
     * \note See inria::make_ostream_joiner for an easy way to create an new
     * joiner.
     *
     * ### Example
     * ~~~~{.cpp}
     * #include <iostream>
     * #include <algorithm>
     * #include <vector>
     *
     * #include "Utils/Contribs/inria/ostream_joiner.hpp"
     *
     * int main() {
     *     std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7};
     *     auto oj = inria::make_ostream_joiner(std::cout, "::");
     *     std::copy(vec.begin(), vec.end(), oj);
     *     // outputs: "1::2::3::4::5::6::7"
     * }
     * ~~~~
     *
     */
    template<typename DelimT, typename CharT = char, typename Traits = std::char_traits<CharT> >
    class ostream_joiner {
    public:
        using char_type = CharT;
        using traits_type = Traits;
        using ostream_type = std::basic_ostream<CharT, Traits>;

        using value_type = void;
        using difference_type = void;
        using pointer = void;
        using reference = void;
        using iterator_category = std::output_iterator_tag;

        /**
         * \brief Builds a new ostream_joiner
         *
         * \param stream The output stream to be accessed by this iterator
         * \param delimiter The delimiter to be instered into the stram in
         * between two outputs
         */
        ostream_joiner(ostream_type& stream, const DelimT& delimiter) :
            _os(std::addressof(stream)),
            _delim(delimiter)
        {}

        /**
         * \brief Builds a new ostream_joiner
         *
         * \param stream The output stream to be accessed by this iterator
         * \param delimiter The delimiter to be instered into the stram in
         * between two outputs
         */
        ostream_joiner(ostream_type& stream, DelimT&& delimiter) :
            _os(std::addressof(stream)),
            _delim(std::move(delimiter))
        {}

        ostream_joiner(const ostream_joiner&) = default;
        ostream_joiner(ostream_joiner&&) = default;
        ostream_joiner& operator=(const ostream_joiner&) = default;
        ostream_joiner& operator=(ostream_joiner&&) = default;

        /**
         * \brief Writes the value to the stream
         *
         * If a value was previously output, inserts a delimiter.
         *
         * \tparam T Type of the value
         *
         * \param value Value to output
         *
         * \return *this
         */
        template<typename T>
        ostream_joiner& operator=(const T& value) {
            if(! this->_first) {
                *(this->_os) << this->_delim;
            }
            this->_first = false;
            *(this->_os) << value;
            return *this;
        }

        /**
         * \brief Does nothing
         * \return *this
         */
        ostream_joiner& operator*() {
            return *this;
        }

        /**
         * \brief Does nothing
         * \return *this
         */
        ostream_joiner& operator++() {
            return *this;
        }

        /**
         * \brief Does nothing
         * \return *this
         */
        ostream_joiner& operator++(int) {
            return *this;
        }

    private:
        /// Pointer to the given output stream
        ostream_type* const _os;
        /// Delimiter to insert between output elements
        const DelimT _delim;
        /// Flag fo first output execution
        bool _first = true;
    };


    /** \brief Create ostream_joiner from arguments
     *
     * Automatically deduces the ostream_joiner type from the arguments
     *
     * \param os Output stream to give to the iterator constructor
     * \param delimiter The delimiter
     *
     * \return A new ostream_joiner object
     */
    template<typename DelimT, typename CharT, typename Traits>
    ostream_joiner<typename std::decay<DelimT>::type, CharT, Traits>
    make_ostream_joiner(std::basic_ostream<CharT, Traits>& os, DelimT&& delimiter) {
        return {os, std::forward<DelimT>(delimiter)};
    }

}

#endif /* OSTREAM_JOINER_HPP */
