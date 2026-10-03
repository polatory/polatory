#ifndef _IO_HPP_
#define _IO_HPP_

#include <ostream>
#include <cassert>

namespace inria {
namespace io {
namespace details {

/**
 * \brief Print configuration
 */
template<class Separator, class Open, class Close>
struct io_range_data {
    const Separator& sep;
    const Open& open;
    const Close& close;
    // Clang segfaults if the constructor is not defined
    io_range_data(const Separator& s, const Open& o, const Close& c) :
        sep(s), open(o), close(c) {}
};

template<class C, class Separator, class Open, class Close,
         bool is_const, bool is_rvalue_ref>
struct io_range_t;

/**
 * \brief Keep a reference to range container
 */
template<class C, class Separator, class Open, class Close>
struct io_range_t<C, Separator, Open, Close, false, false> {
    C& c;
    io_range_data<Separator, Open, Close> data;
    enum {reference_range};
};

/**
 * \brief Keep a const reference to range container
 */
template<class C, class Separator, class Open, class Close>
struct io_range_t<C, Separator, Open, Close, true, false> {
    const C& c;
    io_range_data<Separator, Open, Close> data;
    enum {const_reference_range};
};

/**
 * \brief Store the container when printing an rvalue container
 */
template<class C, class Separator, class Open, class Close, bool B>
struct io_range_t<C, Separator, Open, Close, B, true> {
    C c;
    io_range_data<Separator, Open, Close> data;
    enum {value_range};
};

/**
 * \brief Formatted range output operator
 */
template<class C, class Separator, class Open, class Close, bool B, bool B2>
std::ostream& operator<<(std::ostream& os,
                         const io_range_t<C, Separator, Open, Close, B, B2>& c)
{
    auto b = std::begin(c.c);
    const auto e = std::end(c.c);
    os << c.data.open;
    if(!(b == e)) {
        os << *b;
        ++b;
    }
    while(!(b == e)) {
        os << c.data.sep;
        os << *b;
        ++b;
    }
    os << c.data.close;
    return os;
}

} // end namespace details

/**
 * \brief Create an object from a range that can be printed
 */
template<class C, class Separator = const char*, class Open = const char*, class Close = const char*>
auto range(C&& c, const Separator& sep = ", ", const Open& open = "", const Close& close = "")
    -> details::io_range_t<
        C, Separator, Open, Close,
        std::is_const<typename std::remove_reference<C>::type>::value,
        std::is_rvalue_reference<decltype(std::forward<C>(c))>::value>
{
    return details::io_range_t<
        C, Separator, Open, Close,
        std::is_const<typename std::remove_reference<C>::type>::value,
        std::is_rvalue_reference<decltype(std::forward<C>(c))>::value>
    {std::forward<C>(c), {{sep}, {open}, {close}}};
}


}} // end namespace inria[::io]


#endif /* _IO_HPP_ */
