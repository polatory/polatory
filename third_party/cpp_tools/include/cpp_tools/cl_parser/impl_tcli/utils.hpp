#ifndef CPP_TOOLS_CL_PARSER_UTILS_HPP
#define CPP_TOOLS_CL_PARSER_UTILS_HPP

#include <tuple>
#include <utility>

namespace cpp_tools::cl_parser::utils
{
    /**
     * \brief Collective operation result
     *
     * Collective operations on a tuple are made by calling a functor with a
     * state over all elements. If the functor defines a `collective_result()`
     * method, get its returned value.
     *
     * \tparam F functor type, argument-deduced
     *
     * \param functor Callable object to get a result from
     *
     * \return `functor.collective_result()`
     */
    template<class F>
    auto get_collective_result(F&& functor) -> decltype(functor.collective_result())
    {
        return functor.collective_result();
    }

    /**
     * \brief Collective operation result fallback
     *
     * If not `collective_result()` method is defined by the arguments, this
     * overload is called. This is a no-op.
     */
    inline void get_collective_result(...) {}

    /**
     * \brief for_each_in_tuple implementation
     */
    template<class Tuple, class F, class... Args, std::size_t... Is>
    auto for_each_in_tuple_impl(std::index_sequence<Is...>, Tuple&& t, F&& func, Args&&... args)
      -> decltype(get_collective_result(func))
    {
        auto l = {0, (std::forward<F>(func)(std::get<Is>(std::forward<Tuple>(t)), std::forward<Args>(args)...), 0)...};
        (void)l;
        return get_collective_result(func);
    }

    /**
     * \brief Calls a functor over the elements of a tuple
     *
     * \tparam Tuple Tuple type
     * \tparam F Functor type
     * \tparam Args Additional arguments to pass to the functor
     *
     * \param t Tuple to iterate over
     * \param func Functor to call
     * \param args Additional arguments
     *
     * \return `f.collective_result()` if it exists, `void` otherwise.
     *
     * Example (C++14):
     * ```{cpp}
     * std::tuple<int, double, char> t(1, 3.14, 'c');
     * // The following
     * for_each_in_tuple(t, [](auto a, char delim){std::cout << a << delim;}, ':')
     * // is equivalent to
     * std::cout << std::get<0>(t) << ':';
     * std::cout << std::get<1>(t) << ':';
     * std::cout << std::get<2>(t) << ':';
     * ```
     *
     * Output:
     * ```
     * 1:3.14:c
     * ```
     */
    template<class Tuple, class F, class... Args>
    auto for_each_in_tuple(Tuple&& t, F&& func, Args&&... args)
#ifndef __INTEL_COMPILER
      -> decltype(
        for_each_in_tuple_impl(std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>(),
                               std::forward<Tuple>(t), std::forward<F>(func), std::forward<Args>(args)...))
#endif
    {
        return for_each_in_tuple_impl(
          std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>(), std::forward<Tuple>(t),
          std::forward<F>(func), std::forward<Args>(args)...);
    }

}   // namespace cpptools::cl_parser::utils

#endif /* TCLI_UTILS_HPP */
