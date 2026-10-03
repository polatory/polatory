// --------------------------------
// See LICENCE file at project root
// File : scalfmm/container/iterator.hpp
// --------------------------------
#ifndef SCALFMM_CONTAINER_ITERATOR_HPP
#define SCALFMM_CONTAINER_ITERATOR_HPP

#include <cstddef>
#include <iterator>

namespace scalfmm::container
{
    /**
     * @brief Forward declaration for traits support (proxy_iterator).
     *
     * @tparam VariadicContainer
     * @tparam DerivedVariadic
     * @tparam Seq
     * @tparam IsConst
     */
    template<class VariadicContainer, class DerivedVariadic, typename Seq, bool IsConst>
    class proxy_iterator;

    /**
     * @brief Forward declaration for traits support (variadic_adaptor).
     *
     * @tparam Derived
     * @tparam Containers
     */
    template<typename Derived, typename... Containers>
    struct variadic_adaptor;

    /**
     * @brief Forward declaration for traits support (particle_container).
     *
     * @tparam Particle
     */
    template<typename Particle>
    class particle_container;

    /**
     * @brief Forward declaration for traits support (iterator_traits).
     *
     * @tparam Iterator
     */
    template<typename Iterator>
    struct iterator_traits;

    /**
     * @brief
     *
     * @tparam Seq
     * @tparam IsConst
     * @tparam Derived
     * @tparam Containers
     */
    template<typename Seq, bool IsConst, typename Derived, typename... Containers>
    struct iterator_traits<
      proxy_iterator<variadic_adaptor<Derived, Containers...>, variadic_adaptor<Derived, Containers...>, Seq, IsConst>>
      : public std::iterator_traits<proxy_iterator<variadic_adaptor<Derived, Containers...>,
                                                   variadic_adaptor<Derived, Containers...>, Seq, IsConst>>
    {
        using container_type = variadic_adaptor<Derived, Containers...>;
    };

    /**
     * @brief
     *
     * @tparam Seq
     * @tparam IsConst
     * @tparam Particle
     * @tparam Derived
     * @tparam Containers
     */
    template<typename Seq, bool IsConst, typename Particle, typename Derived, typename... Containers>
    struct iterator_traits<
      proxy_iterator<variadic_adaptor<Derived, Containers...>, particle_container<Particle>, Seq, IsConst>>
      : public std::iterator_traits<
          proxy_iterator<variadic_adaptor<Derived, Containers...>, particle_container<Particle>, Seq, IsConst>>
    {
        using container_type = particle_container<Particle>;
        using particle_type = typename container_type::particle_type;
        using tuple_type = typename container_type::tuple_type;
        using position_value_type = typename particle_type::position_value_type;
        static constexpr std::size_t dimension = particle_type::dimension;
        static constexpr std::size_t dimension_size = particle_type::dimension_size;
        using position_type = typename particle_type::position_type;
        using position_tuple_type = typename particle_type::position_tuple_type;
        using range_position_type = typename particle_type::range_position_type;

        using inputs_value_type = typename particle_type::inputs_value_type;
        static constexpr std::size_t inputs_size = particle_type::inputs_size;
        using inputs_type = typename particle_type::inputs_type;
        using inputs_tuple_type = typename particle_type::inputs_tuple_type;
        using range_inputs_type = typename particle_type::range_inputs_type;

        using outputs_value_type = typename particle_type::outputs_value_type;
        static constexpr std::size_t outputs_size = particle_type::outputs_size;
        using outputs_type = typename particle_type::outputs_type;
        using outputs_tuple_type = typename particle_type::outputs_tuple_type;
        using range_outputs_type = typename particle_type::range_outputs_type;

        using variables_type = typename particle_type::variables_type;
        static constexpr std::size_t variables_size = particle_type::variables_size;
        using range_variables_type = typename particle_type::range_variables_type;
    };

}   // namespace scalfmm::container

#endif   // SCALFMM_CONTAINER_ITERATOR_HPP
