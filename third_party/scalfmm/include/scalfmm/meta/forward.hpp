// --------------------------------
// See LICENCE file at project root
// File : scalfmm/meta/forward.hpp
// --------------------------------
#ifndef SCALFMM_META_FORWARD_HPP
#define SCALFMM_META_FORWARD_HPP

// Forward declaration for traits support
#include <cstddef>

namespace scalfmm::container
{
    /**
     * @brief Forward declaration for 'variadic_adaptor'.
     *
     * @tparam Derived
     * @tparam Containers
     */
    template<typename Derived, typename... Containers>
    struct variadic_adaptor;

    /**
     * @brief Forward declaration for 'unique_variadic_container'.
     *
     * @tparam Derived
     * @tparam Container
     * @tparam Types
     */
    template<typename Derived, template<typename U, typename Allocator> class Container, typename... Types>
    struct unique_variadic_container;

    /**
     * @brief Forward declaration for 'variadic_container'.
     *
     * @tparam Derived
     * @tparam Types
     */
    template<typename Derived, typename... Types>
    struct variadic_container;

    /**
     * @brief Forward declaration for 'variadic_container_tuple'.
     *
     * @tparam Derived
     * @tparam Tuple
     */
    template<typename Derived, typename Tuple>
    struct variadic_container_tuple;

    /**
     * @brief Forward declaration for 'point_impl'.
     *
     * @tparam ValueType
     * @tparam Dimension
     */
    template<typename ValueType, std::size_t Dimension>
    struct point_impl;

    /**
     * @brief Forward declaration for 'point_proxy'.
     *
     * @tparam ValueType
     * @tparam Dimension
     */
    template<typename ValueType, std::size_t Dimension>
    struct point_proxy;

    /**
     * @brief Forward declaration for 'point'.
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam Enable
     */
    template<typename ValueType, std::size_t Dimension, typename Enable>
    struct point;

    /**
     * @brief Forward declaration for 'particle'.
     *
     * @tparam PositionType
     * @tparam PositionDim
     * @tparam InputsType
     * @tparam NInputs
     * @tparam OutputsType
     * @tparam MOutputs
     * @tparam Variables
     */
    template<typename PositionType, std::size_t PositionDim, typename InputsType, std::size_t NInputs,
             typename OutputsType, std::size_t MOutputs, typename... Variables>
    struct particle;
}   // namespace scalfmm::container

#endif   // SCALFMM_META_FORWARD_HPP
