// --------------------------------
// See LICENCE file at project root
// File : scalfmm/container/particle.hpp
// --------------------------------
#ifndef SCALFMM_CONTAINER_PARTICLE_HPP
#define SCALFMM_CONTAINER_PARTICLE_HPP

#include "scalfmm/container/particle_impl.hpp"
#include "scalfmm/container/particle_proxy.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"

#include <cstddef>
#include <ostream>
#include <type_traits>

/**
 * @brief Multi-purpose particle implementation
 *
 * This template implementation of a particle allows simple reuse for several
 * use cases. The aim it to provide an interface that is compatible with the
 * rest of ScalFMM. It is mainly intended to be used as an interface for the
 * particle containers.
 *
 * The Types parameter pack can accept any type that is to be considered as a
 * particle attribute. You can also specify scalfmm::pack type to factorize
 * several types.
 *
 * In the following example, the two specializations of the class will give the
 * same final structure.
 *
 * ```
 * using FReal = double;
 * static constexpr std::size_t dimension = 3;
 *
 * particle<FReal, dimension, int, float, float, float, float>;
 * particle<FReal, dimension, int, scalfmm::meta::pack<4, float> >;
 * ```
 *
 * The base of these two classes is
 * ```
 * std::tuple<double, double, double, int, float, float, float, float>;
 * ```
 *
 * @warning Although the classes will have the same final layout, C++ considers
 * these two classes to be different !
 *
 * ##### Example
 *
 * ```
 * // Define a 3D particle with an int attribute
 * using Particle = particle<double, 3, int>;
 *
 * Particle p;
 * p.get<>
 * ```
 *
 *
 * @tparam FReal Floating point type
 * @tparam dimension Space dimension count
 * @tparam Types Attributes type list
 */
namespace scalfmm::container
{
    /**
     * @brief Particle traits to extract info about the particle.
     *
     * @tparam Particle
     */
    template<typename Particle>
    struct particle_traits
    {
        using position_value_type = typename Particle::position_value_type;
        static constexpr std::size_t dimension_size = Particle::dimension_size;
        using position_type = typename Particle::position_type;

        using inputs_value_type = typename Particle::inputs_value_type;
        static constexpr std::size_t inputs_size = Particle::inputs_size;
        using inputs_type = typename Particle::inputs_type;

        using outputs_value_type = typename Particle::outputs_value_type;
        static constexpr std::size_t outputs_size = Particle::outputs_size;
        using outputs_type = typename Particle::outputs_type;

        using variables_type = typename Particle::variables_type;
        static constexpr std::size_t variables_size = Particle::variables_size;

        using range_position_type = meta::make_range_sequence<0, dimension_size>;
        using range_inputs_type = meta::make_range_sequence<dimension_size, dimension_size + inputs_size>;
        using range_outputs_type =
          meta::make_range_sequence<dimension_size + inputs_size, dimension_size + inputs_size + outputs_size>;
        using range_variables_type =
          meta::make_range_sequence<dimension_size + inputs_size + outputs_size,
                                    dimension_size + inputs_size + outputs_size + variables_size>;
        static constexpr std::size_t number_of_elements = dimension_size + inputs_size + outputs_size + variables_size;
    };

    /**
     * @brief
     *
     * This is the particle container. It holds a nd position, N number of inputs and M number of outputs.
     * If you pass references as PositionType, InputsType and OutputsType, the particle becomes a
     * proxy meaning you can wrap a tuple and modify the the tuple through the proxy.
     * This is useful the directly modify the variadic container particle_container.
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
    struct particle
      : std::conditional_t<
          meta::all(std::is_reference_v<PositionType>, std::is_reference_v<InputsType>,
                    std::is_reference_v<OutputsType>, meta::all(std::is_reference_v<Variables>...)),
          particle_proxy<std::remove_reference_t<PositionType>, PositionDim, std::remove_reference_t<InputsType>,
                         NInputs, std::remove_reference_t<OutputsType>, MOutputs,
                         std::remove_reference_t<Variables>...>,
          particle_impl<PositionType, PositionDim, InputsType, NInputs, OutputsType, MOutputs, Variables...>>
    {
      public:
        // If all value types are references we instanciate a proxy. We remove the reference to pass the type to
        // the proxy because it will be passed to a std::reference_wrapper.
        // In the other case i.e no references are passed, we create a simple particle holding all the values.
        static constexpr bool is_referenced =
          meta::all(std::is_reference_v<PositionType>, std::is_reference_v<InputsType>,
                    std::is_reference_v<OutputsType>, meta::all(std::is_reference_v<Variables>...));
        using base_type = std::conditional_t<
          is_referenced,
          particle_proxy<std::remove_reference_t<PositionType>, PositionDim, std::remove_reference_t<InputsType>,
                         NInputs, std::remove_reference_t<OutputsType>, MOutputs,
                         std::remove_reference_t<Variables>...>,
          particle_impl<PositionType, PositionDim, InputsType, NInputs, OutputsType, MOutputs, Variables...>>;

        using proxy_type = particle<std::remove_reference_t<std::remove_const_t<PositionType>>&, PositionDim,
                                    std::remove_reference_t<std::remove_const_t<InputsType>>&, NInputs,
                                    std::remove_reference_t<std::remove_const_t<OutputsType>>&, MOutputs,
                                    std::remove_reference_t<std::remove_const_t<Variables>>&...>;
        using const_proxy_type = particle<std::add_const_t<std::remove_reference_t<PositionType>>&, PositionDim,
                                          std::add_const_t<std::remove_reference_t<InputsType>>&, NInputs,
                                          std::add_const_t<std::remove_reference_t<OutputsType>>&, MOutputs,
                                          std::add_const_t<std::remove_reference_t<Variables>>&...>;

        using base_type::base_type;
        constexpr particle() = default;
        constexpr particle(particle const&) = default;
        constexpr particle(particle&&) noexcept = default;
        constexpr inline auto operator=(particle const&) -> particle& = default;
        constexpr inline auto operator=(particle&&) noexcept -> particle& = default;
        ~particle() = default;

        // Common formatted output operator
        inline friend auto operator<<(std::ostream& os, const particle& part) -> std::ostream&
        {
            os << part.position();
            if constexpr(base_type::inputs_size > 0)
            {
                io::print(os, part.inputs());
            }
            if constexpr(base_type::outputs_size > 0)
            {
                io::print(os, part.outputs());
            }
            if constexpr(base_type::variables_size > 0)
            {
                io::print(os, part.variables());
            }
            return os;
        }

        // Common size functions.

        /**
         * @brief Size of dimensions.
         *
         * @return std::size_t
         */
        [[nodiscard]] constexpr inline auto sizeof_dimension() const noexcept -> std::size_t { return PositionDim; }

        /**
         * @brief  Number of inputs.
         *
         * @return std::size_t
         */
        [[nodiscard]] constexpr inline auto sizeof_inputs() const noexcept -> std::size_t { return NInputs; }

        /**
         * @brief Number of outputs.
         *
         * @return std::size_t
         */
        [[nodiscard]] constexpr inline auto sizeof_outputs() const noexcept -> std::size_t { return MOutputs; }

        /**
         * @brief  Number of variables.
         *
         * @return std::size_t
         */
        [[nodiscard]] constexpr inline auto sizeof_variables() const noexcept -> std::size_t
        {
            return sizeof...(Variables);
        }
    };

    /**
     * @brief Returns the underlying reference to the position array (const version).
     *
     * @tparam Particle
     * @param part
     * @return Particle::position_type const&
     */
    template<typename Particle>
    constexpr inline auto position(Particle const& part) -> typename Particle::position_type const&
    {
        return part.position();
    }

    /**
     * @brief Returns the ith component of the positions (const version).
     *
     * @tparam Particle
     * @param part
     * @param i
     * @return Particle::position_value_type const&
     */
    template<typename Particle>
    constexpr inline auto position(Particle const& part, std::size_t i) -> typename Particle::position_value_type const&
    {
        return part.position(i);
    }

    /**
     * @brief  Returns the underlying reference to the position array.
     *
     * @tparam Particle
     * @param part
     * @return Particle::position_type&
     */
    template<typename Particle>
    constexpr inline auto position(Particle& part) -> typename Particle::position_type&
    {
        return part.position();
    }

    /**
     * @brief Returns the ith component of the positions.
     *
     * @tparam Particle
     * @param part
     * @param i
     * @return Particle::position_value_type&
     */
    template<typename Particle>
    constexpr inline auto position(Particle const& part, std::size_t i) -> typename Particle::position_value_type&
    {
        return part.position(i);
    }

    /**
     * @brief Returns the underlying reference to the inputs array (const version).
     *
     * @tparam Particle
     * @param part
     * @return Particle::inputs_type const&
     */
    template<typename Particle>
    constexpr inline auto inputs(Particle const& part) -> typename Particle::inputs_type const&
    {
        return part.inputs();
    }

    /**
     * @brief Returns the ith component of the inputs (const version).
     *
     * @tparam Particle
     * @param part
     * @param i
     * @return Particle::inputs_value_type const&
     */
    template<typename Particle>
    constexpr inline auto inputs(Particle const& part, std::size_t i) -> typename Particle::inputs_value_type const&
    {
        return part.inputs(i);
    }

    /**
     * @brief Returns the underlying reference to the inputs array.
     *
     * @tparam Particle
     * @param part
     * @return Particle::inputs_type&
     */
    template<typename Particle>
    constexpr inline auto inputs(Particle& part) -> typename Particle::inputs_type&
    {
        return part.inputs();
    }

    /**
     * @brief Returns the ith component of the inputs.
     *
     * @tparam Particle
     * @param part
     * @param i
     * @return Particle::inputs_value_type&
     */
    template<typename Particle>
    constexpr inline auto inputs(Particle const& part, std::size_t i) -> typename Particle::inputs_value_type&
    {
        return part.inputs(i);
    }

    /**
     * @brief Returns the underlying reference to the outputs array (const version).
     *
     * @tparam Particle
     * @param part
     * @return Particle::outputs_type const&
     */
    template<typename Particle>
    constexpr inline auto outputs(Particle const& part) -> typename Particle::outputs_type const&
    {
        return part.outputs();
    }

    /**
     * @brief Returns the ith component of the outputs (const version).
     *
     * @tparam Particle
     * @param part
     * @param i
     * @return Particle::outputs_value_type const&
     */
    template<typename Particle>
    constexpr inline auto outputs(Particle const& part, std::size_t i) -> typename Particle::outputs_value_type const&
    {
        return part.outputs(i);
    }

    /**
     * @brief Returns the underlying reference to the outputs array.
     *
     * @tparam Particle
     * @param part
     * @return Particle::outputs_type&
     */
    template<typename Particle>
    constexpr inline auto outputs(Particle& part) -> typename Particle::outputs_type&
    {
        return part.outputs();
    }

    /**
     * @brief Returns the ith component of the outputs.
     *
     * @tparam Particle
     * @param part
     * @param i
     * @return Particle::outputs_value_type&
     */
    template<typename Particle>
    constexpr inline auto outputs(Particle const& part, std::size_t i) -> typename Particle::outputs_value_type&
    {
        return part.outputs(i);
    }

    /**
     * @brief Returns the underlying tuple of variables in a tuple (const version).
     *
     * @tparam Particle
     * @param part
     * @return Particle::variables_type const&
     */
    template<typename Particle>
    constexpr inline auto variables(Particle const& part) -> typename Particle::variables_type const&
    {
        return part.variables();
    }

    /**
     * @brief Returns the underlying tuple of variables in a tuple.
     *
     * @tparam Particle
     * @param part
     * @return Particle::variables_type const&
     */
    template<typename Particle>
    constexpr inline auto variables(Particle& part) -> typename Particle::variables_type&
    {
        return part.variables();
    }

}   // namespace scalfmm::container

#endif   // SCALFMM_CONTAINER_PARTICLE_HPP
