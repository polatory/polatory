// --------------------------------
// See LICENCE file at project root
// File : scalfmm/container/particle_impl.hpp
// --------------------------------
#ifndef SCALFMM_CONTAINER_PARTICLE_IMPL_HPP
#define SCALFMM_CONTAINER_PARTICLE_IMPL_HPP

#include "scalfmm/container/point.hpp"
#include "scalfmm/meta/type_pack.hpp"
#include "scalfmm/meta/utils.hpp"
#include <inria/integer_sequence.hpp>

#include <array>
#include <cstddef>
#include <iterator>
#include <tuple>
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
 * In the following example, the two specialisations of the class will give the
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
 * \warning Although the classes will have the same final layout, C++ considers
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
     * @brief
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
    struct particle_impl
    {
      public:
        using position_value_type = PositionType;
        static constexpr std::size_t dimension = PositionDim;
        static constexpr std::size_t dimension_size = PositionDim;
        using position_type = container::point<position_value_type, dimension_size>;
        using position_tuple_type = meta::generate_tuple_t<position_value_type, dimension_size>;
        using range_position_type = meta::make_range_sequence<0, dimension_size>;

        using inputs_value_type = InputsType;
        static constexpr std::size_t inputs_size = NInputs;
        using inputs_type = std::array<inputs_value_type, inputs_size>;
        using inputs_tuple_type = meta::generate_tuple_t<inputs_value_type, inputs_size>;
        using range_inputs_type = meta::make_range_sequence<dimension_size, dimension_size + inputs_size>;

        using outputs_value_type = OutputsType;
        static constexpr std::size_t outputs_size = MOutputs;
        using outputs_type = std::array<outputs_value_type, outputs_size>;
        using outputs_tuple_type = meta::generate_tuple_t<outputs_value_type, outputs_size>;
        using range_outputs_type =
          meta::make_range_sequence<dimension_size + inputs_size, dimension_size + inputs_size + outputs_size>;

        using variables_type = std::tuple<Variables...>;
        static constexpr std::size_t variables_size = sizeof...(Variables);
        using range_variables_type =
          meta::make_range_sequence<dimension_size + inputs_size + outputs_size,
                                    dimension_size + inputs_size + outputs_size + variables_size>;

        using tuple_type =
          typename meta::cat<typename meta::pack_expand_tuple<meta::pack<dimension_size, position_value_type>,
                                                              meta::pack<inputs_size, inputs_value_type>,
                                                              meta::pack<outputs_size, outputs_value_type>>,
                             variables_type>::type;

        /**
         * @brief
         *
         */
        constexpr particle_impl() = default;

        /**
         * @brief
         *
         */
        constexpr particle_impl(particle_impl const&) = default;

        /**
         * @brief
         *
         */
        constexpr particle_impl(particle_impl&&) noexcept = default;

        /**
         * @brief
         *
         * @return particle_impl&
         */
        constexpr inline auto operator=(particle_impl const&) -> particle_impl& = default;

        /**
         * @brief
         *
         * @return particle_impl&
         */
        constexpr inline auto operator=(particle_impl&&) noexcept -> particle_impl& = default;

        /**
         * @brief Destroy the particle impl object
         *
         */
        ~particle_impl() = default;

        /**
         * @brief
         *
         * @param p position of the particle
         * @param i input associated to the particle
         * @param o output  associated to the particle
         * @param vs variables
         *
         * @return
         */
        constexpr particle_impl(position_type p, inputs_value_type i, outputs_value_type o, Variables... vs)
          : m_position(p)
          , m_variables(vs...)
        {
            std::fill(std::begin(m_inputs), std::end(m_inputs), i);
            std::fill(std::begin(m_outputs), std::end(m_outputs), o);
        }

        /**
         * @brief
         *
         * @param p
         * @param i
         * @param o
         * @param vs
         *
         * @return
         *
         */
        constexpr particle_impl(position_type p, inputs_type i, outputs_type o, Variables... vs)
          : m_position(p)
          , m_inputs(i)
          , m_outputs(o)
          , m_variables(vs...)
        {
        }

        /**
         * @brief Construct a new particle impl object
         *
         * @param t
         */
        particle_impl(tuple_type t)
          : m_position(meta::to_array(meta::sub_tuple(t, range_position_type{})))
          , m_inputs{meta::to_array(meta::sub_tuple(t, range_inputs_type{}))}
          , m_outputs{meta::to_array(meta::sub_tuple(t, range_outputs_type{}))}
          , m_variables(meta::sub_tuple(t, range_variables_type{}))
        {
        }

        /**
         * @brief
         *
         * @return tuple_type
         */
        [[nodiscard]] constexpr inline auto as_tuple() const noexcept -> tuple_type
        {
            return std::tuple_cat(meta::to_tuple(m_position), meta::to_tuple(m_inputs), meta::to_tuple(m_outputs),
                                  m_variables);
        }

        /**
         * @brief
         *
         * @param p
         */
        constexpr inline auto position(position_type p) noexcept -> void { m_position = p; }

        /**
         * @brief
         *
         * @return position_type const&
         */
        [[nodiscard]] constexpr inline auto position() const noexcept -> position_type const& { return m_position; }

        /**
         * @brief
         *
         * @return position_type&
         */
        [[nodiscard]] constexpr inline auto position() noexcept -> position_type& { return m_position; }

        /**
         * @brief
         *
         * @param i
         * @return position_value_type
         */
        [[nodiscard]] constexpr inline auto position(std::size_t i) const noexcept -> position_value_type
        {
            return m_position.at(i);
        }

        /**
         * @brief
         *
         * @param i
         * @return position_value_type&
         */
        [[nodiscard]] constexpr inline auto position(std::size_t i) noexcept -> position_value_type&
        {
            return m_position.at(i);
        }

        /**
         * @brief
         *
         * @param i
         */
        constexpr inline auto inputs(inputs_type i) noexcept -> void { m_inputs = i; }

        /**
         * @brief
         *
         * @return inputs_type const&
         */
        [[nodiscard]] constexpr inline auto inputs() const noexcept -> inputs_type const& { return m_inputs; }

        /**
         * @brief
         *
         * @return inputs_type&
         */
        [[nodiscard]] constexpr inline auto inputs() noexcept -> inputs_type& { return m_inputs; }

        /**
         * @brief
         *
         * @param i
         * @return inputs_value_type
         */
        [[nodiscard]] constexpr inline auto inputs(std::size_t i) const noexcept -> inputs_value_type
        {
            return m_inputs.at(i);
        }

        /**
         * @brief
         *
         * @param i
         * @return inputs_value_type&
         */
        [[nodiscard]] constexpr inline auto inputs(std::size_t i) noexcept -> inputs_value_type&
        {
            return m_inputs.at(i);
        }

        /**
         * @brief
         *
         * @param i
         */
        constexpr inline auto outputs(outputs_type i) noexcept -> void { m_outputs = i; }

        /**
         * @brief
         *
         * @return outputs_type const&
         */
        [[nodiscard]] constexpr inline auto outputs() const noexcept -> outputs_type const& { return m_outputs; }

        /**
         * @brief
         *
         * @return outputs_type&
         */
        [[nodiscard]] constexpr inline auto outputs() noexcept -> outputs_type& { return m_outputs; }

        /**
         * @brief
         *
         * @param i
         * @return outputs_value_type
         */
        [[nodiscard]] constexpr inline auto outputs(std::size_t i) const noexcept -> outputs_value_type
        {
            return m_outputs.at(i);
        }

        /**
         * @brief
         *
         * @param i
         * @return outputs_value_type&
         */
        [[nodiscard]] constexpr inline auto outputs(std::size_t i) noexcept -> outputs_value_type&
        {
            return m_outputs.at(i);
        }

        /**
         * @brief
         *
         * @return variables_type const&
         */
        [[nodiscard]] constexpr inline auto variables() const noexcept -> variables_type const& { return m_variables; }

        /**
         * @brief
         *
         * @return variables_type&
         */
        [[nodiscard]] constexpr inline auto variables() noexcept -> variables_type& { return m_variables; }

        /**
         * @brief
         *
         * @tparam T
         * @param v
         * @return std::enable_if_t<std::is_same_v<T, variables_type>, void>
         */
        template<typename T>
        constexpr inline auto variables(T v) noexcept -> std::enable_if_t<std::is_same_v<T, variables_type>, void>
        {
            m_variables = v;
        }

        /**
         * @brief
         *
         * @tparam Vs
         * @param vs
         * @return std::enable_if_t<(sizeof...(Vs) != 0), void>
         */
        template<typename... Vs>
        constexpr inline auto variables(Vs... vs) noexcept -> std::enable_if_t<(sizeof...(Vs) != 0), void>
        {
            m_variables = std::make_tuple(vs...);
        }

      private:
        position_type m_position{};
        inputs_type m_inputs{};
        outputs_type m_outputs{};
        variables_type m_variables{};
    };

}   // namespace scalfmm::container

#endif   // SCALFMM_CONTAINER_PARTICLE_HPP
