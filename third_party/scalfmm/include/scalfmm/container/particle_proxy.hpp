// --------------------------------
// See LICENCE file at project root
// File : scalfmm/container/particle_proxy.hpp
// --------------------------------
#ifndef SCALFMM_CONTAINER_PARTICLE_PROXY_HPP
#define SCALFMM_CONTAINER_PARTICLE_PROXY_HPP

#include "scalfmm/container/point.hpp"
#include "scalfmm/container/reference_sequence.hpp"
#include "scalfmm/meta/type_pack.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>

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
     * @tparam NOutputs
     * @tparam Variables
     */
    template<typename PositionType, std::size_t PositionDim, typename InputsType, std::size_t NInputs,
             typename OutputsType, std::size_t NOutputs, typename... Variables>
    struct particle_proxy
    {
      public:
        using position_value_type = PositionType;
        static constexpr std::size_t dimension = PositionDim;
        static constexpr std::size_t dimension_size = PositionDim;
        // here we add a reference to instantiate a proxy of the point
        using position_type = container::point<std::add_lvalue_reference_t<position_value_type>, dimension_size>;
        using const_position_type =
          container::point<std::add_const_t<std::add_lvalue_reference_t<position_value_type>>, dimension_size>;
        using range_position_type = meta::make_range_sequence<0, dimension_size>;

        using inputs_value_type = InputsType;
        static constexpr std::size_t inputs_size = NInputs;
        using inputs_type = std::array<std::reference_wrapper<inputs_value_type>, inputs_size>;
        using const_inputs_type = std::array<std::reference_wrapper<std::add_const_t<inputs_value_type>>, inputs_size>;
        using range_inputs_type = meta::make_range_sequence<dimension_size, dimension_size + inputs_size>;

        using outputs_value_type = OutputsType;
        static constexpr std::size_t outputs_size = NOutputs;
        using outputs_type = std::array<std::reference_wrapper<outputs_value_type>, outputs_size>;
        using const_outputs_type =
          std::array<std::reference_wrapper<std::add_const_t<outputs_value_type>>, outputs_size>;
        using range_outputs_type =
          meta::make_range_sequence<dimension_size + inputs_size, dimension_size + inputs_size + outputs_size>;

        using variables_type = std::tuple<std::add_lvalue_reference_t<Variables>...>;
        using const_variables_type = std::tuple<std::add_lvalue_reference_t<std::add_const_t<Variables>>...>;
        static constexpr std::size_t variables_size = sizeof...(Variables);
        using range_variables_type =
          meta::make_range_sequence<dimension_size + inputs_size + outputs_size,
                                    dimension_size + inputs_size + outputs_size + variables_size>;

        static constexpr bool is_const_proxy =
          meta::all(std::is_const_v<PositionType>, std::is_const_v<InputsType>, std::is_const_v<OutputsType>,
                    meta::all(std::is_const_v<Variables>...));

        // tuple type from which we can construct the proxy.
        using tuple_type = typename meta::cat<
          typename meta::pack_expand_tuple<meta::pack<dimension_size, std::add_lvalue_reference_t<position_value_type>>,
                                           meta::pack<inputs_size, std::add_lvalue_reference_t<inputs_value_type>>,
                                           meta::pack<outputs_size, std::add_lvalue_reference_t<outputs_value_type>>>,
          variables_type>::type;

        /**
         * @brief
         *
         */
        constexpr particle_proxy() = delete;

        /**
         * @brief
         *
         */
        constexpr particle_proxy(particle_proxy const&) = default;

        /**
         * @brief
         *
         */
        constexpr particle_proxy(particle_proxy&&) noexcept = default;

        /**
         * @brief
         *
         * @return particle_proxy&
         */
        constexpr inline auto operator=(particle_proxy const&) -> particle_proxy& = default;

        /**
         * @brief
         *
         * @return particle_proxy&
         */
        constexpr inline auto operator=(particle_proxy&&) noexcept -> particle_proxy& = default;

        /**
         * @brief Destroy the particle proxy object
         *
         */
        ~particle_proxy() = default;

        /**
         * @brief Construct a new particle proxy object
         *
         * @param t
         */
        particle_proxy(tuple_type t)
          : m_position(container::get_reference_sequence(meta::sub_tuple(t, range_position_type{})))
          , m_inputs(container::get_reference_sequence(meta::sub_tuple(t, range_inputs_type{})))
          , m_outputs(container::get_reference_sequence(meta::sub_tuple(t, range_outputs_type{})))
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
         * @return const_position_type const&
         */
        [[nodiscard]] constexpr inline auto position() const noexcept -> const_position_type const&
        {
            return m_position;
        }

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
         * @return position_value_type const&
         */
        [[nodiscard]] constexpr inline auto position(std::size_t i) const noexcept -> position_value_type const&
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
         * @return const_inputs_type const&
         */
        [[nodiscard]] constexpr inline auto inputs() const noexcept -> const_inputs_type const& { return m_inputs; }

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
         * @return inputs_value_type const&
         */
        [[nodiscard]] constexpr inline auto inputs(std::size_t i) const noexcept -> inputs_value_type const&
        {
            return m_inputs.at(i).get();
        }

        /**
         * @brief
         *
         * @param i
         * @return inputs_value_type&
         */
        [[nodiscard]] constexpr inline auto inputs(std::size_t i) noexcept -> inputs_value_type&
        {
            return m_inputs.at(i).get();
        }

        /**
         * @brief
         *
         * @return const_outputs_type const&
         */
        [[nodiscard]] constexpr inline auto outputs() const noexcept -> const_outputs_type const& { return m_outputs; }

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
         * @return outputs_value_type const&
         */
        [[nodiscard]] constexpr inline auto outputs(std::size_t i) const noexcept -> outputs_value_type const&
        {
            return m_outputs.at(i).get();
        }

        /**
         * @brief
         *
         * @param i
         * @return outputs_value_type&
         */
        [[nodiscard]] constexpr inline auto outputs(std::size_t i) noexcept -> outputs_value_type&
        {
            return m_outputs.at(i).get();
        }

        /**
         * @brief
         *
         * @return const_variables_type
         */
        [[nodiscard]] constexpr inline auto variables() const noexcept -> const_variables_type { return m_variables; }

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
        std::conditional_t<is_const_proxy, const_position_type, position_type> m_position;
        std::conditional_t<is_const_proxy, const_inputs_type, inputs_type> m_inputs;
        std::conditional_t<is_const_proxy, const_outputs_type, outputs_type> m_outputs;
        std::conditional_t<is_const_proxy, const_variables_type, variables_type> m_variables;
    };
}   // namespace scalfmm::container

#endif   // SCALFMM_CONTAINER_particle_proxy_PROXY_HPP
