// --------------------------------
// See LICENCE file at project root
// File : scalfmm/container/particle_container.hpp
// --------------------------------
#ifndef SCALFMM_CONTAINER_PARTICLE_CONTAINER_HPP
#define SCALFMM_CONTAINER_PARTICLE_CONTAINER_HPP

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/variadic_adaptor.hpp"
#include "scalfmm/meta/type_pack.hpp"
#include "scalfmm/meta/utils.hpp"

#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace scalfmm::container
{
    /**
     * @brief This class stores the particles in the leaf of the tree.
     *
     * The particle container allows you to store particle in a generic
     * manner in a SOA container.
     * The container contains 4 blocks
     *  - the positions
     *  - input data
     *  - the output data
     *  - the variables
     * The data (positions, inputs and outputs) are of the same type, the variables can have different types (float,
     *  int, ...).
     *
     * The container is seen as a tuple structured in blocks.
     *
     * @tparam Particle
     *
     * @image  html particles_container.svg "Particle container class"
     * @image  latex particles_container.pdf "Particle container class" width=0.5*\textwidth
     */
    template<typename Particle>
    class particle_container
      : public variadic_container_tuple<particle_container<Particle>, typename Particle::tuple_type>
    {
      public:
        using particle_type = Particle;

        using position_value_type = typename particle_type::position_value_type;
        using inputs_value_type = typename particle_type::inputs_value_type;
        using outputs_value_type = typename particle_type::outputs_value_type;
        using variables_type = typename particle_type::variables_type;

        static constexpr std::size_t dimension = particle_type::dimension_size;
        static constexpr std::size_t inputs_size = particle_type::inputs_size;
        static constexpr std::size_t outputs_size = particle_type::outputs_size;

        // base type : concatenating the particle tuple with the indexes needed
        // in order to allocate the vector
        using tuple_type =
          typename meta::cat<typename meta::pack_expand_tuple<meta::pack<dimension, position_value_type>,
                                                              meta::pack<inputs_size, inputs_value_type>,
                                                              meta::pack<outputs_size, outputs_value_type>>,
                             variables_type>::type;
        using self_type = particle_container<particle_type>;
        using base_type = variadic_container_tuple<self_type, tuple_type>;
        using value_type = particle_type;
        using proxy_type = typename particle_type::proxy_type;
        using const_proxy_type = typename particle_type::const_proxy_type;

        // Forwarding constructors
        using base_type::base_type;

        /**
         * @brief
         *
         * @param i
         * @return particle_type
         */
        [[nodiscard]] inline auto particle(std::size_t i) const -> particle_type
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return particle_type(*it);
        }

        /**
         * @brief
         *
         * @param i
         * @return proxy_type
         */
        [[nodiscard]] inline auto at(std::size_t i) -> proxy_type
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return proxy_type(*it);
        }

        /**
         * @brief
         *
         * @param i
         * @return const_proxy_type
         */
        [[nodiscard]] inline auto at(std::size_t i) const -> const_proxy_type
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return const_proxy_type(*it);
        }

        /**
         * @brief
         *
         * @param i
         * @return proxy_type
         */
        [[nodiscard]] inline auto operator[](std::size_t i) noexcept -> proxy_type
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return proxy_type(*it);
        }

        /**
         * @brief
         *
         * @param i
         * @return const_proxy_type
         */
        [[nodiscard]] inline auto operator[](std::size_t i) const noexcept -> const_proxy_type
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return const_proxy_type(*it);
        }

        /**
         * @brief
         *
         * @return the number of particles inside the container
         */
        [[nodiscard]] inline auto size() const -> std::size_t { return std::get<0>(base_type::all_size()); }

        /**
         * @brief
         *
         * @param i
         * @param p
         */
        inline auto insert_particle(std::size_t i, particle_type p) -> void
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            *it = p.as_tuple();
        }

        /**
         * @brief
         *
         * @param i
         * @return particle_type::position_type
         */
        [[nodiscard]] inline auto position(std::size_t i) const -> typename particle_type::position_type
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return meta::to_array(meta::sub_tuple(*it, typename particle_type::range_position_type{}));
        }

        /**
         * @brief
         *
         * @param i
         * @return auto
         */
        [[nodiscard]] inline auto position_as_tuple(std::size_t i) const
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return meta::sub_tuple(*it, typename particle_type::range_position_type{});
        }

        /**
         * @brief
         *
         * @param i
         * @param p
         */
        inline auto insert_position(std::size_t i, typename particle_type::position_type p) -> void
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            meta::sub_tuple(*it, typename particle_type::range_position_type{}) = meta::to_tuple(p);
        }

        /**
         * @brief
         *
         * @param i
         * @param p
         */
        inline auto insert_position(std::size_t i, typename particle_type::position_tuple_type p) -> void
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            meta::sub_tuple(*it, typename particle_type::range_position_type{}) = meta::to_tuple(p);
        }

        /**
         * @brief
         *
         * @param i
         * @return particle_type::inputs_type
         */
        [[nodiscard]] inline auto inputs(std::size_t i) const -> typename particle_type::inputs_type
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return meta::to_array(meta::sub_tuple(*it, typename particle_type::range_inputs_type{}));
        }

        /**
         * @brief
         *
         * @param i
         * @return auto
         */
        [[nodiscard]] inline auto inputs_as_tuple(std::size_t i) const
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return meta::sub_tuple(*it, typename particle_type::range_inputs_type{});
        }

        /**
         * @brief
         *
         * @param i
         * @param in_input
         */
        inline auto insert_inputs(std::size_t i, typename particle_type::inputs_type in_input) -> void
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            meta::sub_tuple(*it, typename particle_type::range_inputs_type{}) = meta::to_tuple(in_input);
        }

        /**
         * @brief
         *
         * @param i
         * @param p
         */
        inline auto insert_inputs(std::size_t i, typename particle_type::inputs_tuple_type p) -> void
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            meta::sub_tuple(*it, typename particle_type::range_inputs_type{}) = p;
        }

        /**
         * @brief
         *
         * @param i
         * @return particle_type::outputs_type
         */
        [[nodiscard]] inline auto outputs(std::size_t i) const -> typename particle_type::outputs_type
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return meta::to_array(meta::sub_tuple(*it, typename particle_type::range_outputs_type{}));
        }

        /**
         * @brief
         *
         * @param i
         * @return auto
         */
        [[nodiscard]] inline auto outputs_as_tuple(std::size_t i) const
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return meta::sub_tuple(*it, typename particle_type::range_outputs_type{});
        }

        /**
         * @brief
         *
         * @param i
         * @param p
         */
        inline auto insert_outputs(std::size_t i, typename particle_type::outputs_type p) -> void
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            meta::sub_tuple(*it, typename particle_type::range_outputs_type{}) = meta::to_tuple(p);
        }

        /**
         * @brief
         *
         * @param i
         * @param p
         */
        inline auto insert_outputs(std::size_t i, typename particle_type::outputs_tuple_type p) -> void
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            meta::sub_tuple(*it, typename particle_type::range_outputs_type{}) = p;
        }

        /**
         * @brief
         *
         * @param i
         * @return particle_type::variables_type
         */
        [[nodiscard]] inline auto variables(std::size_t i) const -> typename particle_type::variables_type
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            return meta::sub_tuple(*it, typename particle_type::range_variables_type{});
        }

        /**
         * @brief
         *
         * @param i
         * @param p
         */
        inline auto insert_variables(std::size_t i, typename particle_type::variables_type p) -> void
        {
            auto it{std::begin(*this)};
            std::advance(it, i);
            meta::sub_tuple(*it, typename particle_type::range_variables_type{}) = p;
        }

        /**
         * @brief
         *
         */
        inline auto reset_positions() noexcept -> void
        {
            auto it_begin{std::begin(*this)};
            auto it_end{std::end(*this)};

            while(it_begin != it_end)
            {
                auto proxy = proxy_type(*it_begin);
                for(std::size_t ii{0}; ii < dimension; ++ii)
                {
                    proxy.position(ii) = position_value_type(0.0);
                }
                ++it_begin;
            }
        }

        /**
         * @brief
         *
         */
        inline auto reset_inputs() noexcept -> void
        {
            auto it_begin{std::begin(*this)};
            auto it_end{std::end(*this)};

            while(it_begin != it_end)
            {
                auto proxy = proxy_type(*it_begin);
                for(std::size_t ii{0}; ii < inputs_size; ++ii)
                {
                    proxy.inputs(ii) = inputs_value_type(0.0);
                }
                ++it_begin;
            }
        }

        /**
         * @brief
         *
         */
        inline auto reset_outputs() noexcept -> void
        {
            auto it_begin{std::begin(*this)};
            auto it_end{std::end(*this)};

            while(it_begin != it_end)
            {
                auto proxy = proxy_type(*it_begin);
                for(std::size_t ii{0}; ii < outputs_size; ++ii)
                {
                    proxy.outputs(ii) = outputs_value_type(0.0);
                }
                ++it_begin;
            }
        }

        /**
         * @brief
         *
         */
        inline auto reset_variables() noexcept -> void
        {
            auto it_begin{std::begin(*this)};
            auto it_end{std::end(*this)};

            while(it_begin != it_end)
            {
                auto proxy = proxy_type(*it_begin);
                proxy.variables() = variables_type{};
                ++it_begin;
            }
        }

        /**
         * @brief
         *
         */
        inline auto reset_particles() noexcept -> void
        {
            auto it_begin{std::begin(*this)};
            auto it_end{std::end(*this)};

            while(it_begin != it_end)
            {
                auto proxy = proxy_type(*it_begin);
                for(std::size_t ii{0}; ii < dimension; ++ii)
                {
                    proxy.position(ii) = position_value_type(0.0);
                }
                for(std::size_t ii{0}; ii < inputs_size; ++ii)
                {
                    proxy.inputs(ii) = inputs_value_type(0.0);
                }
                for(std::size_t ii{0}; ii < outputs_size; ++ii)
                {
                    proxy.outputs(ii) = outputs_value_type(0.0);
                }
                proxy.variables() = variables_type{};
                ++it_begin;
            }
        }

        /**
         * @brief
         *
         * @param os
         * @param container
         * @return std::ostream&
         */
        inline friend auto operator<<(std::ostream& os, const particle_container& container) -> std::ostream&
        {
            for(std::size_t i{0}; i < container.size(); ++i)
            {
                auto const& p = container.at(i);
                std::cout << i << " " << p << std::endl;
            }
            return os;
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto position_begin() noexcept
        {
            using position_range = typename container::particle_traits<particle_type>::range_position_type;
            return this->template sbegin<position_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto position_begin() const noexcept
        {
            using position_range = typename container::particle_traits<particle_type>::range_position_type;
            return this->template sbegin<position_range>();
        }

        /// @brief
        ///
        /// @return
        [[nodiscard]] constexpr inline auto position_end() noexcept
        {
            using position_range = typename container::particle_traits<particle_type>::range_position_type;
            return this->template send<position_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto position_end() const noexcept
        {
            using position_range = typename container::particle_traits<particle_type>::range_position_type;
            return this->template send<position_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto inputs_begin() noexcept
        {
            using inputs_range = typename container::particle_traits<particle_type>::range_inputs_type;
            return this->template sbegin<inputs_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto inputs_begin() const noexcept
        {
            using inputs_range = typename container::particle_traits<particle_type>::range_inputs_type;
            return this->template sbegin<inputs_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto inputs_end() noexcept
        {
            using inputs_range = typename container::particle_traits<particle_type>::range_inputs_type;
            return this->template send<inputs_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto inputs_end() const noexcept
        {
            using inputs_range = typename container::particle_traits<particle_type>::range_inputs_type;
            return this->template send<inputs_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto outputs_begin() noexcept
        {
            using outputs_range = typename container::particle_traits<particle_type>::range_outputs_type;
            return this->template sbegin<outputs_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto outputs_begin() const noexcept
        {
            using outputs_range = typename container::particle_traits<particle_type>::range_outputs_type;
            return this->template sbegin<outputs_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto outputs_end() noexcept
        {
            using outputs_range = typename container::particle_traits<particle_type>::range_outputs_type;
            return this->template send<outputs_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto outputs_end() const noexcept
        {
            using outputs_range = typename container::particle_traits<particle_type>::range_outputs_type;
            return this->template send<outputs_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto variables_begin() noexcept
        {
            using variables_range = typename container::particle_traits<particle_type>::range_variables_type;
            return this->template sbegin<variables_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto variables_begin() const noexcept
        {
            using variables_range = typename container::particle_traits<particle_type>::range_variables_type;
            return this->template sbegin<variables_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto variables_end() noexcept
        {
            using variables_range = typename container::particle_traits<particle_type>::range_variables_type;
            return this->template send<variables_range>();
        }

        /**
         * @brief
         *
         * @return constexpr auto
         */
        [[nodiscard]] constexpr inline auto variables_end() const noexcept
        {
            using variables_range = typename container::particle_traits<particle_type>::range_variables_type;
            return this->template send<variables_range>();
        }
    };

    /**
     * @brief
     *
     * @tparam C
     * @param c
     * @return constexpr auto
     */
    template<typename C>
    constexpr inline auto position_begin(C&& c)
    {
        return std::forward<C>(c).position_begin();
    }

    /**
     * @brief
     *
     * @tparam C
     * @param c
     * @return constexpr auto
     */
    template<typename C>
    constexpr inline auto position_end(C&& c)
    {
        return std::forward<C>(c).position_end();
    }

    /**
     * @brief
     *
     * @tparam C
     * @param c
     * @return constexpr auto
     */
    template<typename C>
    constexpr inline auto inputs_begin(C&& c)
    {
        return std::forward<C>(c).inputs_begin();
    }

    /**
     * @brief
     *
     * @tparam C
     * @param c
     * @return constexpr auto
     */
    template<typename C>
    constexpr inline auto inputs_end(C&& c)
    {
        return std::forward<C>(c).inputs_end();
    }

    /**
     * @brief
     *
     * @tparam C
     * @param c
     * @return constexpr auto
     */
    template<typename C>
    constexpr inline auto outputs_begin(C&& c)
    {
        return std::forward<C>(c).outputs_begin();
    }

    /**
     * @brief
     *
     * @tparam C
     * @param c
     * @return constexpr auto
     */
    template<typename C>
    constexpr inline auto outputs_end(C&& c)
    {
        return std::forward<C>(c).outputs_end();
    }

    /**
     * @brief
     *
     * @tparam C
     * @param c
     * @return constexpr auto
     */
    template<typename C>
    constexpr inline auto variables_begin(C&& c)
    {
        return std::forward<C>(c).variables_begin();
    }

    /**
     * @brief
     *
     * @tparam C
     * @param c
     * @return constexpr auto
     */
    template<typename C>
    constexpr inline auto variables_end(C&& c)
    {
        return std::forward<C>(c).variables_end();
    }

}   // namespace scalfmm::container

#endif   // SCALFMM_CONTAINER_PARTICLE_CONTAINER_HPP
