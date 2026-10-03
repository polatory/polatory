// --------------------------------
// See LICENCE file at project root
// File : scalfmm/memory/storage.hpp
// --------------------------------
#ifndef SCALFMM_MEMORY_STORAGE_HPP
#define SCALFMM_MEMORY_STORAGE_HPP

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/options/options.hpp"

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xfixed.hpp"
#include "xtensor/containers/xtensor.hpp"

#include <array>
#include <complex>
#include <vector>

namespace scalfmm::memory
{
    /**
     * @brief
     *
     * @tparam S
     */
    template<typename S>
    struct storage_traits;

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam _static
     * @tparam axes
     */
    template<typename ValueType, std::size_t Dimension, bool _static, std::size_t... axes>
    struct alignas(XTENSOR_FIXED_ALIGN) tensor_storage
    {
        static constexpr std::size_t dimension = Dimension;
        using self_type = tensor_storage<ValueType, Dimension, _static, axes...>;
        using value_type = typename storage_traits<self_type>::value_type;
        using inner_type = typename storage_traits<self_type>::inner_type;
        using tensor_type = typename storage_traits<self_type>::tensor_type;
        using outer_shape = typename storage_traits<self_type>::outer_shape;

        /**
         * @brief Construct a new tensor storage object
         *
         */
        tensor_storage() = default;

        /**
         * @brief Construct a new tensor storage object
         *
         */
        tensor_storage(tensor_storage const&) = default;

        /**
         * @brief Construct a new tensor storage object
         *
         */
        tensor_storage(tensor_storage&&) noexcept = default;

        /**
         * @brief
         *
         * @return tensor_storage&
         */
        inline auto operator=(tensor_storage const&) -> tensor_storage& = default;

        /**
         * @brief
         *
         * @return tensor_storage&
         */
        inline auto operator=(tensor_storage&&) noexcept -> tensor_storage& = default;

        /**
         * @brief Destroy the tensor storage object
         *
         */
        ~tensor_storage() = default;

        /**
         * @brief Construct a new tensor storage object
         *
         * @param size
         * @param init
         */
        tensor_storage(std::size_t size, value_type init = value_type(0.))
        {
            std::vector shape(dimension, size);

            std::array stops{axes...};

            auto fill = [&](auto... is)
            {
                auto& t_at_is = m_tensor.at(is...);
                t_at_is.resize(shape);
                std::fill(t_at_is.begin(), t_at_is.end(), init);
            };

            meta::looper<sizeof...(axes)>{}(fill, stops);
        }

        /**
         * @brief Construct a new tensor storage object
         *
         * @param shape
         * @param init
         */
        tensor_storage(std::initializer_list<std::size_t> shape, value_type init = value_type(0.))
        {
            std::array stops{axes...};

            auto fill = [&](auto... is)
            {
                auto& t_at_is = m_tensor.at(is...);
                t_at_is.resize(shape);
                std::fill(t_at_is.begin(), t_at_is.end(), init);
            };

            meta::looper<sizeof...(axes)>{}(fill, stops);
        }

        /**
         * @brief Construct a new tensor storage object
         *
         * @param shape
         * @param init
         */
        tensor_storage(std::vector<std::size_t> shape, value_type init = value_type(0.))
        {
            std::array stops{axes...};

            auto fill = [&](auto... is)
            {
                auto& t_at_is = m_tensor.at(is...);
                t_at_is.resize(shape);
                std::fill(t_at_is.begin(), t_at_is.end(), init);
            };

            meta::looper<sizeof...(axes)>{}(fill, stops);
        }

        /**
         * @brief
         *
         */
        auto reset() noexcept -> void
        {
            std::array stops{axes...};

            auto fill = [&](auto... is)
            {
                auto& t_at_is = m_tensor.at(is...);
                std::fill(t_at_is.begin(), t_at_is.end(), value_type(0.0));
            };

            meta::looper<sizeof...(axes)>{}(fill, stops);
        }

        /**
         * @brief
         *
         * @return tensor_type const&
         */
        auto get() const noexcept -> tensor_type const& { return m_tensor; }

        /**
         * @brief
         *
         * @return tensor_type&
         */
        auto get() noexcept -> tensor_type& { return m_tensor; }

        /**
         * @brief
         *
         * @tparam Is
         * @param i
         * @return inner_type const&
         */
        template<typename... Is>
        constexpr auto at(Is... i) const noexcept -> inner_type const&
        {
            return m_tensor.at(i...);
        }

        /**
         * @brief
         *
         * @tparam Is
         * @param i
         * @return inner_type&
         */
        template<typename... Is>
        constexpr auto at(Is... i) noexcept -> inner_type&
        {
            return m_tensor.at(i...);
        }

      private:
        tensor_type m_tensor{};
    };

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam _static
     * @tparam axes
     */
    template<typename ValueType, std::size_t Dimension, bool _static, std::size_t... axes>
    struct storage_traits<tensor_storage<ValueType, Dimension, _static, axes...>>
    {
        using value_type = ValueType;
        using inner_type = std::conditional_t<_static, xt::xtensor<value_type, Dimension>, xt::xarray<value_type>>;
        using outer_shape = xt::xshape<axes...>;
        using tensor_type = xt::xtensor_fixed<inner_type, outer_shape>;
    };

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam k
     */
    template<typename ValueType, std::size_t Dimension, std::size_t k>
    struct alignas(XTENSOR_FIXED_ALIGN) multipoles_storage : protected tensor_storage<ValueType, Dimension, true, k>
    {
        static constexpr std::size_t dimension = Dimension;
        using base_type = tensor_storage<ValueType, Dimension, true, k>;
        using self_type = multipoles_storage<ValueType, Dimension, k>;
        using value_type = typename storage_traits<base_type>::value_type;
        using inner_type = typename storage_traits<base_type>::inner_type;
        using tensor_type = typename storage_traits<base_type>::tensor_type;
        using outer_shape = typename storage_traits<base_type>::outer_shape;
        using multipoles_iterator_type = std::array<typename inner_type::storage_type::iterator, k>;
        using multipoles_const_iterator_type = std::array<typename inner_type::storage_type::const_iterator, k>;
        using base_type::base_type;
        using base_type::get;

        /**
         * @brief
         *
         * @return tensor_type const&
         */
        auto multipoles() const noexcept -> tensor_type const& { return get(); }

        /**
         * @brief
         *
         * @return tensor_type const&
         */
        auto cmultipoles() const noexcept -> tensor_type const& { return get(); }

        /**
         * @brief
         *
         * @return tensor_type&
         */
        auto multipoles() noexcept -> tensor_type& { return get(); }

        /**
         * @brief
         *
         * @param i
         * @return inner_type const&
         */
        constexpr auto multipoles(std::size_t i) const noexcept -> inner_type const& { return this->at(i); }

        /**
         * @brief
         *
         * @param i
         * @return inner_type const&
         */
        constexpr auto cmultipoles(std::size_t i) const noexcept -> inner_type const& { return this->at(i); }

        /**
         * @brief
         *
         * @param i
         * @return inner_type&
         */
        constexpr auto multipoles(std::size_t i) noexcept -> inner_type& { return this->at(i); }

        /**
         * @brief
         *
         */
        inline auto reset_multipoles() noexcept -> void { base_type::reset(); }

        // Accessors to multipoles and locals iterators

        /**
         * @brief
         *
         * @return multipoles_iterator_type
         */
        [[nodiscard]] inline auto multipoles_begin() -> multipoles_iterator_type
        {
            multipoles_iterator_type its;
            for(std::size_t m{0}; m < k; ++m)
            {
                its.at(m) = this->at(m).begin();
            }
            return its;
        }

        /**
         * @brief
         *
         * @return multipoles_const_iterator_type
         */
        [[nodiscard]] inline auto multipoles_begin() const -> multipoles_const_iterator_type
        {
            multipoles_const_iterator_type its;
            for(std::size_t m{0}; m < k; ++m)
            {
                its.at(m) = this->at(m).cbegin();
            }
            return its;
        }

        /**
         * @brief
         *
         * @return multipoles_const_iterator_type
         */
        [[nodiscard]] inline auto cmultipoles_begin() const -> multipoles_const_iterator_type
        {
            multipoles_const_iterator_type its;
            for(std::size_t m{0}; m < k; ++m)
            {
                its.at(m) = this->at(m).cbegin();
            }
            return its;
        }

        /**
         * @brief
         *
         * @return multipoles_iterator_type
         */
        [[nodiscard]] inline auto multipoles_end() -> multipoles_iterator_type
        {
            multipoles_iterator_type its;
            for(std::size_t m{0}; m < k; ++m)
            {
                its.at(m) = this->at(m).end();
            }
            return its;
        }

        /**
         * @brief
         *
         * @return multipoles_const_iterator_type
         */
        [[nodiscard]] inline auto multipoles_end() const -> multipoles_const_iterator_type
        {
            multipoles_const_iterator_type its;
            for(std::size_t m{0}; m < k; ++m)
            {
                its.at(m) = this->at(m).cend();
            }
            return its;
        }

        /**
         * @brief
         *
         * @return multipoles_const_iterator_type
         */
        [[nodiscard]] inline auto cmultipoles_end() const -> multipoles_const_iterator_type
        {
            multipoles_const_iterator_type its;
            for(std::size_t m{0}; m < k; ++m)
            {
                its.at(m) = this->at(m).cend();
            }
            return its;
        }
    };

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam axes
     */
    template<typename ValueType, std::size_t Dimension, std::size_t... axes>
    struct storage_traits<multipoles_storage<ValueType, Dimension, axes...>>
      : storage_traits<tensor_storage<ValueType, Dimension, true, axes...>>
    {
    };

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam k
     */
    template<typename ValueType, std::size_t Dimension, std::size_t k>
    struct alignas(XTENSOR_FIXED_ALIGN) locals_storage : private tensor_storage<ValueType, Dimension, true, k>
    {
        static constexpr std::size_t dimension = Dimension;
        using base_type = tensor_storage<ValueType, Dimension, true, k>;
        using self_type = locals_storage<ValueType, Dimension, k>;
        using value_type = typename storage_traits<base_type>::value_type;
        using inner_type = typename storage_traits<base_type>::inner_type;
        using tensor_type = typename storage_traits<base_type>::tensor_type;
        using outer_shape = typename storage_traits<base_type>::outer_shape;
        using locals_iterator_type = std::array<typename inner_type::storage_type::iterator, k>;
        using locals_const_iterator_type = std::array<typename inner_type::storage_type::const_iterator, k>;

        using base_type::base_type;
        using base_type::get;

        /**
         * @brief
         *
         * @return tensor_type const&
         */
        auto locals() const noexcept -> tensor_type const& { return get(); }

        /**
         * @brief
         *
         * @return tensor_type const&
         */
        auto clocals() const noexcept -> tensor_type const& { return get(); }

        /**
         * @brief
         *
         * @return tensor_type&
         */
        auto locals() noexcept -> tensor_type& { return get(); }

        /**
         * @brief
         *
         * @param i
         * @return inner_type const&
         */
        constexpr auto locals(std::size_t i) const noexcept -> inner_type const& { return this->at(i); }

        /**
         * @brief
         *
         * @param i
         * @return inner_type const&
         */
        constexpr auto clocals(std::size_t i) const noexcept -> inner_type const& { return this->at(i); }

        /**
         * @brief
         *
         * @param i
         * @return inner_type&
         */
        constexpr auto locals(std::size_t i) noexcept -> inner_type& { return this->at(i); }

        /**
         * @brief
         *
         */
        inline auto reset_locals() noexcept -> void { base_type::reset(); }

        // Accesors to multipoles and locals iterators

        /**
         * @brief
         *
         * @return locals_iterator_type
         */
        [[nodiscard]] inline auto locals_begin() -> locals_iterator_type
        {
            locals_iterator_type its;
            for(std::size_t n{0}; n < k; ++n)
            {
                its.at(n) = this->at(n).begin();
            }
            return its;
        }

        /**
         * @brief
         *
         * @return locals_const_iterator_type
         */
        [[nodiscard]] inline auto locals_begin() const -> locals_const_iterator_type
        {
            locals_const_iterator_type its;
            for(std::size_t n{0}; n < k; ++n)
            {
                its.at(n) = this->at(n).cbegin();
            }
            return its;
        }

        /**
         * @brief
         *
         * @return locals_const_iterator_type
         */
        [[nodiscard]] inline auto clocals_begin() const -> locals_const_iterator_type
        {
            locals_const_iterator_type its;
            for(std::size_t n{0}; n < k; ++n)
            {
                its.at(n) = this->at(n).cbegin();
            }
            return its;
        }

        /**
         * @brief
         *
         * @return locals_iterator_type
         */
        [[nodiscard]] inline auto locals_end() -> locals_iterator_type
        {
            locals_iterator_type its;
            for(std::size_t n{0}; n < k; ++n)
            {
                its.at(n) = this->at(n).end();
            }
            return its;
        }

        /**
         * @brief
         *
         * @return locals_const_iterator_type
         */
        [[nodiscard]] inline auto locals_end() const -> locals_const_iterator_type
        {
            locals_const_iterator_type its;
            for(std::size_t n{0}; n < k; ++n)
            {
                its.at(n) = this->at(n).cend();
            }
            return its;
        }

        /**
         * @brief
         *
         * @return locals_const_iterator_type
         */
        [[nodiscard]] inline auto clocals_end() const -> locals_const_iterator_type
        {
            locals_const_iterator_type its;
            for(std::size_t n{0}; n < k; ++n)
            {
                its.at(n) = this->at(n).cend();
            }
            return its;
        }
    };

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam axes
     */
    template<typename ValueType, std::size_t Dimension, std::size_t... axes>
    struct storage_traits<locals_storage<ValueType, Dimension, axes...>>
      : storage_traits<tensor_storage<ValueType, Dimension, true, axes...>>
    {
    };

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam k
     */
    template<typename ValueType, std::size_t Dimension, std::size_t k>
    struct alignas(XTENSOR_FIXED_ALIGN) transformed_multipoles_storage
      : private tensor_storage<ValueType, Dimension, false, k>
    {
        static constexpr std::size_t dimension = Dimension;
        using base_type = tensor_storage<ValueType, Dimension, false, k>;
        using self_type = transformed_multipoles_storage<ValueType, Dimension, k>;
        using value_type = typename storage_traits<base_type>::value_type;
        using inner_type = typename storage_traits<base_type>::inner_type;
        using tensor_type = typename storage_traits<base_type>::tensor_type;
        using outer_shape = typename storage_traits<base_type>::outer_shape;

        using base_type::base_type;
        using base_type::get;

        /**
         * @brief Construct a new transformed multipoles storage object
         *
         */
        transformed_multipoles_storage() = default;

        /**
         * @brief Construct a new transformed multipoles storage object
         *
         */
        transformed_multipoles_storage(transformed_multipoles_storage const&) = default;

        /**
         * @brief Construct a new transformed multipoles storage object
         *
         */
        transformed_multipoles_storage(transformed_multipoles_storage&&) noexcept = default;

        /**
         * @brief
         *
         * @return transformed_multipoles_storage&
         */
        inline auto operator=(transformed_multipoles_storage const&) -> transformed_multipoles_storage& = default;

        /**
         * @brief
         *
         * @return transformed_multipoles_storage&
         */
        inline auto operator=(transformed_multipoles_storage&&) noexcept -> transformed_multipoles_storage& = default;

        /**
         * @brief Destroy the transformed multipoles storage object
         *
         */
        ~transformed_multipoles_storage() = default;

        /**
         * @brief Construct a new transformed multipoles storage object
         *
         * @param size
         * @param init
         */
        transformed_multipoles_storage(std::size_t size, value_type init = value_type(0.))
          : base_type(compute_shape(size), init)
        {
        }

        /**
         * @brief
         *
         * @return tensor_type const&
         */
        auto transformed_multipoles() const noexcept -> tensor_type const& { return get(); }

        /**
         * @brief
         *
         * @return tensor_type const&
         */
        auto ctransformed_multipoles() const noexcept -> tensor_type const& { return get(); }

        /**
         * @brief
         *
         * @return tensor_type&
         */
        auto transformed_multipoles() noexcept -> tensor_type& { return get(); }

        /**
         * @brief
         *
         * @param i
         * @return inner_type const&
         */
        constexpr auto transformed_multipoles(std::size_t i) const noexcept -> inner_type const& { return get(i); }

        /**
         * @brief
         *
         * @param i
         * @return inner_type const&
         */
        constexpr auto ctransformed_multipoles(std::size_t i) const noexcept -> inner_type const& { return get(i); }

        /**
         * @brief
         *
         * @param i
         * @return inner_type&
         */
        constexpr auto transformed_multipoles(std::size_t i) noexcept -> inner_type& { return get(i); }

      private:
        /**
         * @brief
         *
         * @param size
         * @return std::vector<std::size_t>
         */
        auto compute_shape(std::size_t size) -> std::vector<std::size_t>
        {
            std::vector shape(dimension, 2 * size - 1);
            shape.at(dimension - 1) = size;
            return shape;
        }
    };

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam axes
     */
    template<typename ValueType, std::size_t Dimension, std::size_t... axes>
    struct storage_traits<transformed_multipoles_storage<ValueType, Dimension, axes...>>
      : storage_traits<tensor_storage<ValueType, Dimension, false, axes...>>
    {
    };

    /**
     * @brief
     *
     * @tparam D_
     * @tparam D
     * @return std::size_t
     */
    template<std::size_t D_, std::size_t... D>
    static constexpr auto check_dimensions() -> std::size_t
    {
        if constexpr(((D_ == D) && ...))
        {
            return D_;
        }
        else
        {
            return 0;
        }
    }

    /**
     * @brief
     *
     * @tparam Storages
     */
    template<typename... Storages>
    struct alignas(XTENSOR_FIXED_ALIGN) aggregate_storage : public Storages...
    {
        static constexpr std::size_t dimension{check_dimensions<Storages::dimension...>()};
        static constexpr std::size_t number_aggregates = sizeof...(Storages);
        static_assert(dimension != 0, "Storages have different dimensions !");
        using self_type = aggregate_storage<Storages...>;
        using value_type = std::tuple<typename storage_traits<Storages>::value_type...>;
        using inner_type = std::tuple<typename storage_traits<Storages>::inner_type...>;   // the true tensor
        using tensor_type = std::tuple<typename storage_traits<Storages>::tensor_type...>;
        using outer_shape = std::tuple<typename storage_traits<Storages>::outer_shape...>;

        /**
         * @brief Construct a new aggregate storage object
         *
         */
        aggregate_storage() = default;

        /**
         * @brief Construct a new aggregate storage object
         *
         */
        aggregate_storage(aggregate_storage const&) = default;

        /**
         * @brief Construct a new aggregate storage object
         *
         */
        aggregate_storage(aggregate_storage&&) noexcept = default;

        /**
         * @brief
         *
         * @return aggregate_storage&
         */
        inline auto operator=(aggregate_storage const&) -> aggregate_storage& = default;

        /**
         * @brief
         *
         * @return aggregate_storage&
         */
        inline auto operator=(aggregate_storage&&) noexcept -> aggregate_storage& = default;

        /**
         * @brief Destroy the aggregate storage object
         *
         */
        ~aggregate_storage() = default;

        /**
         * @brief Construct a new aggregate storage object
         *
         * @param size
         */
        aggregate_storage(std::size_t size)
          : Storages(size)...
        {
        }

        /**
         * @brief Construct a new aggregate storage object
         *
         * @param size
         * @param init
         */
        aggregate_storage(std::size_t size, typename storage_traits<Storages>::value_type... init)
          : Storages(size, init)...
        {
        }

        /**
         * @brief Construct a new aggregate storage object
         *
         * @tparam ShapeVector
         * @param shape
         */
        template<typename... ShapeVector>
        aggregate_storage(ShapeVector... shape)
          : Storages(shape)...
        {
        }

        /**
         * @brief Construct a new aggregate storage object
         *
         * @tparam ShapeVector
         * @tparam ValueType
         * @param pair
         */
        template<typename... ShapeVector, typename... ValueType>
        aggregate_storage(std::pair<ShapeVector, ValueType>... pair)
          : Storages(pair.first, pair.second)...
        {
        }
    };

    /**
     * @brief
     *
     * @tparam Storages
     */
    template<typename... Storages>
    struct storage_traits<aggregate_storage<Storages...>>
    {
        static constexpr std::size_t number_aggregates = sizeof...(Storages);
        using value_type = std::tuple<typename storage_traits<Storages>::value_type...>;
        using inner_type = std::tuple<typename storage_traits<Storages>::inner_type...>;
        using tensor_type = std::tuple<typename storage_traits<Storages>::tensor_type...>;
        using outer_shape = std::tuple<typename storage_traits<Storages>::outer_shape...>;
    };

    /**
     * @brief
     *
     * @tparam T
     * @tparam typename
     */
    template<typename T, typename = void>
    struct id_value_type
    {
        using type = T;
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct id_value_type<T, std::void_t<typename T::value_type>>
    {
        using type = typename T::value_type;
    };
}   // namespace scalfmm::memory

#endif   // SCALFMM_MEMORY_STORAGE_HPP
