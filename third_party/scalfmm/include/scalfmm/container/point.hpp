// --------------------------------
// See LICENCE file at project root
// File : scalfmm/container/point.hpp
// --------------------------------
#ifndef SCALFMM_CONTAINER_POINT_HPP
#define SCALFMM_CONTAINER_POINT_HPP

#include "scalfmm/container/reference_sequence.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/utils.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <tuple>
#include <type_traits>

namespace scalfmm::container
{
    /**
     * @brief classic point implementation
     *
     * @tparam Arithmetic
     * @tparam Dimension
     */
    template<typename Arithmetic, std::size_t Dimension>
    struct point_impl : public std::array<Arithmetic, Dimension>
    {
      public:
        using base_type = std::array<Arithmetic, Dimension>;
        /// Floating number type
        using value_type = Arithmetic;
        /// Dimension type
        using dimension_type = std::size_t;
        /// Space dimension count
        constexpr static const std::size_t dimension = Dimension;

        /**
         * @brief Construct a new point impl object
         *
         * @param l
         */
        point_impl(std::initializer_list<value_type> l) { std::copy(l.begin(), l.end(), this->begin()); }

        /**
         * @brief Construct a new point impl object
         *
         */
        point_impl() = default;

        /**
         * @brief Construct a new point impl object
         *
         */
        point_impl(point_impl const&) = default;

        /**
         * @brief Construct a new point impl object
         *
         */
        point_impl(point_impl&&) noexcept = default;

        /**
         * @brief
         *
         * @return point_impl&
         */
        auto operator=(point_impl const&) -> point_impl& = default;

        /**
         * @brief
         *
         * @return point_impl&
         */
        auto operator=(point_impl&&) noexcept -> point_impl& = default;

        /**
         * @brief Destroy the point impl object
         *
         */
        ~point_impl() = default;

        /**
         * @brief Construct a new point impl object
         *
         * @param a
         */
        point_impl(std::array<value_type, dimension> a)
          : base_type(a)
        {
        }

        /**
         * @brief Construct a new point impl object
         *
         * @param to_splat
         */
        explicit point_impl(value_type to_splat)
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] = to_splat;
            }
        }

        /**
         * @brief Construct a new point impl object
         *
         * @tparam U
         * @param other
         */
        template<typename U>
        point_impl(point_impl<U, dimension> const& other)
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] = value_type(other[i]);
            }
        }

        /**
         * @brief Construct a new point impl object
         *
         * @tparam U
         * @param other
         */
        template<typename U>
        explicit point_impl(U const* other)
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] = value_type(other[i]);
            }
        }
    };

    /**
     * @brief proxy for point.
     *
     * @tparam Arithmetic
     * @tparam Dimension
     */
    template<typename Arithmetic, std::size_t Dimension>
    struct point_proxy : public std::array<std::reference_wrapper<Arithmetic>, Dimension>
    {
      public:
        using base_type = std::array<std::reference_wrapper<Arithmetic>, Dimension>;
        /// Floating number type
        using value_type = std::decay_t<Arithmetic>;
        /// Reference wrapper type
        using reference_wrapper_type = std::reference_wrapper<value_type>;
        using const_reference_wrapper_type = std::reference_wrapper<std::add_const_t<value_type>>;
        /// Dimension type
        using dimension_type = std::size_t;

        /**
         * @brief Space dimension (static)
         *
         */
        constexpr static const std::size_t dimension = Dimension;

        /**
         * @brief Construct a new point proxy object
         *
         */
        point_proxy() = delete;

        /**
         * @brief Construct a new point proxy object
         *
         */
        point_proxy(point_proxy const&) = default;

        /**
         * @brief Construct a new point proxy object
         *
         */
        point_proxy(point_proxy&&) noexcept = default;

        /**
         * @brief
         *
         * @return point_proxy&
         */
        [[nodiscard]] auto operator=(point_proxy const&) -> point_proxy& = default;

        /**
         * @brief
         *
         * @return point_proxy&
         */
        [[nodiscard]] auto operator=(point_proxy&&) noexcept -> point_proxy& = default;

        /**
         * @brief Destroy the point proxy object
         *
         */
        ~point_proxy() = default;

        /**
         * @brief Construct a new point proxy object
         *
         * @param a
         */
        explicit point_proxy(std::array<value_type, dimension>& a)
          : base_type(get_reference_sequence(a))
        {
        }

        /**
         * @brief Construct a new point proxy object
         *
         * @param a
         */
        explicit point_proxy(std::array<value_type, dimension> const& a)
          : base_type(get_reference_sequence(a))
        {
        }

        /**
         * @brief Construct a new point proxy object
         *
         * @param a
         */
        explicit point_proxy(std::array<reference_wrapper_type, dimension> const& a)
          : base_type(a)
        {
        }

        /**
         * @brief Construct a new point proxy object
         *
         * @param a
         */
        explicit point_proxy(std::array<const_reference_wrapper_type, dimension> const& a)
          : base_type(a)
        {
        }

        /**
         * @brief Construct a new point proxy object
         *
         * @tparam Ts
         * @param a
         */
        template<typename... Ts, std::enable_if_t<meta::all(std::is_same_v<value_type, Ts>...), int> = 0>
        explicit point_proxy(std::tuple<Ts...>& a)
          : base_type(get_reference_sequence(a))
        {
        }

        /**
         * @brief Construct a new point proxy object
         *
         * @tparam Ts
         * @param a
         */
        template<typename... Ts, std::enable_if_t<meta::all(std::is_same_v<value_type, Ts>...), int> = 0>
        explicit point_proxy(std::tuple<Ts...> const& a)
          : base_type(get_reference_sequence(a))
        {
        }

        /**
         * @brief Construct a new point proxy object
         *
         * constructor from tuples of references
         *
         * @tparam Ts
         * @param a
         */
        template<typename... Ts,
                 std::enable_if_t<
                   meta::all(std::is_same_v<std::add_lvalue_reference_t<value_type>, Ts>...) ||
                     meta::all(std::is_same_v<std::add_lvalue_reference_t<std::add_const_t<value_type>>, Ts>...),
                   int> = 0>
        explicit point_proxy(std::tuple<Ts...>&& a)
          : base_type(get_reference_sequence(a))
        {
        }

        /**
         * @brief element access function returning the underlying reference.
         *
         * @param pos
         * @return value_type&
         */
        [[nodiscard]] constexpr inline auto at(std::size_t pos) -> value_type& { return base_type::at(pos).get(); }

        /**
         * @brief element access function returning the underlying reference.
         *
         * @param pos
         * @return value_type const&
         */
        [[nodiscard]] constexpr inline auto at(std::size_t pos) const -> value_type const&
        {
            return base_type::at(pos).get();
        }

        /**
         * @brief
         *
         * @param pos
         * @return value_type&
         */
        [[nodiscard]] constexpr inline auto operator[](std::size_t pos) -> value_type& { return at(pos); }

        /**
         * @brief
         *
         * @param pos
         * @return value_type const&
         */
        [[nodiscard]] constexpr inline auto operator[](std::size_t pos) const -> value_type const& { return at(pos); }
    };

    /**
     * @brief entry point and test if the type is arithmetic
     *
     * @tparam Arithmetic
     * @tparam Dimension
     * @tparam Enable
     */
    template<typename Arithmetic, std::size_t Dimension = 3, typename Enable = void>
    struct point
    {
        static_assert(meta::is_arithmetic<std::decay_t<Arithmetic>>::value, "Point's inner type should be arithmetic!");
    };

    /**
     * @brief
     *
     * selection on the template parameter
     * if Arithmetic is a ref with get a proxy, if not a classic point.
     *
     * @tparam Arithmetic
     * @tparam Dimension
     */
    template<typename Arithmetic, std::size_t Dimension>
    struct point<Arithmetic, Dimension,
                 typename std::enable_if<meta::is_arithmetic<std::decay_t<Arithmetic>>::value>::type>
      : std::conditional_t<std::is_reference_v<Arithmetic>, point_proxy<std::remove_reference_t<Arithmetic>, Dimension>,
                           point_impl<Arithmetic, Dimension>>
    {
        /**
         * @brief Space dimension (static).
         *
         */
        static constexpr std::size_t dimension{Dimension};

        using arithmetic_type = Arithmetic;
        using value_type = std::decay_t<Arithmetic>;
        using point_proxy_type = point_proxy<std::remove_reference_t<arithmetic_type>, dimension>;
        using point_impl_type = point_impl<std::remove_reference_t<arithmetic_type>, dimension>;
        using base_type = std::conditional_t<std::is_reference_v<Arithmetic>, point_proxy_type, point_impl_type>;

        using base_type::base_type;

        /**
         * @brief
         *
         * @param p
         * @return auto
         */
        auto operator=(point<value_type, dimension> p) noexcept
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] = p[i];
            }
            return *this;
        }

        /**
         * @brief Addition assignment operator.
         *
         * @tparam PointOrProxyOrArray
         * @param other
         * @return point&
         */
        template<typename PointOrProxyOrArray>
        inline auto operator+=(PointOrProxyOrArray const& other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] += other[i];
            }
            return *this;
        }

        /**
         * @brief Soustraction assignment operator.
         *
         * @tparam PointOrProxyOrArray
         * @param other
         * @return point&
         */
        template<typename PointOrProxyOrArray>
        inline auto operator-=(PointOrProxyOrArray const& other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] -= other[i];
            }
            return *this;
        }

        /**
         * @brief Data to data multiplication assignment.
         *
         * @tparam PointOrProxyOrArray
         * @param other
         * @return point&
         */
        template<typename PointOrProxyOrArray>
        inline auto operator*=(PointOrProxyOrArray const& other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] *= other[i];
            }
            return *this;
        }

        /**
         * @brief Data to data division assignment.
         *
         * @tparam PointOrProxyOrArray
         * @param other
         * @return point&
         */
        template<typename PointOrProxyOrArray>
        inline auto operator/=(PointOrProxyOrArray const& other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] /= other[i];
            }
            return *this;
        }

        /**
         * @brief Addition assignment operator.
         *
         * @param other
         * @return point&
         */
        inline auto operator+=(value_type other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] += other;
            }
            return *this;
        }

        /**
         * @brief Soustraction assignment operator
         *
         * @param other
         * @return point&
         */
        inline auto operator-=(value_type other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] -= other;
            }
            return *this;
        }

        /**
         * @brief Data to data multiplication assignment.
         *
         * @param other
         * @return point&
         */
        inline auto operator*=(value_type other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] *= other;
            }
            return *this;
        }

        /**
         * @brief Data to data division assignment.
         *
         * @param other
         * @return point&
         */
        inline auto operator/=(value_type other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] /= other;
            }
            return *this;
        }

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param os
         * @param pos
         * @return std::ostream&
         */
        template<typename A, std::size_t D>
        inline friend auto operator<<(std::ostream& os, const point<A, D>& pos) -> std::ostream&;

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param other
         * @param another
         * @return point<std::decay_t<A>, D>
         */
        template<typename A, std::size_t D>
        inline friend auto operator+(point other, point const& another) -> point<std::decay_t<A>, D>;

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param other
         * @param another
         * @return point<std::decay_t<A>, D>
         */
        template<typename A, std::size_t D>
        inline friend auto operator-(point other, point const& another) -> point<std::decay_t<A>, D>;

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param other
         * @param another
         * @return point<std::decay_t<A>, D>
         */
        template<typename A, std::size_t D>
        inline friend auto operator*(point other, point const& another) -> point<std::decay_t<A>, D>;

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param other
         * @param another
         * @return point<std::decay_t<A>, D>
         */
        template<typename A, std::size_t D>
        inline friend auto operator/(point other, point const& another) -> point<std::decay_t<A>, D>;

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param other
         * @param another
         * @return point<std::decay_t<A>, D>
         */
        template<typename A, std::size_t D>
        inline friend auto operator+(point other, value_type another) -> point<std::decay_t<A>, D>;

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param other
         * @param another
         * @return point<std::decay_t<A>, D>
         */
        template<typename A, std::size_t D>
        inline friend auto operator-(point other, value_type another) -> point<std::decay_t<A>, D>;

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param other
         * @param another
         * @return point<std::decay_t<A>, D>
         */
        template<typename A, std::size_t D>
        inline friend auto operator*(point other, value_type another) -> point<std::decay_t<A>, D>;

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param other
         * @param another
         * @return point<std::decay_t<A>, D>
         */
        template<typename A, std::size_t D>
        inline friend auto operator/(point other, value_type another) -> point<std::decay_t<A>, D>;

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param another
         * @param other
         * @return point<std::decay_t<A>, D>
         */
        template<typename A, std::size_t D>
        inline friend auto operator+(value_type another, point other) -> point<std::decay_t<A>, D>;

        /**
         * @brief
         *
         * @tparam A
         * @tparam D
         * @param another
         * @param other
         * @return point<std::decay_t<A>, D>
         */
        template<typename A, std::size_t D>
        inline friend auto operator*(value_type another, point other) -> point<std::decay_t<A>, D>;

        /**
         * @brief  compute the minimum of the coordinates of point_impl.
         *
         * @return the minimum of the coordinates.
         */
        inline auto min() const -> value_type
        {
            value_type min{std::numeric_limits<value_type>::max()};
            for(auto a: *this)
            {
                min = std::min(min, a);
            }
            return min;
        }

        /**
         * @brief  compute the minimum of the coordinates of point_impl.
         *
         * @return the minimum of the coordinates.
         */
        inline auto max() const -> value_type
        {
            value_type max{-std::numeric_limits<value_type>::max()};
            for(auto a: *this)
            {
                max = std::max(max, a);
            }
            return max;
        }

        /**
         * @brief norm compute the L2 norm of the point_impl.
         *
         * @return the L2 norm of the point_impl.
         */
        inline auto norm() const -> value_type
        {
            value_type square_sum{0};
            for(auto a: *this)
            {
                square_sum += a * a;
            }
            return std::sqrt(square_sum);
        }

        /**
         * @brief norm2 compute the L2 norm squared of the point_impl.
         *
         * @return the L2 norm squared of the point_impl.
         */
        inline auto norm2() const -> value_type
        {
            value_type square_sum{0};
            for(auto a: *this)
            {
                square_sum += a * a;
            }
            return square_sum;
        }

        /**
         * @brief
         *
         * @tparam PointOrProxyOrArray
         * @param p
         * @return value_type
         */
        template<typename PointOrProxyOrArray>
        inline auto distance(PointOrProxyOrArray const& p) const -> value_type
        {
            value_type square_sum{0};
            for(std::size_t i{0}; i < dimension; ++i)
            {
                auto tmp = (*this)[i] - p.at(i);
                square_sum += tmp * tmp;
            }
            return std::sqrt(square_sum);
        }
    };

    /**
     * @brief
     *
     * @tparam Arithmetic
     * @tparam Dimension
     * @param os
     * @param pos
     * @return std::ostream&
     */
    template<typename Arithmetic, std::size_t Dimension>
    inline auto operator<<(std::ostream& os, const point<Arithmetic, Dimension>& pos) -> std::ostream&
    {
        os << "[";
        for(std::size_t i{0}; i < Dimension - 1; ++i)
        {
            os << pos.at(i) << ", ";
        }
        os << pos.at(Dimension - 1) << "]";
        return os;
    }

    /**
     * @brief Addition operator.
     *
     * @tparam A
     * @tparam D
     * @param other
     * @param another
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator+(point<A, D> other, point<A, D> const& another) -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) + another.at(i);
        }
        return res;
    }

    /**
     * @brief Substraction operator.
     *
     * @tparam A
     * @tparam D
     * @param other
     * @param another
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator-(point<A, D> other, point<A, D> const& another) -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) - another.at(i);
        }
        return res;
    }

    /**
     * @brief Multiply operator.
     *
     * @tparam A
     * @tparam D
     * @param other
     * @param another
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator*(point<A, D> other, point<A, D> const& another) -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) * another.at(i);
        }
        return res;
    }

    /**
     * @brief Divide operator.
     *
     * @tparam A
     * @tparam D
     * @param other
     * @param another
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator/(point<A, D> other, point<A, D> const& another) -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) / another.at(i);
        }
        return res;
    }

    /**
     * @brief Addition operator.
     *
     * @tparam A
     * @tparam D
     * @param other
     * @param another
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator+(point<A, D> other, std::decay_t<A> another) -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) + another;
        }
        return res;
    }

    /**
     * @brief Substraction operator.
     *
     * @tparam A
     * @tparam D
     * @param other
     * @param another
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator-(point<A, D> other, std::decay_t<A> another) -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) - another;
        }
        return res;
    }

    /**
     * @brief Multiply operator.
     *
     * @tparam A
     * @tparam D
     * @param other
     * @param another
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator*(point<A, D> other, std::decay_t<A> another) -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) * another;
        }
        return res;
    }

    /**
     * @brief Divide operator.
     *
     * @tparam A
     * @tparam D
     * @param other
     * @param another
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator/(point<A, D> other, std::decay_t<A> another) -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) / another;
        }
        return res;
    }

    /**
     * @brief Addition operator.
     *
     * @tparam A
     * @tparam D
     * @param another
     * @param other
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator+(std::decay_t<A> another, point<A, D> other) -> point<std::decay_t<A>, D>
    {
        return other + another;
    }

    /**
     * @brief Multiply operator.
     *
     * @tparam A
     * @tparam D
     * @param another
     * @param other
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator*(std::decay_t<A> another, point<A, D> other) -> point<std::decay_t<A>, D>
    {
        return other * another;
    }

    /**
     * @brief Divide operator.
     *
     * @tparam A
     * @tparam D
     * @param another
     * @param other
     * @return point<std::decay_t<A>, D>
     */
    template<typename A, std::size_t D>
    inline auto operator/(std::decay_t<A> another, point<A, D> other) -> point<std::decay_t<A>, D>
    {
        point<A, D> val(another);
        return val / other;
    }

    /**
     * @brief  Equality test operator (for SIMD case)
     *
     * @tparam Arithmetic xsimd type
     * @tparam Dimension
     * @param lhs
     * @param rhs
     * @return std::enable_if_t<meta::is_simd<Arithmetic>::value, xsimd::batch_bool<typename Arithmetic:value_type>>
     */
    template<typename Arithmetic, std::size_t Dimension>
    inline auto operator==(const point_impl<Arithmetic, Dimension>& lhs, const point_impl<Arithmetic, Dimension>& rhs)
      -> std::enable_if_t<meta::is_simd<Arithmetic>::value, xsimd::batch_bool<typename Arithmetic::value_type>>
    {
        auto lhs_it = lhs.begin();
        auto rhs_it = rhs.begin();

        auto equal = *lhs_it++ == *rhs_it++;

        for(std::size_t i = 1; i < Dimension; i++)
        {
            equal = equal && (*lhs_it++ == *rhs_it++);
        }

        return equal;
    }
}   // end of namespace scalfmm::container

#endif
