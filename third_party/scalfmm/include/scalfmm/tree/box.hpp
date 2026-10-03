// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tree/box.hpp
// --------------------------------
#ifndef SCALFMM_TREE_BOX_HPP
#define SCALFMM_TREE_BOX_HPP

#include "scalfmm/tree/morton_curve.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>

namespace scalfmm::component
{
    /**
     * @brief Implements a N dimensions box
     *
     * @author Quentin Khan <quentin.khan@inria.fr>
     *
     * The box is described by two opposite corners : the maximum and
     * the minimum one. All the class transformations maintain this
     * predicate.
     *
     * @tparam Position
     * @tparam SpaceFillingCurve A templatize implementation of a space filling curve
     */
    template<class Position, template<std::size_t> class SpaceFillingCurve = z_curve>
    class box
    {
      public:
        /// Position type
        using position_type = Position;
        /// Floating number representation
        using value_type = typename position_type::value_type;
        /// Space dimension
        constexpr static const std::size_t dimension = position_type::dimension;
        /// Space filling curve type
        using space_filling_curve_t = SpaceFillingCurve<dimension>;

      private:
        /**
         * @brief Minimum corner.
         *
         */
        position_type m_c1;

        /**
         * @brief Maximum corner.
         *
         */
        position_type m_c2;

        /**
         * @brief Center.
         *
         */
        position_type m_center;

        /**
         * @brief
         *
         */
        space_filling_curve_t m_space_filling_curve;

        /**
         * @brief get the periodicity per direction.
         *
         */
        std::array<bool, dimension> m_periodicity;

        /**
         * @brief Boolean indicating if a direction is periodic.
         *
         */
        bool m_is_periodic;

        /**
         * @brief Rearranges the corners to ensure the maximum-minimum predicate.
         *
         */
        auto rearrange_corners() -> void
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                if(m_c1[i] > m_c2[i])
                {
                    std::swap(m_c1[i], m_c2[i]);
                }
            }
            m_center = (m_c1 + m_c2) * 0.5;
        }

      public:
        /**
         * @brief Accessor for the minimum corner.
         *
         * @return position_type const&
         */
        [[nodiscard]] inline auto c1() const noexcept -> position_type const& { return m_c1; }

        /**
         * @brief Accessor for the maximum corner.
         *
         * @return position_type const&
         */
        [[nodiscard]] inline auto c2() const noexcept -> position_type const& { return m_c2; }

        /**
         * @brief Construct a new box object
         *
         * Builds an empty box at the origin
         *
         */
        box() = default;

        /**
         * @brief Construct a new box object
         *
         * Copies an existing box
         *
         */
        box(const box&) = default;

        /**
         * @brief
         *
         * @param other
         * @return box&
         */
        auto operator=(const box& other) -> box& = default;

        /**
         * @brief Construct a new box object
         *
         * Move constructor
         */
        box(box&&) noexcept = default;

        /**
         * @brief
         *
         * Move assignment
         *
         * @param other
         * @return box&
         */
        auto operator=(box&& other) noexcept -> box& = default;

        /**
         * @brief Destroy the box object
         *
         */
        ~box() = default;

        /**
         * @brief Builds a cube from the lower corner and its side length.
         *
         * @param min_corner
         * @param side_length
         */
        [[deprecated]] box(const position_type& min_corner, value_type side_length)
          : m_c1(min_corner)
          , m_c2(min_corner)
          , m_space_filling_curve()
          , m_is_periodic(false)
        {
            if(side_length < 0)
            {
                side_length = -side_length;
            }
            for(auto& v: m_periodicity)
            {
                v = false;
            }
            for(auto&& d: m_c2)
            {
                d += side_length;
            }

            m_center = (m_c1 + m_c2) * 0.5;
        }

        /**
         * @brief Construct a new box object
         *
         * Builds a cube using its center and width.
         *
         * @param width Cube width.
         * @param box_center Cube center.
         */
        box(value_type width, position_type const& box_center)
          : m_c1(box_center)
          , m_c2(box_center)
          , m_center(box_center)
          , m_space_filling_curve()
          , m_is_periodic(false)
        {
            if(width < 0)
            {
                width = -width;
            }
            for(auto& v: m_periodicity)
            {
                v = false;
            }
            value_type radius = width / 2;

            for(auto&& d: m_c1)
            {
                d -= radius;
            }

            for(auto&& d: m_c2)
            {
                d += radius;
            }
        }

        /**
         * @brief Builds a box from two corners.
         *
         * The maximum and minimum corners are deduced from the given corners.
         *
         * @param corner_1
         * @param corner_2
         */
        box(const position_type& corner_1, const position_type& corner_2)
          : m_c1(corner_1)
          , m_c2(corner_2)
          , m_space_filling_curve()
          , m_is_periodic(false)
        {
            for(auto& v: m_periodicity)
            {
                v = false;
            }

            rearrange_corners();
        }

        /**
         * @brief Changes the box corners.
         *
         * The maximum and minimum corners are deduced from the given corners.
         *
         * @param corner_1
         * @param corner_2
         */
        auto set(const position_type& corner_1, const position_type& corner_2) -> void
        {
            m_c1 = corner_1;
            m_c2 = corner_2;

            rearrange_corners();
        }

        /**
         * @brief Checks whether a position is within the box bounds.
         *
         * @param p The position to check.
         * @return true
         * @return false
         */
        [[nodiscard]] auto contains(const position_type& p) const -> bool
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                if(p[i] < m_c1[i] || p[i] > m_c2[i])
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Checks whether an object's position is within the box bounds.
         *
         * The object must implement a 'position_type position() const;' method.
         *
         * @tparam T The object type.
         * @param obj The object which position to check.
         * @return true
         * @return false
         */
        template<class T>
        [[nodiscard]] inline auto contains(const T& obj) const -> bool
        {
            return contains(obj.position());
        }

        /**
         * @brief Accessor for the box center.
         *
         * @return position_type const&
         */
        [[nodiscard]] inline auto center() const -> position_type const& { return m_center; }

        /**
         * @brief Access for the box center.
         *
         * @tparam Int
         * @param added_levels
         * @return position_type const
         */
        template<typename Int>
        [[nodiscard]] auto extended_center(Int added_levels) const -> position_type const
        {
            if(m_is_periodic)
            {
                const auto original_width = this->width(0);
                const auto offset = 0.5 * (this->extended_width(added_levels) - original_width);
                return m_center + position_type(offset);
            }
            else
            {
                std::clog << "Warning extended_center called but not periodic simulation.\n";
                return m_center;
            }
        }

        /**
         * @brief Accessor for the box corners.
         *
         * The corners are numbered using a space filling curve.
         *
         * @param idx The corner index.
         * @return position_type The idx'th corner.
         */
        [[nodiscard]] auto corner(std::size_t idx) const -> position_type
        {
            position_type c;
            std::size_t i = 0;
            for(bool choice: m_space_filling_curve.position(idx))
            {
                c[i] = choice ? m_c2[i] : m_c1[i];
                ++i;
            }
            return c;
        }

        /**
         * @brief Setter for the corners.
         *
         * Moves a corner to a new position and modifies the relevant other
         * ones. The corners are numbered using a space filling curve.
         *
         * @param idx The moved corner index.
         * @param pos The new corner position.
         */
        auto corner(std::size_t idx, const position_type& pos) -> void
        {
            std::size_t i = 0;
            for(bool choice: m_space_filling_curve.position(idx))
            {
                if(choice)
                {
                    m_c2[i] = pos[i];
                }
                else
                {
                    m_c1[i] = pos[i];
                }
                ++i;
            }
            rearrange_corners();
        }
#ifdef scalfmm_BUILD_PBC

        /**
         * @brief Set the periodicity in the direction dir.
         *
         * @param dir direction.
         * @param per true if the direction dir is periodic otherwise false.
         */
        auto set_periodicity(int dir, bool per) -> void
        {
            m_periodicity[dir] = per;
            m_is_periodic = m_is_periodic || per;
        }

        /**
         * @brief Set the periodicity in all directions.
         *
         * @tparam Vector
         * @param pbl a vector of boolean that specifies whether the direction is periodic or not.
         */
        template<typename Vector>
        auto set_periodicity(const Vector& pbl) -> void
        {
            for(std::size_t d = 0; d < pbl.size(); ++d)
                set_periodicity(d, pbl[d]);
        }
#endif

        /**
         * @brief Returns the array of periodic direction.
         *
         * @return std::array<bool, dimension>
         */
        inline auto get_periodicity() const -> std::array<bool, dimension> { return m_periodicity; }

        /**
         * @brief Tells whether there is a periodic direction.
         *
         * @return true
         * @return false
         */
        inline auto is_periodic() const noexcept -> bool { return m_is_periodic; }

        /**
         * @brief Returns the width for given dimension.
         *
         * @param dim
         * @return decltype(std::abs(m_c2[dim] - m_c1[dim]))
         */
        [[nodiscard]] inline auto width(std::size_t dim) const noexcept -> decltype(std::abs(m_c2[dim] - m_c1[dim]))
        {
            return std::abs(m_c2[dim] - m_c1[dim]);
        }

        /**
         * @brief Returns the extended width for periodic simulation for added_levels.
         *
         * @tparam Int
         * @param added_levels
         * @return value_type
         */
        template<typename Int>
        [[nodiscard]] auto extended_width(Int added_levels) const noexcept -> value_type
        {
            auto width = this->width(0);

            if(m_is_periodic || added_levels > 0)
            {
                width *= ((4) << added_levels);
            }
            else
            {
                std::clog << "Warning extended_width called but not periodic simulation.\n";
            }
            return width;
        }

        /**
         * @brief Sums the corners of two boxes
         *
         * @param other
         * @return box
         */
        inline auto operator+(const box& other) const -> box { return box(m_c1 + other.m_c1, m_c2 + other.m_c2); }

        /**
         * @brief Tests two boxes equality
         *
         * @param other
         * @return true
         * @return false
         */
        inline auto operator==(const box& other) const -> bool { return c1() == other.c1() && c2() == other.c2(); }

        /**
         * @brief Tests two boxes inequality
         *
         * @param other
         * @return true
         * @return false
         */
        inline auto operator!=(const box& other) const -> bool { return !this->operator==(other); }

        /**
         * @brief
         *
         * @param os
         * @param box
         * @return std::ostream&
         */
        friend auto operator<<(std::ostream& os, const box& box) -> std::ostream&
        {
            os << " [" << box.c1() << "," << box.c2() << "] ; periodicity: ";
            if(box.is_periodic())
            {
                os << std::boolalpha << " ( ";
                const auto& per = box.get_periodicity();
                for(std::size_t i = 0; i < dimension - 1; ++i)
                {
                    os << per[i] << ", ";
                }
                os << per[dimension - 1] << ")";
            }
            else
            {
                os << std::boolalpha << " none ";
            }
            return os;
        }
    };
}   // namespace scalfmm::component
#endif
