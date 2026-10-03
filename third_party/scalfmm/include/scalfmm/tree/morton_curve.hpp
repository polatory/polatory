// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tree/morton_curve.hpp
// --------------------------------
#ifndef SCALFMM_TREE_MORTON_CURVE_HPP
#define SCALFMM_TREE_MORTON_CURVE_HPP

#include "scalfmm/container/point.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/utils/math.hpp"

#include <array>
#include <cstddef>

namespace scalfmm::component
{
    /** Provides the corner traversal order of an N dimension hypercube
     *
     * The positions returned are array of booleans. Each boolean tells where
     * to place the element in the binary grid.
     *
     * For instance, in 2D:
     *
     *
     *         __0__ __1__
     *        |     |     |
     *       0|     |  X  |  pos(X) = [true, false]
     *        |_____|_____|
     *        |     |     |
     *       1|     |  Y  |  pos(Y) = [true, true ]
     *        |_____|_____|
     *
     *
     * @tparam Dimension The hypercube dimension.
     */
    template<std::size_t Dimension>
    struct z_curve
    {
      public:
        /**
         * @brief Static dimension of the space.
         *
         */
        static constexpr std::size_t dimension = Dimension;
        /// Template alias
        template<typename T>
        using position_alias = container::point<T, dimension>;
        /// Position type used
        using position_type = container::point<bool, dimension>;

        /**
         * @brief Position count in the grid.
         *
         */
        static constexpr std::size_t pos_count = math::pow(2, dimension);

        /**
         * @brief
         *
         */
        constexpr z_curve() { _positions = create_array(); }

        /**
         * @brief
         *
         */
        constexpr z_curve(z_curve const&) = default;

        /**
         * @brief
         *
         */
        constexpr z_curve(z_curve&&) noexcept = default;

        /**
         * @brief
         *
         * @return z_curve&
         */
        constexpr inline auto operator=(z_curve const&) -> z_curve& = default;

        /**
         * @brief
         *
         * @return z_curve&
         */
        constexpr inline auto operator=(z_curve&&) noexcept -> z_curve& = default;

        /**
         * @brief Destroy the z curve object
         *
         */
        ~z_curve() = default;

      private:
        /// Array of positions type
        using position_array_t = std::array<position_type, pos_count>;

        /**
         * @brief Create an array of positions to initialize #_positions
         *
         * @return position_array_t
         */
        constexpr auto create_array() noexcept -> position_array_t
        {
            position_array_t positions;

            for(std::size_t i = 0; i < pos_count; ++i)
            {
                for(std::size_t j = 0, k = i; j < dimension; ++j, k >>= 1)
                {
                    positions[i][j] = k % 2;
                }
            }
            return positions;
        }

        /**
         * @brief Array to cache the positions corresponding to indices.
         *
         */
        position_array_t _positions;

      public:
        /**
         * @brief the position corresponding to an index
         *
         * @param idx The index of the point in the space filling curve
         * @return The position corresponding to the space filling curve index
         */
        [[nodiscard]] constexpr auto position(std::size_t idx) const noexcept -> position_type
        {
            return _positions[idx];
        }

        /**
         * @brief Index in the space filling curve of a boolean position
         *
         * @param p The position
         * @return The space filling curve index corresponding to the position
         */
        [[nodiscard]] constexpr auto index(const position_type& p) const noexcept -> std::size_t
        {
            std::size_t idx = 0;
            for(auto i: p)
            {
                idx <<= 1;
                idx += i;
            }
            return idx;
        }

        /**
         * @brief Index in the space filling curve of a real position relative to the center of the hypercube.
         *
         * @param p The position
         * @param center The center of the hypercube
         * @return The space filling curve index corresponding to the position
         */
        template<typename T>
        [[nodiscard]] constexpr auto index(const position_alias<T>& p,
                                           const position_alias<T>& center) const noexcept -> std::size_t
        {
            std::size_t idx = 0;
            for(std::size_t i = 0; i < dimension; ++i)
            {
                idx <<= 1;
                idx += p[i] > center[i];
            }
            return idx;
        }
    };
}   // namespace scalfmm::component
#endif
