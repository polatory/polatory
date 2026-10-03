// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/mapping.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_MAPPING_HPP
#define SCALFMM_INTERPOLATION_MAPPING_HPP

namespace scalfmm::interpolation
{
    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    class map_loc_glob
    {
      public:
        using value_type = T;
        using inner_type = typename T::value_type;

        /**
         * @brief Construct a new map loc glob object
         *
         * @param center
         * @param width
         */
        explicit map_loc_glob(const value_type& center, const value_type& width)
          : m_a(center - (width * half))
          , m_b(center + (width * half))
        {
        }

        /**
         * @brief
         *
         * @param loc_pos
         * @param glob_pos
         */
        inline void operator()(const value_type& loc_pos, value_type& glob_pos) const
        {
            glob_pos = ((m_a + m_b) * (half)) + (m_b - m_a) * loc_pos * half;
        }

        /**
         * @brief
         *
         * @param loc_pos
         * @return value_type
         */
        [[nodiscard]] inline auto operator()(const value_type& loc_pos) const -> value_type
        {
            return (((m_a + m_b) * half) + (m_b - m_a) * loc_pos * half);
        }

      private:
        const inner_type half{0.5};
        const value_type m_a;
        const value_type m_b;
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    class map_glob_loc
    {
      public:
        using value_type = T;
        using inner_type = typename T::value_type;

        /**
         * @brief Construct a new map glob loc object
         *
         * @param center
         * @param width
         */
        explicit map_glob_loc(const value_type& center, const value_type& width)
          : m_a(center - (width * half))
          , m_b(center + (width * half))
        {
        }

        /**
         * @brief
         *
         * @param glob_pos
         * @param loc_pos
         */
        inline void operator()(const value_type& glob_pos, value_type& loc_pos) const
        {
            loc_pos = (two * glob_pos - m_b - m_a) / (m_b - m_a);
        }

        /**
         * @brief
         *
         * @param glob_pos
         * @return value_type
         */
        [[nodiscard]] inline auto operator()(const value_type& glob_pos) const -> value_type
        {
            return (two * glob_pos - m_b - m_a) / (m_b - m_a);
        }

        /**
         * @brief
         *
         * @return value_type
         */
        inline auto jacobian() const -> value_type { return two / (m_b - m_a); }

      private:
        const inner_type two{2.};
        const inner_type half{0.5};
        const value_type m_a;
        const value_type m_b;
    };

}   // namespace scalfmm::interpolation
#endif   // SCALFMM_INTERPOLATION_MAPPING_HPP
