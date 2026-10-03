// --------------------------------
// See LICENCE file at project root
// File : scalfmm/lists/policies.hpp
// --------------------------------
#ifndef SCALFMM_LISTS_POLICIES_HPP
#define SCALFMM_LISTS_POLICIES_HPP

namespace scalfmm::list
{
    /**
     * @brief The different policies available for algorithms (lists construction)
     *
     */
    struct policies
    {
        static constexpr int sequential = 1;   ///< for sequential algorithm
        static constexpr int omp = 2;          ///< for OpenMP algorithm
        static constexpr int starpu = 3;       ///< for StarPU runtime algorithm (not yet implemented)
    };
}   // namespace scalfmm::list

#endif   // SCALFMM_LISTS_POLICIES_HPP
