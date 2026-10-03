// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/omp/priorities.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_PRIORITIES_HPP
#define SCALFMM_ALGORITHMS_OMP_PRIORITIES_HPP

#ifdef __clang__
#define ACCESS_DEP(V, i) V[i]
#elif defined(__GNUC__)
#define ACCESS_DEP(V, i) V.at(i)
#else
#warning "MACRO FOR DEPENDENCIES : OTHER COMPILER TO CHECK (NOT GCC & CLANG) !"
#define ACCESS_DEP(V, i) V.at(i)
#endif

namespace scalfmm::algorithms::omp
{
    /**
    * @brief Priorities of the different passes for the task-based algorithms.
    */
    struct priorities
    {
        static constexpr int max = 10;
        static constexpr int p2m = 9;
        static constexpr int m2m = 8;
        static constexpr int m2l = 7;
        static constexpr int l2l = 6;
        static constexpr int p2p_big = 5;
        static constexpr int l2p = 3;
        static constexpr int p2p_small = 2;
    };

    /**
    * @brief Scales the priority values according to the level.
    */
    inline int scale_prio(int prio, std::size_t level) { return prio + (level - 2) * 3; }
}   // namespace scalfmm::algorithms::omp

#endif   // SCALFMM_ALGORITHMS_OMP_PRIORITIES_HPP
