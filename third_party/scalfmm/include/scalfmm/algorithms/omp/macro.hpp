// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/omp/macro.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_MACRO_HPP
#define SCALFMM_ALGORITHMS_OMP_MACRO_HPP

#if _OPENMP >= 201811
#define commute mutexinoutset
#else
#define commute inout
#endif

#endif   // SCALFMM_ALGORITHMS_OMP_MACRO_HPP
