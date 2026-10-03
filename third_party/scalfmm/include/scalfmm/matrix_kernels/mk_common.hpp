// --------------------------------
// See LICENCE file at project root
// File : scalfmm/matrix_kernels/mk_common.hpp
// --------------------------------
#pragma once

namespace scalfmm::matrix_kernels
{
    /**
     * @brief specifies whether the kernel is homogeneous or not.
     */
    enum struct homogeneity
    {
        homogenous,
        non_homogenous
    };

    /**
     * @brief specifies whether the kernel is symmetric or not.
     */
    enum struct symmetry
    {
        symmetric,
        //   antisymmetric,
        non_symmetric
    };

    /**
     * @brief describes the convergence of the expansion at level 0
     *    in the periodic case
     */
    enum struct periodicity
    {
        absolutely,   //< absolutely convergent series
        condionally   //< conditionally convergent series
    };

    /**
     * @brief specifies if the kernel can be evaluated at x == y (compile-time)
     */
    struct smooth_tag
    {
    };

}   // namespace scalfmm::matrix_kernels
