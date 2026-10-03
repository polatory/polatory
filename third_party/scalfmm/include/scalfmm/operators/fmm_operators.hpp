// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/fmm_operators.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_FMM_OPERATORS_HPP
#define SCALFMM_OPERATORS_FMM_OPERATORS_HPP

#include <cstdlib>
#include <iostream>

#include <cpp_tools/colors/colorized.hpp>
#include <scalfmm/meta/traits.hpp>

namespace scalfmm::operators
{
    /**
     *  @brief The near_field_operator class
     *
     * This class concerns the near field operator for the fmm operators.
     *    It is used in direct pass in all algorithms.
     *    In general, it is enough to only have the matrix kernel member.
     *
     * The following example constructs the near field operator to compute the potential and the
     *     force associated to the potential 1/r
     *
     * \code
     *   using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r;
     *   using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
     * \endcode
     *
     * @tparam MatrixKernelType
     */
    template<typename MatrixKernelType>
    class near_field_operator
    {
      public:
        using matrix_kernel_type = MatrixKernelType;
        using self_type = near_field_operator<MatrixKernelType>;

        /**
         * @brief Construct a new near field operator object
         *
         */
        near_field_operator(near_field_operator const&) = delete;

        /**
         * @brief Construct a new near field operator object
         *
         */
        near_field_operator(near_field_operator&&) = delete;

        /**
         * @brief
         *
         * @return near_field_operator&
         */
        auto operator=(near_field_operator const&) -> near_field_operator& = default;

        /**
         * @brief
         *
         * @return near_field_operator&
         */
        auto operator=(near_field_operator&&) -> near_field_operator& = delete;

        /**
         * @brief near_field_operator constructor
         *  The separation_criterion is the matrix_kernel one and mutual is set to true
         *   and mutual is set to true
         *
         */
        near_field_operator()
          : m_matrix_kernel(matrix_kernel_type())
          , m_mutual(true)
        {
            m_separation_criterion = m_matrix_kernel.separation_criterion;
        }

        /**
         * @brief near_field_operator constructor
         *  The separation_criterion is the matrix_kernel one and mutual is set to true and
         *
         * @param[in] mutual  to specify if in p2p we use mutual interaction
         */
        near_field_operator(bool mutual)
          : m_matrix_kernel(matrix_kernel_type())
          , m_mutual(mutual)
        {
            m_separation_criterion = m_matrix_kernel.separation_criterion;
        }

        /**
         * @brief near_field_operator constructor
         *  The separation_criterion is the matrix_kernel one and mutual is set to true
         * and
         *
         * @param[in] matrix_kernel the matrix kernel used int the near field
         */
        near_field_operator(matrix_kernel_type const& matrix_kernel)
          : m_matrix_kernel(matrix_kernel)
          , m_separation_criterion(matrix_kernel.separation_criterion)
          , m_mutual(true)
        {
        }

        /**
         * @brief Construct a new near field operator object
         *
         * @param matrix_kernel the matrix kernel used in the near field.
         * @param mutual the boolean to specify whether the p2p is symmetric or not (mutual interactions).
         */
        near_field_operator(matrix_kernel_type const& matrix_kernel, const bool mutual)
          : m_matrix_kernel(matrix_kernel)
          , m_separation_criterion(matrix_kernel.separation_criterion)
          , m_mutual(mutual)
        {
        }

        /**
         * @brief
         *
         * @return matrix_kernel_type const&
         */
        auto matrix_kernel() const -> matrix_kernel_type const& { return m_matrix_kernel; }

        /**
         * @brief Get the matrix kernel 
         *
         * @return matrix_kernel_type&
         */
        auto matrix_kernel() -> matrix_kernel_type& { return m_matrix_kernel; }

        /**
         * @brief
         *
         * @return int
         */
        auto separation_criterion() const -> int { return m_separation_criterion; }

        /**
         * @brief
         *
         * @return int&
         */
        auto separation_criterion() -> int& { return m_separation_criterion; }

        /**
         * @brief The accessor to specify if we are using the symmetric P2P (mutual interactions)
         *
         * @return true
         * @return false
         */
        auto mutual() const -> bool { return m_mutual; }

        /**
         * @brief The accessor to specify if we are using the symmetric P2P (mutual interactions)
         *
         * @return true
         * @return false
         */
        auto mutual() -> bool& { return m_mutual; }

      private:
        matrix_kernel_type m_matrix_kernel;   ///< he matrix kernel used in the near field.

        int m_separation_criterion{
          matrix_kernel_type::separation_criterion};   ///< The separation criterion used in the near field.

        bool m_mutual{true};   ///< To specify if we use symmetric algorithm for the p2p (mutual interactions).
    };

    /**
     * @brief
     *
     * This class concerns the far field operator for the fmm operators.
     *
     *   The following example constructs the far field operator based on the uniform interpolation to compute
     *      the potential associated to the potential 1/r
     *
     * \code
     *     using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
     *     using interpolation_type = scalfmm::interpolation::uniform_interpolator<double, dimension,
     *     matrix_kernel_type>;
     *     using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;
     * \endcode
     *
     * @tparam ApproximationType
     * @tparam ComputeGradient
     */
    template<typename ApproximationType, bool ComputeGradient = false>
    class far_field_operator
    {
      public:
        using approximation_type = ApproximationType;
        static constexpr bool compute_gradient = ComputeGradient;
        ///<  precise the homogeneous type of the kernel
        static constexpr auto homogeneity_tag = approximation_type::homogeneity_tag;
        static constexpr auto is_smooth = meta::is_smooth_v<typename approximation_type::matrix_kernel_type>;

        /**
         * @brief
         *
         */
        far_field_operator() = delete;

        /**
         * @brief Construct a new far field operator object
         *
         */
        far_field_operator(far_field_operator const&) = delete;

        /**
         * @brief Construct a new far field operator object
         *
         */
        far_field_operator(far_field_operator&&) = delete;

        /**
         * @brief
         *
         * @return far_field_operator&
         */
        auto operator=(far_field_operator const&) -> far_field_operator& = default;

        /**
         * @brief
         *
         * @return far_field_operator&
         */
        auto operator=(far_field_operator&&) -> far_field_operator& = delete;

        /**
         * @brief Construct a new far field operator object
         *
         * @param approximation_method
         */
        far_field_operator(approximation_type const& approximation_method)
          : m_approximation(approximation_method)
        {
        }

        /**
         * @brief
         *
         * @return approximation_type const&
         */
        auto approximation() const -> approximation_type const& { return m_approximation; }

        /**
         * @brief
         *
         * @return int
         */
        auto separation_criterion() const -> int { return m_separation_criterion; }

        /**
         * @brief
         *
         * @return int&
         */
        auto separation_criterion() -> int& { return m_separation_criterion; }

      private:
        // Here, be careful with this reference with the lifetime of matrix_kernel.

        approximation_type const& m_approximation;   ///<  the approximation method used in the near field.

        int m_separation_criterion{approximation_type::separation_criterion};   ///<  the separation criterion.
    };

    /**
     * @brief
     *
     *  The following example constructs the near field operator to compute the potential associated to the potential
     *  1/r
     *
     * \code
     *   // Near field
     *   using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r;
     *   using nar_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
     *   // Far field is the uniform interpolation
     *   using far_field_type = scalfmm::interpolation::uniform_interpolator<double, dimension, matrix_kernel_type>;
     *    using fmm_operator = scalfmm::operators::fmm_operators<near_field_type, far_field_type>  ;
     * \endcode
     *
     * The following example constructs the near-field operator to calculate the potential and
     *  the force associated with the potential 1/r.  The far field considers the
     *  one_over_r matrix kernel and calculates the force by deriving the interpolation polynomial
     * of the potential.
     *
     * \code
     *   using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r;
     *   using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
     *   //
     *   using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
     *   // Far field is the uniform interpolation
     *   using far_field_type = scalfmm::interpolation::uniform_interpolator<double, dimension,
     *   far_matrix_kernel_type>; using fmm_operator = scalfmm::operators::fmm_operators<near_field_type,
     *   far_field_type>  ;
     * \endcode
     *
     * @tparam NearFieldType
     * @tparam FarFieldType
     */
    template<typename NearFieldType, typename FarFieldType>
    class fmm_operators
    {
      public:
        using near_field_type = NearFieldType;
        using far_field_type = FarFieldType;

        /**
         * @brief
         *
         */
        fmm_operators() = delete;

        /**
         * @brief Construct a new fmm operators object
         *
         */
        fmm_operators(fmm_operators&&) = delete;

        /**
         * @brief Construct a new fmm operators object
         *
         * @param other
         */
        fmm_operators(fmm_operators const& other) = delete;

        /**
         * @brief
         *
         * @return fmm_operators&
         */
        auto operator=(fmm_operators const&) -> fmm_operators& = delete;

        /**
         * @brief
         *
         * @return fmm_operators&
         */
        auto operator=(fmm_operators&&) -> fmm_operators& = delete;

        /**
         * @brief Construct a new fmm operators object
         *
         * @param near_field
         * @param far_field
         */
        fmm_operators(near_field_type const& near_field, far_field_type const& far_field)
          : m_near_field(near_field)
          , m_far_field(far_field)
        {
            if(m_near_field.separation_criterion() != m_far_field.separation_criterion())
            {
                std::cerr << "The separation criteria is not the same in the near "
                             "and far fields !!"
                          << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        /**
         * @brief  get the near field of the FMM operator
         *
         * @return near_field_type const&
         */
        auto near_field() const -> near_field_type const& { return m_near_field; }

        /**
         * @brief get the far field of the FMM operator
         *
         * @return far_field_type const&
         */
        auto far_field() const -> far_field_type const& { return m_far_field; }
        /// @brief  display the settings on a stream
        /// @param os the stram to display the settings
        inline auto settings(std::ostream& os = std::cout) const noexcept -> void
        {
            std::cout << cpp_tools::colors::blue << "Fmm operators with kernels: " << std::endl
                      << "    near field\n"
                      << "       kernel: " << m_near_field.matrix_kernel().name() << "\n"
                      << "               mutual " << std::boolalpha << m_near_field.mutual() << std::endl
                      << "    far field\n"
                      << "       approximation: " << m_far_field.approximation().name() << std::endl
                      << std::flush << "       kernel: " << m_far_field.approximation().matrix_kernel().name()
                      << std::endl
                      << std::flush;
        }

      private:
        near_field_type const& m_near_field;   ///< the near field used in the fmm operator.

        far_field_type const& m_far_field;   ///< the far field used in the fmm operator.
    };

}   // namespace scalfmm::operators

#endif
