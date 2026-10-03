// --------------------------------
// See LICENCE file at project root
// File : scalfmm/matrix_kernels/gaussian.hpp
// --------------------------------
#pragma once

#include <scalfmm/container/point.hpp>
#include <scalfmm/matrix_kernels/mk_common.hpp>

namespace scalfmm::matrix_kernels
{
    /**
    * @brief This structure corresponds to the Gaussian kernel.
    *
    *   The kernel is defined as \f$k(x,y): R^{km} -> R^{kn}\f$ with \f$ kn = km = 1\f$
     *                            \f$k(x,y) = exp(-| x - y |^2 / (\sigma))\f$
    *
    * - The kernel is not homogeneous.
    * - The kernel is symmetric (depends only on the norm).
    * - The kernel is smooth (and allows self-interactions to be evaluated).
    *
    *  default \f$ \sigma = 2\f$
    *  @GlobalValueType The value type of the coefficient \f$\sigma\f$.
    */
    template<typename GlobalValueType>
    struct gaussian
    {
        using value_type = GlobalValueType;

        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::non_homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::non_symmetric};          // Specify the symmetry of the kernel.

        static constexpr std::size_t km{1};   // The number of inputs of the kernel.
        static constexpr std::size_t kn{1};   // The number of outputs of the kernel.

        static constexpr bool is_smooth{true};   // Specify that the kernel is smooth.

        // Mandatory type
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;   // Matrix type that is used in the kernel.
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;   // Vector type that is used in the kernel.

        value_type m_coeff{value_type(1.)};     // parameter/coefficient of the kernel.
        value_type m_epsilon{value_type(0.)};   // epsilon value for the evaluation at point x (self-interaction).

        /**
         * @brief Set the value of the coefficient of the kernel.
         *
         * @param in_coeff value of the coefficient to be set.
         */
        void set_coeff(value_type in_coeff) { m_coeff = in_coeff; }

        /**
         * @brief Set the value of epsilon (to compute self-interaction).
         *
         * @param in_epsilon value of epsilon to be set.
         */
        void set_epsilon(value_type in_epsilon) { m_epsilon = in_epsilon; }

        /**
	* @brief Returns the name of the kernel.
	*
	* @return A string representing the kernel's name.
	*/
        const std::string name() const
        {
            return std::string("gaussian") + ", coeff = " + std::to_string(m_coeff) +
                   ", m_epsilon = " + std::to_string(m_epsilon);
        }

        /**
	* @brief Returns the mutual coefficient of size \f$kn\f$.
	*
	* This coefficient is used during the direct pass when the kernel is applied
	* to compute interactions inside the leaf. Utilizing the kernel's symmetry
	* reduces the computational complexity from \f$N^2\f$ to \f$N^2 / 2\f$ (where \f$N\f$ is
	* the number of particles).
	*
	* @return A vector containing the mutual coefficients.
	*/
        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            return vector_type<ValueType>{1.0};
        }

        /**
	 * @brief Overload of the `()` operator to evaluate the kernel at point \f$x\f$ (self-interaction).
	*
	* @tparam ValueType Type of the elements in the point.
	* @tparam Dimension The dimension of the point.
	*
	* @param x Multidimensional point (e.g., a 2D point).
	*
	* @return The matrix \f$K(x, x)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType, std::size_t Dimension>
        [[nodiscard]] inline auto operator()(container::point<ValueType, Dimension> const& x) const noexcept
          -> matrix_type<std::decay_t<ValueType>>
        {
            return evaluate(x);
        }

        /**
	 * @brief Evaluates the kernel at points \f$x\f$ (self-interaction).
	*
	* @tparam ValueType Type of the elements in the point.
	* @tparam Dimension The dimension of the point.
	*
	* @param x Multidimensional point (e.g., a 2D point).
	*
	* @return The matrix \f$K(x, x)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType, std::size_t Dimension>
        [[nodiscard]] inline auto evaluate(container::point<ValueType, Dimension> const& x) const noexcept
          -> matrix_type<std::decay_t<ValueType>>
        {
            using decayed_type = std::decay_t<ValueType>;
            matrix_type<decayed_type> tmp;
            std::fill(std::begin(tmp), std::end(tmp), decayed_type(1. + m_epsilon));
            return tmp;
        }

        /**
	* @brief Overload of the `()` operator to evaluate the kernel at points \f$x\f$ and \f$y\f$.
	*
	* @tparam ValueType1 Type of the elements in the first point.
	* @tparam ValueType2 Type of the elements in the second point.
	* @tparam Dimension The dimension of the points.
	*
	* @param x Multidimensional point (e.g., a 2D point).
	* @param y Multidimensional point (e.g., a 2D point).
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2, std::size_t Dimension>
        [[nodiscard]] inline auto operator()(container::point<ValueType1, Dimension> const& x,
                                             container::point<ValueType2, Dimension> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            return evaluate(x, y);
        }

        /**
	* @brief Evaluates the kernel at points \f$x\f$ and \f$y\f$.
	*
	* @tparam ValueType1 Type of the elements in the first point.
	* @tparam ValueType2 Type of the elements in the second point.
	* @tparam Dimension The dimension of the points.
	*
	* @param x Multidimensional point (e.g., a 2D point).
	* @param y Multidimensional point (e.g., a 2D point).
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2, std::size_t Dimension>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, Dimension> const& x,
                                           container::point<ValueType2, Dimension> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            return variadic_evaluate(x, y, std::make_index_sequence<Dimension>{});
        }

        /**
	* @brief Helper variadic function that evaluates the kernel at points \f$x\f$ and \f$y\f$.
	*
	* @tparam ValueType1 Type of the element in the first point.
	* @tparam ValueType2 Type of the element in the second point.
	* @tparam Dimension The dimension of the points.
	* @tparam Is A parameter pack representing the indices for accessing the coordinates of the points.
	*
	* @param xs The first multidimensional point (e.g., a 2D point).
	* @param ys The second multidimensional point (e.g., a 2D point).
	* @param std::index_sequence<Is...> An index sequence used to unpack and iterate over the dimensions.
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2, std::size_t Dimension, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType1, Dimension> const& xs,
                                                    container::point<ValueType2, Dimension> const& ys,
                                                    std::index_sequence<Is...>) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using decayed_type = std::decay_t<ValueType1>;
            auto l2 = decayed_type(-1.0) / (m_coeff);
            decayed_type r2 = (((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...);
            return matrix_type<decayed_type>{xsimd::exp(r2 * l2)};
        }

        static constexpr int separation_criterion{
          1};   // Criterion used to separate near and far field at all level exepct leaf
        static constexpr int leaf_separation_criterion{
          0};   // Criterion used to separate near and far field at leaf level
    };
}   // namespace scalfmm::matrix_kernels
