// --------------------------------
// See LICENCE file at project root
// File : scalfmm/matrix_kernels/scalar_kernels.hpp
// --------------------------------
#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <scalfmm/container/point.hpp>
#include <scalfmm/matrix_kernels/mk_common.hpp>
#include <scalfmm/meta/utils.hpp>
#include <scalfmm/utils/math.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <xtensor/containers/xfixed.hpp>
#include <xtensor/core/xmath.hpp>

#include "scalfmm/meta/utils.hpp"

namespace scalfmm::matrix_kernels::one_d
{
    /**
    * @brief The one_over_x struct corresponds to the \f$1/x\f$ 1-d kernel.
    *
    *   The kernel is defined as \f$k(x,y): R^{km} -> R^{kn}\f$ with \f$ kn = km = 1\f$
    *                            \f$k(x,y) = \frac{1}{x - y }\f$
    *
    * - The kernel is homogeneous \f$ k(ax,ay) = 1/a k(x,y)\f$. The scale factor is \f$1/a\f$.
    * - The kernel is not symmetric
    * 
    *  https://doi.org/10.1137/S0036142997329232
    */
    struct one_over_x
    {
        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::non_symmetric};      // Specify the symmetry of the kernel.
        static constexpr std::size_t km{1};                               // The number of inputs of the kernel.
        static constexpr std::size_t kn{1};                               // The number of outputs of the kernel.

        // Mandatory types
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;   // Matrix type that is used in the kernel.
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;   // Vector type that is used in the kernel.

        /**
   * @brief Returns the name of the kernel.
   *
   * @return A string representing the kernel's name.
   */
        const std::string name() const { return std::string("one_over_x (1-d)"); }

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
            using decayed_type = typename std::decay_t<ValueType>;
            return vector_type<decayed_type>{decayed_type(-1.)};
        }

        /**
   * @brief Overload of the `()` operator to evaluate the kernel at points \f$x\f$ and \f$y\f$.
   *
   * @tparam ValueType1 Type of the elements in the first point.
   * @tparam ValueType2 Type of the elements in the second point.
   *
   * @param x Multidimensional point (e.g., a 2D point).
   * @param y Multidimensional point (e.g., a 2D point).
   *
   * @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
   */
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto operator()(container::point<ValueType1, 1> const& x,
                                             container::point<ValueType2, 1> const& y) const noexcept
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
   *
   * @param x Multidimensional point (e.g., a 2D point).
   * @param y Multidimensional point (e.g., a 2D point).
   *
   * @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
   */
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, 1> const& x,
                                           container::point<ValueType2, 1> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using decayed_type = typename std::decay_t<ValueType1>;
            return matrix_type<decayed_type>{decayed_type(1.0) / (x[0] - y[0])};
        }

        /**
   * @brief Returns the scale factor of the kernel.
   *
   * This method is used only if the kernel is homogeneous.
   *
   * @tparam ValueType Type of the cell width.
   *
   * @param cell_width The width of the cell.
   *
   * @return A vector representing the scale factor.
   */
        template<typename ValueType>
        [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        {
            return vector_type<ValueType>{ValueType(1.) / cell_width};
        }
        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };
}   // namespace scalfmm::matrix_kernels::one_d

namespace scalfmm::matrix_kernels::others
{

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
    * @brief The one_over_r2 struct corresponds to the \f$1/r^2\f$ kernel.
    *
    *   The kernel is defined as \f$k(x,y): R^{km} -> R^{kn}\f$ with \f$ kn = km = 1\f$
    *                            \f$k(x,y) = | x - y |^{-2}\f$
    *
    * - The kernel is homogeneous \f$ k(ax,ay) = 1/a^2 k(x,y)\f$.
    * - The scale factor is \f$1/a^2\f$.
    */
    struct one_over_r2
    {
        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::symmetric};          // Specify the symmetry of the kernel.
        static constexpr std::size_t km{1};                               // The nmuber of inputs of the kernel.
        static constexpr std::size_t kn{1};                               // The number of outputs of the kernels.

        // Mandatory types
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;   // Matrix type that is used in the kernel.
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;   // Vector type that is used in the kernel.

        /**
	* @brief Returns the name of the kernel.
	*
	* @return A string representing the kernel's name.
	*/
        const std::string name() const { return std::string("one_over_r2"); }

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
            return vector_type<ValueType>{ValueType(1.)};
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
	* @brief Returns the scale factor of the kernel.
	*
	* This method is used only if the kernel is homogeneous.
	*
	* @tparam ValueType Type of the cell width.
	*
	* @param cell_width The width of the cell.
	*
	* @return A vector representing the scale factor.
	*/
        template<typename ValueType>
        [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        {
            return vector_type<ValueType>({ValueType(1.) / (cell_width * cell_width)});
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
            using decayed_type = typename std::decay_t<ValueType1>;

            return matrix_type<decayed_type>{decayed_type(1.0) /
                                             (((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...)};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

    /**
    * @brief The grad_one_over_r2 matrix kernel computes the gradient of the one_over_r2 kernel.
    *
    *   The kernel is defined as \f$k(x,y) : R^{km} -> R^{kn}\f$ with \f$km = 1\f$ and \fkn = d\f
    *   with \f$d\f$ the space dimension.
    *
    *                            \f$k(x,y) = \grad | x - y |^{-2}\f$
    *
    * - The kernel is homogeneous \f$k(ax,ay)= 1/a^3 k(x,y)\f$.
    * - The scale factor is \f$1/a^3\f$.
    *
    * @tparam Dimension The dimension of the space.
    */
    template<std::size_t Dimension = 3>
    struct grad_one_over_r2
    {
        static constexpr std::size_t dimension = Dimension;

        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::symmetric};          // Specify the symmetry of the kernel.
        static constexpr std::size_t km{1};                               // The number of inputs of the kernel.
        static constexpr std::size_t kn{Dimension};                       // The number of outputs of the kernels.

        // Mandatory types
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;   // Matrix type that is used in the kernel.
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;   // Vector type that is used in the kernel.

        /**
	* @brief Returns the name of the kernel.
	*
	* @return A string representing the kernel's name.
	*/
        const std::string name() const { return std::string("grad_one_over_r2<" + std::to_string(Dimension) + ">"); }

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
            vector_type<ValueType> mc;
            std::fill(std::begin(mc), std::end(mc), ValueType(-1.));
            return mc;
        }

        /**
	* @brief Overload of the `()` operator to evaluate the kernel at points \f$x\f$ and \f$y\f$.
	*
	* @tparam ValueType1 Type of the elements in the first point.
	* @tparam ValueType2 Type of the elements in the second point.
	*
	* @param x Multidimensional point (e.g., a 2D point).
	* @param y Multidimensional point (e.g., a 2D point).
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto operator()(container::point<ValueType1, dimension> const& x,
                                             container::point<ValueType2, dimension> const& y) const noexcept
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
	*
	* @param x Multidimensional point (e.g., a 2D point).
	* @param y Multidimensional point (e.g., a 2D point).
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, dimension> const& x,
                                           container::point<ValueType2, dimension> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            return variadic_evaluate(x, y, std::make_index_sequence<dimension>{});
        }

        /**
	* @brief Returns the scale factor of the kernel.
	*
	* This method is used only if the kernel is homogeneous.
	*
	* @tparam ValueType Type of the cell width.
	*
	* @param cell_width The width of the cell.
	*
	* @return A vector representing the scale factor.
	*/
        template<typename ValueType>
        [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        {
            auto tmp{math::pow(ValueType(1.) / cell_width, 3)};
            vector_type<ValueType> sf;
            std::fill(std::begin(sf), std::end(sf), tmp);
            return sf;
        }

        /**
	* @brief Helper variadic function that evaluates the kernel at points \f$x\f$ and \f$y\f$.
	*
	* @tparam ValueType1 Type of the element in the first point.
	* @tparam ValueType2 Type of the element in the second point.
	* @tparam Is A parameter pack representing the indices for accessing the coordinates of the points.
	*
	* @param xs The first multidimensional point (e.g., a 2D point).
	* @param ys The second multidimensional point (e.g., a 2D point).
	* @param std::index_sequence<Is...> An index sequence used to unpack and iterate over the dimensions.
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType1, dimension> const& xs,
                                                    container::point<ValueType2, dimension> const& ys,
                                                    std::index_sequence<Is...>) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using std::sqrt;
            using decayed_type = typename std::decay_t<ValueType1>;

            decayed_type tmp = decayed_type(1.0) / (((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...);
            decayed_type r4{pow(tmp, 2)};
            return matrix_type<decayed_type>{(decayed_type(-2.0) * r4 * (xs.at(Is) - ys.at(Is)))...};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

}   // namespace scalfmm::matrix_kernels::others
