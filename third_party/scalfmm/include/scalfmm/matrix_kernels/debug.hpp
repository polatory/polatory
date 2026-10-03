// --------------------------------
// See LICENCE file at project root
// File : scalfmm/matrix_kernels/debug.hpp
// --------------------------------
#pragma once

#include "scalfmm/meta/utils.hpp"

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
#include <xtensor/core/xmath.hpp>

namespace scalfmm::matrix_kernels::debug
{
    /**
    * @brief The one_over_r_non_homogenous struct corresponds to the \f$1/r\f$ kernel
    * (non-homogeneous case).
    *
    *   This matrix kernel is here to debug the non homogenous case.
    *
    *   The kernel is defined as \f$k(x,y): R^{km} -> R^{kn}\f$ with \f$ kn = km = 1\f$
    *                            \f$k(x,y) = | x - y |^{-1}\f$
    */
    struct one_over_r_non_homogenous
    {
        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::non_homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::symmetric};              // Specify the symmetry of the kernel.
        static constexpr std::size_t km{1};                                   // The number of inputs.
        static constexpr std::size_t kn{1};                                   // The number of outputs.

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
        const std::string name() const { return std::string("one_over_r_non_homogenous"); }

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
                                                    std::index_sequence<Is...> is) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using std::sqrt;
            using decayed_type = typename std::decay_t<ValueType1>;
            return matrix_type<decayed_type>{decayed_type(1.0) /
                                             sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...))};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

    /**
    * @brief The one_over_r_non_homogenous struct corresponds to the \f$1/r\f$ kernel
    * (non-symmetric case).
    *
    *   This matrix kernel is here to debug the non symmetric case.
    *
    *   The kernel is defined as \f$k(x,y): R^{km} -> R^{kn}\f$ with \f$ kn = km = 1\f$
    *                            \f$k(x,y) = | x - y |^{-1}\f$
    */
    struct one_over_r_non_symmetric
    {
        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::non_symmetric};      // Specify the symmetry of the kernel.
        static constexpr std::size_t km{1};                               // The number of inputs.
        static constexpr std::size_t kn{1};                               // The number of outputs.

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
        const std::string name() const { return std::string("one_over_r_non_symmetric"); }

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
            return vector_type<ValueType>{ValueType(1.) / cell_width};
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
                                                    std::index_sequence<Is...> is) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using std::sqrt;
            using decayed_type = typename std::decay_t<ValueType1>;
            return matrix_type<decayed_type>{decayed_type(1.0) /
                                             sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...))};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

    /**
    * @brief The one_over_r_non_homogenous struct corresponds to the \f$1/r\f$ kernel
    * (non-homogenous and non-symmetric case).
    *
    *   This matrix kernel is here to debug the non-homogeneous and non-symmetric case.
    *
    *   The kernel is defined as \f$k(x,y): R^{km} -> R^{kn}\f$ with \f$ kn = km = 1\f$
    *                            \f$k(x,y) = | x - y |^{-1}\f$
    */
    struct one_over_r_non_homogenous_non_symmetric
    {
        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::non_homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::non_symmetric};          // Specify the symmetry of the kernel.
        static constexpr std::size_t km{1};                                   // The number of inputs of the kernel.
        static constexpr std::size_t kn{1};                                   // The number of outputs of the kernel.

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
        const std::string name() const { return std::string("one_over_r_non_homogenous_non_symmetric"); }

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
            return vector_type<ValueType>{ValueType(1.) / cell_width};
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
        template<typename ValueType1, typename ValueType2, std::size_t Dimension, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType1, Dimension> const& xs,
                                                    container::point<ValueType2, Dimension> const& ys,
                                                    std::index_sequence<Is...> is) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using std::sqrt;
            using decayed_type = typename std::decay_t<ValueType1>;
            return matrix_type<decayed_type>{decayed_type(1.0) /
                                             sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...))};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

    /**
    * @brief The one_over_r struct corresponds to the \f$1/r\f$ kernel for x != y and 1 if x == y.
    *
    *   This matrix kernel enables the computation of self-interactions.
    *
    *   The kernel is defined as \f$k(x,y): R^{km} -> R^{kn}\f$ with \f$ kn = km = 1\f$
    *                            \f$k(x,y) = | x - y |^{-1}\f$ if $x != y$ and 1 otherwise.
    */
    struct one_over_r_modified : public one_over_r_non_homogenous
    {
        /**
	* @brief Returns the name of the kernel.
	*
	* @return A string representing the kernel's name.
	*/
        const std::string name() const { return std::string("one_over_r_modified"); }

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
            std::fill(std::begin(tmp), std::end(tmp), decayed_type(1.));
            return tmp;
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
            return one_over_r_non_homogenous::evaluate(x, y);
        }
    };

}   // namespace scalfmm::matrix_kernels::debug
