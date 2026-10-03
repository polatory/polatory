// --------------------------------
// See LICENCE file at project root
// File : scalfmm/matrix_kernels/laplace.hpp
// --------------------------------
#pragma once

#include "scalfmm/container/point.hpp"
#include "scalfmm/matrix_kernels/mk_common.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/utils/math.hpp"
#include "xtensor/core/xtensor_forward.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace scalfmm::matrix_kernels::laplace
{
    /**
    * @brief The one_over_r struct corresponds to the \f$1/r\f$ kernel.
    *
    *   The kernel is defined as \f$k(x,y): R^{km} -> R^{kn}\f$ with \f$ kn = km = 1\f$
    *                            \f$k(x,y) = | x - y |^{-1}\f$
    *
    * - The kernel is homogeneous \f$ k(ax,ay) = 1/a k(x,y)\f$.
    * - The scale factor is \f$1/a\f$.
    */
    struct one_over_r
    {
        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::symmetric};          // Specify the symmetry of the kernel.
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
        const std::string name() const { return std::string("one_over_r"); }

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
            return vector_type<decayed_type>{decayed_type(1.)};
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
                                                    std::index_sequence<Is...>) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using decayed_type = typename std::decay_t<ValueType1>;
            return matrix_type<decayed_type>{decayed_type(1.0) /
                                             xsimd::sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...))};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

    /**
    * @brief The like_mrhs simulates two FMMs with independent charges.
    *
    *   The kernel is defined as \f$k(x,y): R^2 -> R^2\f$
    *                            \f$ (q1,q2) --> (p1,p2)\f$
    *                           \f$k(x,y) = | x - y |^{-1} Id_{2x2}\f$
    *
    * - The kernel is homogeneous \f$k(ax,ay) = 1/a k(x,y)\f$.
    * - The scale factor is \f$(1/a, 1/a)\f$.
    */
    struct like_mrhs
    {
        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::symmetric};          // Specify the symmetry of the kernel.
        static constexpr std::size_t km{2};                               // The number of inputs.
        static constexpr std::size_t kn{2};                               // The number of outputs.

        // Mandatory types
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;   // Matrix type that is used in  the kernel.
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;   // Vector type that is used in the kernel.

        /**
	* @brief Returns the name of the kernel.
	*
	* @return A string representing the kernel's name.
	*/
        const std::string name() const { return std::string("like_mrhs multiple charges for 1/r"); }

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
            return vector_type<ValueType>({ValueType(1.), ValueType(1.)});
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
            auto tmp = ValueType(1.) / cell_width;
            return vector_type<ValueType>{tmp, tmp};
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

            auto val = decayed_type(1.0) / xsimd::sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...));
            return matrix_type<decayed_type>{val, decayed_type(0.), decayed_type(0.), val};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

    /**
    * @brief The grad_one_over_r matrix kernel computes the gradient of Laplace kernel.
    *
    *   The kernel is defined as \f$k(x,y) : R^{km} -> R^{kn}\f$ with \f$km = 1\f$ and \fkn = d\f
    *   with \f$d\f$ the space dimension.
    *
    *                            \f$k(x,y) = \grad | x - y |^{-1} = -(x-y) | x - y |^{-3}\f$
    *
    * - The kernel is homogeneous \f$k(ax,ay)= 1/a^2 k(x,y)\f$.
    * - The scale factor is \f$1/a^2\f$.
    */
    template<std::size_t Dimension = 3>
    struct grad_one_over_r
    {
        static constexpr std::size_t dimension = Dimension;

        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::non_symmetric};      // Specify the symmetry of the kernel.
        static constexpr std::size_t km{1};                               // The number of inputs of the kernel.
        static constexpr std::size_t kn{Dimension};                       // The number of outputs of the kernel.

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
        const std::string name() const { return std::string("grad_one_over_r<" + std::to_string(Dimension) + ">"); }

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
            auto tmp{math::pow(ValueType(1.) / cell_width, 2)};
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
            using decayed_type = typename std::decay_t<ValueType1>;

            decayed_type tmp =
              decayed_type(1.0) / xsimd::sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...));
            decayed_type r3{xsimd::pow(tmp, int(3))};
            return matrix_type<decayed_type>{r3 * (ys.at(Is) - xs.at(Is))...};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

    /**
    * @brief The val_grad_one_over_r struct is the kernel that computes the value and the gradient of the Laplace
    * kernel.
    *
    *     This is a specific kernel used to compute the  value of the kernel and its gradient
    *
    *     The kernel is defined as \f$k(x,y): R^{km} -> R^{kn}\f$ with \f$km = 1\f$ and \f$kn = d+1\f$
    *     with \f$d\f$ the space dimension.
    *
    *                     \f$k(x,y) = ( | x - y |^{-1}, \grad | x - y |^{-1} ) = (| x - y |^{-1}, -(x-y) | x - y |^{-3}\f)$
    *
    * - The kernel is homogeneous.
    * - The scale factor \f$(1/a, 1/a^2 ... 1/a^2)\f$.
    */
    template<std::size_t Dimension = 3>
    struct val_grad_one_over_r
    {
        static constexpr std::size_t dimension = Dimension;

        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::non_symmetric};      // Specify the symmetry of the kernel.
        static constexpr std::size_t km{1};                               // The number of inputs.
        static constexpr std::size_t kn{1 + Dimension};                   // The number of outputs.

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
        const std::string name() const { return std::string("val_grad_one_over_r<" + std::to_string(Dimension) + ">"); }

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
            vector_type<decayed_type> mc;
            mc.fill(decayed_type(-1.));
            mc.at(0) = decayed_type(1.0);
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
	* @brief Returns the scale factor of the kernel (no meaning here).
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
            using decayed_type = typename std::decay_t<ValueType>;
            auto tmp{decayed_type(1.) / cell_width};
            auto tmp_{math::pow(tmp, 2)};
            vector_type<decayed_type> sf;
            sf.fill(tmp_);
            sf.at(0) = tmp;
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
            using decayed_type = typename std::decay_t<ValueType1>;

            decayed_type tmp =
              decayed_type(1.0) / xsimd::sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...));
            decayed_type r3{xsimd::pow(tmp, int(3))};
            return matrix_type<decayed_type>{tmp, (r3 * (ys.at(Is) - xs.at(Is)))...};   //-r3 * (xs - ys);
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

    /**
    * @brief The ln_2d struct corresponds to the \f$log(r)\f$ kernel (only in 2D).
    *
    *   The kernel is defined as \f$k(x,y): R^{1} -> R^{kn}\f$ with\f$ kn = km = 1\f$.
    *
    *           \f$k(x,y) = \ln| x - y |\f$
    *
    * - The kernel is non homogeneous.
    */
    struct ln_2d
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
        const std::string name() const { return std::string("ln_2d"); }

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
	* @param x 2D point.
	* @param y 2D point.
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto operator()(container::point<ValueType1, 2> const& x,
                                             container::point<ValueType2, 2> const& y) const noexcept
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
	* @param x 2D point.
	* @param y 2D point.
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, 2> const& x,
                                           container::point<ValueType2, 2> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using decayed_type = typename std::decay_t<ValueType1>;

            return matrix_type<decayed_type>{xsimd::log(
              xsimd::sqrt((x.at(0) - y.at(0)) * (x.at(0) - y.at(0)) + (x.at(1) - y.at(1)) * (x.at(1) - y.at(1))))};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

    /**
    * \brief The grad ln_2d struct corresponds to the \f$ ld/dr log(r)\f$ kernel (only in 2D).
    *
    *   The kernel is defined as \f$k(x,y): R^{1} -> R^{2}\f$  with \f$kn = 2\$ and \f$km = 1\f$.
    *
    *           \f$k(x,y) = ( (x-y)/| x - y |)\f$
    *
    * - The kernel is homogeneous
    */
    struct grad_ln_2d
    {
        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::non_symmetric};      // Specify the symmetry of the kernel.
        static constexpr std::size_t km{1};                               // The number of inputs.
        static constexpr std::size_t kn{2};                               // The number of outputs.

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
        const std::string name() const { return std::string("grad_ln_2d"); }

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
            return vector_type<ValueType>({ValueType(-1.), ValueType(-1.)});
        }

        /**
	* @brief Overload of the `()` operator to evaluate the kernel at points \f$x\f$ and \f$y\f$.
	*
	* @tparam ValueType1 Type of the elements in the first point.
	* @tparam ValueType2 Type of the elements in the second point.
	*
	* @param x 2D point.
	* @param y 2D point.
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto operator()(container::point<ValueType1, 2> const& x,
                                             container::point<ValueType2, 2> const& y) const noexcept
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
	* @param x 2D point.
	* @param y 2D point.
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, 2> const& x,
                                           container::point<ValueType2, 2> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using decayed_type = typename std::decay_t<ValueType1>;

            auto diff = x - y;
            decayed_type tmp = decayed_type(1.0) / (diff.at(0) * diff.at(0) + diff.at(1) * diff.at(1));
            return matrix_type<decayed_type>{tmp * diff.at(0), tmp * diff.at(1)};
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
            return vector_type<ValueType>{ValueType(1. / cell_width), ValueType(1. / cell_width)};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

    /**
    * @brief The val_grad_ln_2d struct corresponds to the \f$ ld/dr log(r)\f$ kernel.
    *
    *   The kernel is defined as \f$k(x,y): R^{1} -> R^{kn}\f$ with \f$kn = 3\f$ and \f$km = 1\f$.
    *
    *           \f$k(x,y) = (ln(| x - y |^), (x-y)/| x - y |^2)\f$
    *
    * - The kernel is non homogeneous
    */
    struct val_grad_ln_2d
    {
        // Mandatory constants
        static constexpr auto homogeneity_tag{homogeneity::non_homogenous};   // Specify the homogeneity of the kernel.
        static constexpr auto symmetry_tag{symmetry::non_symmetric};          // Specify the symmetry of the kernel.
        static constexpr std::size_t km{1};                                   // Specify the number of inputs.
        static constexpr std::size_t kn{3};                                   // Specify the number of outputs.

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
        const std::string name() const { return std::string("val_grad_ln_2d"); }

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
            return vector_type<ValueType>({ValueType(1.), ValueType(-1.), ValueType(-1.)});
        }

        /**
	* @brief Overload of the `()` operator to evaluate the kernel at points \f$x\f$ and \f$y\f$.
	*
	* @tparam ValueType1 Type of the elements in the first point.
	* @tparam ValueType2 Type of the elements in the second point.
	*
	* @param x 2D point.
	* @param y 2D point.
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto operator()(container::point<ValueType1, 2> const& x,
                                             container::point<ValueType2, 2> const& y) const noexcept
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
	* @param x 2D point.
	* @param y 2D point.
	*
	* @return The matrix \f$K(x, y)\f$ in a vector (row-major storage).
	*/
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, 2> const& x,
                                           container::point<ValueType2, 2> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using decayed_type = typename std::decay_t<ValueType1>;
            auto diff = x - y;
            decayed_type norm = xsimd::sqrt(diff.at(0) * diff.at(0) + diff.at(1) * diff.at(1));
            decayed_type tmp = decayed_type(1.0) / norm;
            return matrix_type<decayed_type>{xsimd::log(norm), tmp * diff.at(0), tmp * diff.at(1)};
        }

        static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
    };

}   // namespace scalfmm::matrix_kernels::laplace
