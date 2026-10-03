// --------------------------------
// See LICENCE file at project root
// File : scalfmm/matrix_kernels/template_matrix_kernel.hpp
// --------------------------------
#pragma once

/**
 * @brief Represents a kernel with properties and methods for evaluation and configuration.
 *
 * The name of the struct corresponds to the kernel \f$ K(x,y) \f$.
 * The kernel is defined as \f$K(x,y): R^{km} -> R^{kn}\f$.
 *
 * - The kernel is homogeneous if \f$K(ax, ay) = a^p K(x, y)\f$ holds.
 * - The kernel is symmetric if it satisfies all symmetries (e.g., axes, \f$x = y\f$, etc.).
 */
struct name
{
    // Mandatory constants
    /**
     * @brief Specifies the homogeneity of the kernel.
     *
     * Can be `homogeneity::homogeneous` or `homogeneity::non_homogeneous`.
     */
    static constexpr auto homogeneity_tag{};

    /**
     * @brief Specifies the symmetry of the kernel.
     *
     * Can be `symmetry::symmetric` or `symmetry::non_symmetric`.
     */
    static constexpr auto symmetry_tag{};

    /**
     * @brief Number of inputs (\f$km\f$) for the kernel.
     */
    static constexpr std::size_t km{};

    /**
     * @brief Number of outputs (\f$kn\f$) for the kernel.
     */
    static constexpr std::size_t kn{};

    /**
     * @brief Criterion used to separate near and far fields.
     */
    static constexpr int separation_criterion{1};

    // Mandatory types
    /**
     * @brief Type representing a matrix used in the kernel.
     *
     * @tparam ValueType Type of the elements in the matrix.
     */
    template<typename ValueType>
    using matrix_type = std::array<ValueType, kn * km>;

    /**
     * @brief Type representing a vector used in the kernel.
     *
     * @tparam ValueType Type of the elements in the vector.
     */
    template<typename ValueType>
    using vector_type = std::array<ValueType, kn>;

    /**
     * @brief Returns the name of the kernel.
     *
     * @return A string representing the kernel's name.
     */
    const std::string name() const { return std::string("name"); }

    /**
     * @brief Returns the mutual coefficient of size \f$kn\f$ such that  \f$ f(x_i,x_j) = mutual_coefficient f(x_j,x_i)\f$.
     *
     * This coefficient is used during the direct pass when the kernel is applied
     * to compute interactions inside the leaf. Utilizing the kernel's symmetry
     * reduces the computational complexity from \f$N^2\f$ to \f$N^2 / 2\f$ (where \f$N\f$ is
     * the number of particles).
     *
     * @return A vector containing the mutual coefficients.
     */
    [[nodiscard]] inline constexpr auto mutual_coefficient() const { return vector_type<ValueType>{ValueType(1.)}; }

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
        using decayed_type = typename std::decay_t<ValueType1>;

        return matrix_type<decayed_type>{...};
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
        return vector_type<ValueType>{...};
    }
};
