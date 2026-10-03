// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/barycentric/barycentric_interpolator.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_BARYCENTRIC_BARYCENTRIC_STORAGE_HPP
#define SCALFMM_INTERPOLATION_BARYCENTRIC_BARYCENTRIC_STORAGE_HPP

#include "scalfmm/container/point.hpp"
#include "scalfmm/container/variadic_adaptor.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/interpolation/interpolator.hpp"
#include "scalfmm/interpolation/m2l_handler.hpp"
#include "scalfmm/interpolation/mapping.hpp"
#include "scalfmm/interpolation/permutations.hpp"
#include "scalfmm/interpolation/traits.hpp"
#include "scalfmm/matrix_kernels/mk_common.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/simd/memory.hpp"
#include "scalfmm/simd/utils.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"
#include "xflens/cxxblas/typedefs.h"
#include "xtensor-blas/xblas_utils.hpp"
#include "xtensor/core/xlayout.hpp"
#include "xtensor/core/xoperation.hpp"

#include <cpp_tools/colors/colorized.hpp>

#include "xsimd/xsimd.hpp"
#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xblas_config.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/misc/xcomplex.hpp"
#include "xtensor/io/xio.hpp"
#include "xtensor/misc/xmanipulation.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/core/xnoalias.hpp"
#include "xtensor/views/xslice.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xtensor_forward.hpp"
#include "xtensor/utils/xtensor_simd.hpp"
#include "xtensor/core/xvectorize.hpp"
#include "xtl/xclosure.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// This class implements barycentric Lagrange interpolation. By default, we use chebyshev points of the second kind, but
// chebyshev points of the first kind can be used instead simply by uncommenting the following macro. #define
// ENABLE_CHEBYSHEV_FIRST_KIND

// #define WEIGHTS_FIRST_KIND
#define WEIGHTS_UNIFORM

namespace scalfmm::interpolation
{
    template<typename ValueType, std::size_t Dimension, typename MatrixKernel, typename Settings>
    struct barycentric_interpolator
      : public impl::interpolator<barycentric_interpolator<ValueType, Dimension, MatrixKernel, Settings>>
      , public impl::m2l_handler<barycentric_interpolator<ValueType, Dimension, MatrixKernel, Settings>>
    {
        static_assert(options::support(options::_s(Settings{}),
                                       options::_s(options::barycentric_dense, options::barycentric_low_rank)),
                      "unsupported barycentric interpolator options!");

      public:
        using value_type = ValueType;
        static constexpr std::size_t dimension = Dimension;
        using size_type = std::size_t;

        using settings = Settings;
        using matrix_kernel_type = MatrixKernel;
        using self_type = barycentric_interpolator<value_type, dimension, matrix_kernel_type, settings>;
        using base_interpolator_type = impl::interpolator<self_type>;
        using base_m2l_handler_type = impl::m2l_handler<self_type>;

        using base_interpolator_type::base_interpolator_type;
        using base_m2l_handler_type::base_m2l_handler_type;

        /**
         * @brief Construct a new barycentric interpolator object
         *
         */
        barycentric_interpolator() = delete;

        /**
         * @brief Construct a new barycentric interpolator object
         *
         */
        barycentric_interpolator(barycentric_interpolator const&) = delete;

        /**
         * @brief Construct a new barycentric interpolator object
         *
         */
        barycentric_interpolator(barycentric_interpolator&&) noexcept = delete;

        /**
         * @brief
         *
         * @return barycentric_interpolator&
         */
        auto operator=(barycentric_interpolator const&) -> barycentric_interpolator& = delete;

        /**
         * @brief
         *
         * @return barycentric_interpolator&
         */
        auto operator=(barycentric_interpolator&&) noexcept -> barycentric_interpolator& = delete;

        /**
         * @brief Destroy the barycentric interpolator object
         *
         */
        ~barycentric_interpolator() = default;

        /**
         * @brief Constructor of the interpolator based on barycentric Lagrange interpolation.
         * It initializes the interpolation and transfer tools (M2L). Weights and roots (Chebyshev points of the first
         * or second kind) are precalculated. In addition, we also precompute d/dx S_i(x_j) for i,j in
         * {1,...,order-1}.
         *
         * @param far_field : matrix kernel used in the far field computation
         * @param order : order of the simulation
         * @param tree_height : height of the tree (= width of the simulation box)
         * @param root_cell_width : width of the root cell
         * @param cell_width_extension : parameter to extend the width of each cell (by default = 0)
         */
        barycentric_interpolator(matrix_kernel_type const& far_field, size_type order, size_type tree_height,
                                 value_type root_cell_width, value_type cell_width_extension = value_type(0.))

          : base_interpolator_type(order, tree_height, root_cell_width, cell_width_extension, true)
          , base_m2l_handler_type(far_field, roots_impl(), tree_height, root_cell_width, cell_width_extension, true)
          , m_roots_for_polynomials(roots_impl())
          , m_weights_for_polynomials(weights_impl())
          , m_derivative_of_roots(set_m_derivative_of_roots(order))
          , m_tolerance(std::numeric_limits<value_type>::min())
        {
            base_interpolator_type::initialize();
            base_m2l_handler_type::initialize(order, root_cell_width, tree_height);
        }

        /**
         * @brief Constructor of the interpolator based on barycentric Lagrange interpolation.
         * This constructor does not take a matrix kernel object as a parameter (it is initialized by default) and
         * calls the previous constructor.
         *
         * @param order : order of the simulation
         * @param tree_height : height of the tree (= width of the simulation box)
         * @param root_cell_width : width of the root cell
         * @param cell_width_extension : parameter to extend the width of each cell (by default = 0)
         */
        barycentric_interpolator(size_type order, size_type tree_height, value_type root_cell_width,
                                 value_type cell_width_extension = value_type(0.))
          : barycentric_interpolator(matrix_kernel_type{}, order, tree_height, root_cell_width, cell_width_extension)
        {
        }

        /**
         * @brief Roots in [-1,1]
         * If the Chebyshev roots of the first kind are used, the roots are computed as (with \f$\ell = order\f$)
         * \f$\bar x_n = \cos\left(\frac{\pi}{2}\frac{2n-1}{\ell}\right)\f$ for  \f$n=1,\dots,\ell\f$
         * If the Chebyshev roots of the second kind are used, the roots are computed as (with \f$\ell = order\f$)
         * \f$\bar x_n = \cos\left(\pi\frac{n-1}{\ell-1}\right)\f$ for \f$n=1,\dots,\ell\f$
         *
         * @return the roots
         */
        [[nodiscard]] inline auto roots_impl() const -> xt::xarray<value_type>
        {
#ifdef ENABLE_CHEBYSHEV_FIRST_KIND
            const auto order{this->order()};
            const value_type coeff = value_type(3.14159265358979323846264338327950) / (value_type(order));
            return xt::cos(coeff * (xt::arange(int(order - 1), int(-1), -1) + 0.5));
#else
            const auto order{this->order()};
            const value_type coeff = value_type(3.14159265358979323846264338327950) / (value_type(order - 1.));
            return xt::cos(coeff * (xt::arange(int(order - 1), int(-1), -1)));
#endif
        }

        /**
         * @brief Weights of the barycentric formula
         * If the Chebyshev roots of the first kind are used, the roots are computed as (with \f$\ell = order\f$)
         * \f$\bar \omega_n = (-1)^{n+1}\sin\left(\frac{pi}{2}\frac{2n-1}{\ell}\right)\f$ for \f$n=1,\dots,\ell\f$
         * If the Chebyshev roots of the second kind are used, the roots are computed as (with \f$\ell = order\f$)
         * \f$\bar \omega_n = (-1)^{n+1}\delta_n\f$ for \f$n=1,\dots,\ell\f$ with \f$\delta_n = 1/2\f$ if
         * \f$n\in\{1,\ell\}\f$ and $\f\delta_n = 1\f$ otherwise
         *
         * @return the weights
         */
        [[nodiscard]] inline auto weights_impl() const -> xt::xarray<value_type>
        {
#ifdef ENABLE_CHEBYSHEV_FIRST_KIND
            const auto order{this->order()};
            const value_type coeff = value_type(3.14159265358979323846264338327950) / (value_type(order));
            return xt::pow(-1, xt::arange(order)) * xt::sin(coeff * (xt::arange(int(order - 1), int(-1), -1) + 0.5));
#else
            const auto order{this->order()};
            xt::xarray<value_type> w = xt::pow(-1, xt::arange(order));
            w[0] *= value_type(0.5);
            w[order - 1] *= value_type(0.5);
            return w;
#endif
        }

        /**
         * @brief S_n(x) Lagrange function based on barycentric formula
         * Computed as \f$S_n(x) = \frac{\frac{w_i}{x - x_i}}{\sum_{k = 0}^{n}{\frac{w_k}{x - x_k}}} \f$ for
         * \f$n=1,\dots,\ell\f$ (with \f$\ell = order\f$)
         *
         * @tparam ComputationType
         * @param x : 1D evaluation point in [-1,1]
         * @param n : index of the Lagrange function
         * @return function value S_n(x)
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto polynomials_impl(ComputationType x, std::size_t n) const -> ComputationType
        {
            simd::compare(x);
            const auto order{this->order()};

            ComputationType flag{-1}, sum{0.};
            ComputationType root_at_o, weights_at_o, diff, final_result;

            for(unsigned int o = 0; o < order; ++o)
            {
                root_at_o = m_roots_for_polynomials[o];
                weights_at_o = m_weights_for_polynomials[o];

                diff = x - root_at_o;
                flag = xsimd::select((xsimd::abs(diff) < m_tolerance), ComputationType(o), flag);

                sum += weights_at_o / diff;
            }

            auto logical_mask = flag > ComputationType(-1);
            auto kronecker_delta = xsimd::select(flag == ComputationType(n), ComputationType(1.), ComputationType(0.));
            final_result =
              ComputationType(m_weights_for_polynomials[n]) / ((x - ComputationType(m_roots_for_polynomials[n])) * sum);

            return xsimd::select(logical_mask, kronecker_delta, final_result);
        }

        /**
         * @brief Compute S_n(x) for \f$n = 1,\dots,\ell\f$ (with \f$\ell = order\f$)
         *
         * @tparam VectorType
         * @tparam ComputationType
         * @tparam Dim
         * @param all_poly : vector containing all the evaluations of S_n(x) for \f$n = 1,\dots,\ell\f$ (with \f$\ell =
         * order\f$
         * @param x : 1D evaluation point in [-1,1]
         * @param order : order of the simulation
         */
        template<typename VectorType, typename ComputationType, std::size_t Dim>
        inline auto fill_all_polynomials_impl(VectorType& all_poly, container::point<ComputationType, Dim>& x,
                                              std::size_t order) const -> void
        {
            for(std::size_t d = 0; d < Dim; ++d)
            {
                for(std::size_t o = 0; o < order; ++o)
                {
                    all_poly[o][d] = this->polynomials_impl(x[d], o);
                }
            }
        }

        /**
         * @brief d/dx S_n(x) gradient of the Lagrange function based on barycentric formula
         * Computed as \f$d/dx S_n(x) = \frac{ \frac{ \omega_n }{ x - x_n } \sum_{j = 0}^{n}{ \frac{ \omega_j (x_j -
         * x_n) }{ ( x - x_j )^2 ( x - x_n ) } } }{\left( \sum_{j = 0}^{\ell}{ \frac{\omega_j}{ x - x_j } } \right)^2}
         * \f$ for \f$n=1,\dots,\ell\f$ (with \f$\ell = order\f$)
         *
         * @tparam ComputationType
         * @param x : 1D evaluation point in [-1,1]
         * @param n : index of the Lagrange function
         * @return function value d/dx S_n(x)
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto derivative_impl(ComputationType x, std::size_t n) const -> ComputationType
        {
            simd::compare(x);
            const auto order{this->order()};
            ComputationType num{0.}, denom{0.}, local_diff, final_result, local_term;
            ComputationType flag{-1.}, singularity{0.};

            for(unsigned int k = 0; k < order; ++k)
            {
                local_diff = ComputationType(x) - ComputationType(m_roots_for_polynomials[k]);

                auto logical_mask = xsimd::abs(local_diff) < m_tolerance;
                flag = xsimd::select(logical_mask, ComputationType(k), flag);
                singularity = xsimd::select(logical_mask, ComputationType(m_derivative_of_roots.at(k, n)), singularity);

                local_term = ComputationType(m_weights_for_polynomials[k]) / local_diff;
                num += local_term *
                       (ComputationType(m_roots_for_polynomials[k]) - ComputationType(m_roots_for_polynomials[n])) /
                       (local_diff * (ComputationType(x) - ComputationType(m_roots_for_polynomials[n])));
                denom += local_term;
            }

            final_result = (ComputationType(m_weights_for_polynomials[n]) * num) /
                           ((ComputationType(x) - ComputationType(m_roots_for_polynomials[n])) * denom * denom);

            return xsimd::select((flag > ComputationType(-1)), singularity, final_result);
        }

        /**
         * @brief Compute the quadrature weights (for the enhancement of low-rank approximation)
         *
         * @param order : order of the simulation
         * @return xtensor container filled with the quadrature weights
         */
        [[nodiscard]] inline auto generate_weights_impl(std::size_t order) const -> xt::xarray<value_type>
        {
#ifdef WEIGHTS_FIRST_KIND

            const value_type coeff = value_type(3.14159265358979323846264338327950) / (value_type(this->order()));
            xt::xarray<value_type> roots = xt::cos(coeff * (xt::arange(int(this->order() - 1), int(-1), -1) + 0.5));

            const auto weights_1d =
              xt::sqrt(value_type(3.14159265358979323846264338327950) / order * xt::sqrt(1 - (roots * roots)));
            xt::xarray<value_type> roots_weights(std::vector(dimension, order));

            auto generate_weights = [&roots_weights, &weights_1d](auto... is)
            { roots_weights.at(is...) = (... * weights_1d.at(is)); };

            std::array<int, dimension> starts{};
            std::array<int, dimension> stops{};
            starts.fill(0);
            stops.fill(order);
            meta::looper_range<dimension>{}(generate_weights, starts, stops);
            return roots_weights;
#endif

#ifdef WEIGHTS_UNIFORM
            return xt::xarray<value_type>(std::vector{math::pow(order, dimension)}, value_type(1.));
#endif
        }
        /**
             * @brief Compute he memory in bytes used by the interpolator
             *
             * @return the memory used by the interpolator
             */
        [[nodiscard]] inline auto memory_usage() const noexcept -> std::size_t const
        {
            return base_m2l_handler_type::memory_usage();
        }
        [[nodiscard]] inline auto name() const noexcept -> std::string const
        {
            settings s{};
            return "barycentric " + std::string(s.value());
        }

      private:
        /**
         * @brief Chebyshev polynomials of the first kind \f$ T_n(x) = \cos(n \acos(x)) \f$
         * * \acos(x))}{\sqrt{1-x^2}} \f$ are needed.
         *
         * @tparam ComputationType
         * @param[in] n : index of the Cheyshev polynomial (1st kind)
         * @param[in] x : 1D evaluation point in [-1,1]
         * @return function value T_n(x)
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto T(const unsigned int n, ComputationType x) const -> ComputationType
        {
            simd::compare(x);

            return xsimd::cos(ComputationType(n) * xsimd::acos(x));
        }

        /**
         * @brief Chebyshev polynomials of the second kind \f$ U_n(x) = \frac{\sin( (n+1)\acos(x) )}{\sqrt{1 - x^2}} \f$
         *
         *  For the derivation of the Chebyshev polynomials of first kind
         * \f$ \frac{\mathrm{d} T_n(x)}{\mathrm{d}x} = n U_{n-1}(x) \f$ the Chebyshev
         * polynomials of second kind \f$ U_{n-1}(x) = \frac{\sin(n \acos(x))}{\sqrt{1-x^2}} \f$
         * are needed.
         *
         * @param[in] n : index of the Cheyshev polynomial (2nd kind)
         * @param[in] x : 1D evaluation point in [-1,1]
         *
         * @return function value U_n(x)
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto U(const unsigned int n, ComputationType x) const -> ComputationType
        {
            simd::compare(x);

            return ComputationType((xsimd::sin((n + 1) * acos(x))) / xsimd::sqrt(1 - x * x));
        }

        /**
         * @brief Compute \f$\left{ d/dx S_j(x_i) | i,j = 1,\dots,\ell \right}\f$ with \f$\ell = order\f$
         *
         * @param order : order of the simulation
         * @return xtensor container filled with d/dx S_j(x_i)
         */
        inline auto set_m_derivative_of_roots(size_type order) -> xt::xarray<value_type>
        {
            xt::xarray<value_type> derivative_of_roots(xt::zeros<value_type>(std::vector<std::size_t>(2, order)));
            auto roots = this->roots_impl();
            auto weights = this->weights_impl();
            value_type local_w_i, local_root_i, local_coeff;

            for(unsigned int i = 0; i < order; ++i)
            {
                local_w_i = weights[i];
                local_root_i = roots[i];
                for(unsigned int j = 0; j < i; ++j)
                {
                    local_coeff = (weights[j] / local_w_i) / (local_root_i - roots[j]);
                    derivative_of_roots.at(i, j) = local_coeff;
                    derivative_of_roots.at(i, i) -= local_coeff;
                }
                for(unsigned int j = i + 1; j < order; ++j)
                {
                    local_coeff = (weights[j] / local_w_i) / (local_root_i - roots[j]);
                    derivative_of_roots.at(i, j) = local_coeff;
                    derivative_of_roots.at(i, i) -= local_coeff;
                }
            }
            return derivative_of_roots;
        }

        // private members

        /**
         * @brief
         *
         */
        xt::xarray<value_type> m_roots_for_polynomials{};

        /**
         * @brief
         *
         */
        xt::xarray<value_type> m_weights_for_polynomials{};

        /**
         * @brief
         *
         */
        value_type m_tolerance{};

        /**
         * @brief
         *
         */
        xt::xarray<value_type> m_derivative_of_roots{};
    };

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam FarFieldMatrixKernel
     * @tparam Settings
     */
    template<typename ValueType, std::size_t Dimension, typename FarFieldMatrixKernel, typename Settings>
    struct interpolator_traits<barycentric_interpolator<ValueType, Dimension, FarFieldMatrixKernel, Settings>>
    {
        using value_type = ValueType;
        using matrix_kernel_type = FarFieldMatrixKernel;

        /**
         * @brief
         *
         */
        static constexpr std::size_t dimension = Dimension;

        /**
         * @brief
         *
         */
        static constexpr bool enable_symmetries = (dimension < 4) ? true : false;

        /**
         * @brief
         *
         */
        static constexpr bool symmetry_support{
          (matrix_kernel_type::symmetry_tag == matrix_kernels::symmetry::symmetric) && enable_symmetries};

        /**
         * @brief
         *
         */
        static constexpr std::size_t kn = matrix_kernel_type::kn;

        /**
         * @brief
         *
         */
        static constexpr std::size_t km = matrix_kernel_type::km;

        using settings = Settings;
        using self_type = barycentric_interpolator<value_type, dimension, matrix_kernel_type, settings>;
        using base_interpolator_type = impl::interpolator<self_type>;
        using base_m2l_handler_type = impl::m2l_handler<self_type>;
        using storage_type = component::grid_storage<value_type, dimension, km, kn>;
        /// Temporary buffer to aggregate the multipole in order to perform matrix-matrix product
        using buffer_value_type = value_type;
        // the type of the element inside the buffer
        using buffer_inner_type = std::conditional_t<symmetry_support, xt::xarray<value_type>, meta::empty_inner>;
        // matrix of size 2  to store  the multipole and the local)
        using buffer_shape_type = std::conditional_t<symmetry_support, xt::xshape<2>, meta::empty_shape>;
        // matrix of size 2  to store  the multipole and the local

        using buffer_type =
          std::conditional_t<symmetry_support, xt::xtensor_fixed<buffer_inner_type, buffer_shape_type>, meta::empty>;

        using multipoles_container_type = typename storage_type::multipoles_container_type;
        using locals_container_type = typename storage_type::locals_container_type;
        using k_tensor_type = xt::xarray<value_type>;
    };
}   // namespace scalfmm::interpolation
#endif   // SCALFMM_INTERPOLATION_BARYCENTRIC_BARYCENTRIC_STORAGE_HPP
