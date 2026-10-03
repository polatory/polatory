// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/chebyshev/chebyshev_interpolator.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_CHEBYSHEV_CHEBYSHEV_INTERPOLATOR_HPP
#define SCALFMM_INTERPOLATION_CHEBYSHEV_CHEBYSHEV_INTERPOLATOR_HPP

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

namespace scalfmm::interpolation
{
    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam MatrixKernel
     * @tparam Settings
     */
    template<typename ValueType, std::size_t Dimension, typename MatrixKernel, typename Settings>
    struct chebyshev_interpolator
      : public impl::interpolator<chebyshev_interpolator<ValueType, Dimension, MatrixKernel, Settings>>
      , public impl::m2l_handler<chebyshev_interpolator<ValueType, Dimension, MatrixKernel, Settings>>
    {
        static_assert(options::support(options::_s(Settings{}),
                                       options::_s(options::chebyshev_dense, options::chebyshev_low_rank)),
                      "unsupported chebyshev interpolator options!");

      public:
        using value_type = ValueType;
        static constexpr std::size_t dimension = Dimension;
        using size_type = std::size_t;

        using settings = Settings;
        using matrix_kernel_type = MatrixKernel;
        using self_type = chebyshev_interpolator<value_type, dimension, matrix_kernel_type, settings>;
        using base_interpolator_type = impl::interpolator<self_type>;
        using base_m2l_handler_type = impl::m2l_handler<self_type>;

        using base_interpolator_type::base_interpolator_type;
        using base_m2l_handler_type::base_m2l_handler_type;

        /**
         * @brief Construct a new chebyshev interpolator object
         *
         */
        chebyshev_interpolator() = delete;

        /**
         * @brief Construct a new chebyshev interpolator object
         *
         */
        chebyshev_interpolator(chebyshev_interpolator const&) = delete;

        /**
         * @brief Construct a new chebyshev interpolator object
         *
         */
        chebyshev_interpolator(chebyshev_interpolator&&) noexcept = delete;

        /**
         * @brief
         *
         * @return chebyshev_interpolator&
         */
        auto operator=(chebyshev_interpolator const&) -> chebyshev_interpolator& = delete;

        /**
         * @brief
         *
         * @return chebyshev_interpolator&
         */
        auto operator=(chebyshev_interpolator&&) noexcept -> chebyshev_interpolator& = delete;

        /**
         * @brief Destroy the chebyshev interpolator object
         *
         */
        ~chebyshev_interpolator() = default;

        /**
         * @brief Construct a new chebyshev interpolator object
         *
         * @param far_field
         * @param order
         * @param tree_height
         * @param root_cell_width
         * @param cell_width_extension
         */
        chebyshev_interpolator(matrix_kernel_type const& far_field, size_type order, size_type tree_height,
                               value_type root_cell_width, value_type cell_width_extension = value_type(0.))
          : base_interpolator_type(order, tree_height, root_cell_width, cell_width_extension, true)
          , base_m2l_handler_type(far_field, roots_impl(), tree_height, root_cell_width, cell_width_extension, true)
          , m_T_of_roots(set_m_T_of_roots(order))
        {
            base_interpolator_type::initialize();
            base_m2l_handler_type::initialize(order, root_cell_width, tree_height);
        }

        /**
         * @brief Construct a new chebyshev interpolator object
         *
         * @param order
         * @param tree_height
         * @param root_cell_width
         * @param cell_width_extension
         */
        chebyshev_interpolator(size_type order, size_type tree_height, value_type root_cell_width,
                               value_type cell_width_extension = value_type(0.))
          : chebyshev_interpolator(matrix_kernel_type{}, order, tree_height, root_cell_width, cell_width_extension)
        {
        }

        /**
         * @brief roots_impl Chebyshev roots in [-1,1]
         *
         * computed as \f$\bar x_n = \cos\left(\frac{\pi}{2}\frac{2n-1}{\ell}\right)\f$ for  \f$n=1,\dots,\ell\f$
         * @param order
         * @return the Chebyshev roots
         */
        [[nodiscard]] inline auto roots_impl() const -> xt::xarray<value_type>
        {
            const value_type coeff = value_type(3.14159265358979323846264338327950) / (value_type(this->order()));
            return xt::cos(coeff * (xt::arange(int(this->order() - 1), int(-1), -1) + 0.5));
        }

        /**
         * @brief
         *
         * S_k(x) Lagrange function based on Chebyshev polynomials of first kind  \f$
         *
         * for \f$p = order+1 \f$ the interpolator S associated to point n
         *  \f$ S_n(x) = \frac{1}{p} + \frac{2}{p} \sum_{k=1}{p-1}{ T_k(x)T_k(x_n)}\f$
         *  then we have  \f$ S_n(x_m) =\delta_{n,m} \f$
         * We use the recurrence relation to construct \f$ T_k(x) \f$
         *
         * @param[in] x coordinate in [-1,1]
         * @param[in] n index
         * @return function value
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto polynomials_impl(ComputationType x, std::size_t n) const -> ComputationType
        {
            simd::compare(x);
            const auto order{this->order()};
            const auto coeff{value_type(2.0 / order)};
            ComputationType L(0.5);
            ComputationType Pn(x), Pnm1(1.), Pnp1(0.);
            ComputationType two_x(2.0 * x);
            //
            const auto t_of_r_ptr = m_T_of_roots.data() + n * order;
            //            t_of_r_ptr += n*order;
            L += x * ComputationType(t_of_r_ptr[1]);
            for(unsigned int o = 2; o < order; ++o)
            {
                Pnp1 = two_x * Pn - Pnm1;
                L += Pnp1 * ComputationType(t_of_r_ptr[o]);
                Pnm1 = Pn;
                Pn = Pnp1;
            }
            return coeff * L;
        }

        /**
         * @brief
         *
         * @tparam VectorType
         * @tparam ComputationType
         * @tparam Dim
         * @param all_poly
         * @param x
         * @param order
         * @return auto
         */
        template<typename VectorType, typename ComputationType, std::size_t Dim>
        inline auto fill_all_polynomials_impl(VectorType& all_poly, container::point<ComputationType, Dim> x,
                                              std::size_t order) const
        {
            //  const auto order{this->order()};
            // using point_type = typename container::point<ComputationType, Dim>;
            // std::vector<point_type, XTENSOR_DEFAULT_ALLOCATOR(point_type)> all_poly(n);
#ifdef OLD
            auto call_polynomials = [](auto x, std::size_t n) { return polynomials(x, n); };

            for(std::size_t o = 0; o < order; ++o)
            {
                all_poly[o] = simd::apply_f<simd_position_type::dimension>(call_polynomials, local_position, o);
            }
#else
            std::vector<ComputationType> poly_of_x(order);
            const ComputationType coeff{value_type(2.0 / order)};
            for(std::size_t d = 0; d < Dim; ++d)
            {
                auto& coord = x[d];
                simd::compare(coord);
                ComputationType two_x(2.0 * coord);
                poly_of_x[0] = ComputationType(1.);
                poly_of_x[1] = ComputationType(coord);
                //
                // build all Chebyshev polynomials in x
                for(unsigned int o = 2; o < order; ++o)
                {
                    poly_of_x[o] = two_x * poly_of_x[o - 1] - poly_of_x[o - 2];
                }

                for(std::size_t n = 0; n < order; ++n)
                {
                    const auto t_of_r_ptr = m_T_of_roots.data() + n * order;
                    ComputationType L(0.5);
                    for(unsigned int o = 1; o < order; ++o)
                    {
                        L += poly_of_x[o] * ComputationType(t_of_r_ptr[o]);
                    }
                    all_poly[n][d] = coeff * L;
                }
            }
#endif
            //   return std::move(all_poly);
        }

        /**
         * @brief
         *
         *  d/dx S_n(x) gradient of the Lagrange function based on Chebyshev
         * polynomials of first kind  \f$
         *
         * for \f$p = order+1 \f$ the interpolator S associated to point n
         *  \f$ d/dx  S_n(x) = \frac{2}{p} \sum_{k=1}{p-1}{ T_k(x_n) d/dx T_k(x)}\f$
         * then
         * \f$ d/dx  S_n(x) = \frac{2}{p} \sum_{k=1}{p}{ T_k(x_n) k U_{k-1}(x)}\f$
         *         *
         * @param[in] x coordinate in [-1,1]
         * @param[in] n index
         *
         * @return function value
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto derivative_impl(ComputationType x, std::size_t n) const -> ComputationType
        {
            using value_type = ComputationType;
            simd::compare(x);
            auto order{this->order()};
            //
            auto coeff{value_type(2.0) / value_type(order)};

            // U(0, x) = 1
            auto t_of_r_ptr = m_T_of_roots.data();
            t_of_r_ptr += n * order;
            value_type L{t_of_r_ptr[1]};
            value_type Un(2. * x), Unm1(1.), Unp1(0.);
            value_type two_x(2.0 * x);

            //{value_type(0)};
            L += value_type(2) * Un * value_type(t_of_r_ptr[2]);

            for(unsigned int k = 3; k < order; ++k)
            {
                Unp1 = two_x * Un - Unm1;
                L += value_type(k) * Unp1 * value_type(t_of_r_ptr[k]);
                Unm1 = Un;
                Un = Unp1;
            }
            return coeff * L;
        }

        /**
         * @brief
         *
         * @tparam ComputationType
         * @param x
         * @param n
         * @return ComputationType
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto derivative_impl_old(ComputationType x, std::size_t n) const -> ComputationType
        {
            using value_type = ComputationType;
            simd::compare(x);
            auto order{this->order()};
            //
            auto coeff{value_type(2.0) / value_type(order)};

            // U(0, x) = 1
            value_type L{m_T_of_roots.at(1, n)};
            //{value_type(0)};
            for(unsigned int k = 2; k < order; ++k)
            {
                L += k * U(k - 1, x) * m_T_of_roots.at(k, n);
            }
            return coeff * L;
        }

        /**
         * @brief
         *
         * Sets the roots of the Chebyshev quadrature weights defined as \f$w_i =
         * \frac{\pi}{\ell}\sqrt{1-\bar x_i^2}\f$ with the Chebyshev roots \f$\bar
         * x\f$.
         *
         * @param order
         * @return xt::xarray<value_type>
         */
        [[nodiscard]] inline auto generate_weights_impl(std::size_t order) const -> xt::xarray<value_type>
        {
            auto roots = this->roots_impl();
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
        }
        /**
             * @brief ompute he memory in bytes used by the interpolator
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
            return "chebyshev " + std::string(s.value());
        }

      private:
        /**
         * @brief Chebyshev polynomials of first kind \f$ T_n(x) = \cos(n \arccos(x)) \f$
         * * \arccos(x))}{\sqrt{1-x^2}} \f$ are needed.
         *
         * @tparam ComputationType
         * @param[in] n index
         * @param[in] x coordinate in [-1,1]
         * @return function value  T_n(x)
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto T(const unsigned int n, ComputationType x) const -> ComputationType
        {
            simd::compare(x);

            return xsimd::cos(ComputationType(n) * xsimd::acos(x));
        }

        /**
         * @brief U the derivation of the Chebyshev polynomials of first kind
         *
         *  For the derivation of the Chebyshev polynomials of first kind
         * \f$ \frac{\mathrm{d} T_n(x)}{\mathrm{d}x} = n U_{n-1}(x) \f$ the Chebyshev
         * polynomials of second kind \f$ U_{n-1}(x) = \frac{\sin(n \arccos(x))}{\sqrt{1-x^2}} \f$
         * are needed.
         *
         * @tparam ComputationType
         * @param[in] n index
         * @param[in] x  coordinate in [-1,1]
         * @return function value  U_n(x)
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto U(const unsigned int n, ComputationType x) const -> ComputationType
        {
            simd::compare(x);

            return ComputationType((xsimd::sin((n + 1) * acos(x))) / xsimd::sqrt(1 - x * x));
        }

        /**
         * @brief Store Tn(x_i)
         *
         * @param order
         * @return xt::xarray<value_type>
         */
        inline auto set_m_T_of_roots(size_type order) -> xt::xarray<value_type>
        {
            xt::xarray<value_type> T_of_roots(std::vector<std::size_t>(2, order));
            auto roots = this->roots();
            // initialize chebyshev polynomials of root nodes: T_o(x_j)
            /// m_T_of_roots[o,j]  = T_o(x_j)
            for(unsigned int o = 0; o < order; ++o)
            {
                for(unsigned int j = 0; j < order; ++j)
                {   // loops on the roots
                    T_of_roots.at(j, o) = T(o, roots[j]);
                }
            }
            return T_of_roots;
        }

        // private members

        /**
         * @brief The chebyshev polynomials of root nodes:  m_T_of_roots[j,o]  = T_o(x_j)
         * (j,o) for contiguous access when x_j is fixed  used in polynomials_impl
         */
        xt::xarray<value_type> m_T_of_roots{};
    };

    /**
     * @brief
     *
     *  xtensor https://xtensor.readthedocs.io/en/latest/quickref/basic.html
     * @tparam ValueType
     * @tparam Dimension
     * @tparam FarFieldMatrixKernel
     * @tparam Settings
     */
    template<typename ValueType, std::size_t Dimension, typename FarFieldMatrixKernel, typename Settings>
    struct interpolator_traits<chebyshev_interpolator<ValueType, Dimension, FarFieldMatrixKernel, Settings>>
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
        using self_type = chebyshev_interpolator<value_type, dimension, matrix_kernel_type, settings>;
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

#endif   // SCALFMM_INTERPOLATION_CHEBYSHEV_HPP
