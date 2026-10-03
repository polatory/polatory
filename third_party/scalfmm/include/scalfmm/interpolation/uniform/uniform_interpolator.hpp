// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/uniform/uniform_interpolator.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_UNIFORM_UNIFORM_INTERPOLATOR_HPP
#define SCALFMM_INTERPOLATION_UNIFORM_UNIFORM_INTERPOLATOR_HPP

#include <xsimd/xsimd.hpp>
//#include <xtensor-fftw/basic.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/utils/xtensor_simd.hpp>
#include <xtensor/core/xvectorize.hpp>

#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/generate_circulent.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/interpolation/interpolator.hpp"
#include "scalfmm/interpolation/m2l_handler.hpp"
#include "scalfmm/interpolation/mapping.hpp"
#include "scalfmm/interpolation/traits.hpp"
#include "scalfmm/interpolation/uniform/uniform_storage.hpp"
#include "scalfmm/matrix_kernels/mk_common.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/simd/memory.hpp"
#include "scalfmm/simd/utils.hpp"
#include "scalfmm/utils/fft.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <array>
#include <vector>

namespace scalfmm::interpolation
{
    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam FarFieldMatrixKernel
     * @tparam Settings
     */
    template<typename ValueType, std::size_t Dimension, typename FarFieldMatrixKernel, typename Settings>
    struct uniform_interpolator
      : public impl::interpolator<uniform_interpolator<ValueType, Dimension, FarFieldMatrixKernel, Settings>>
      , public impl::m2l_handler<uniform_interpolator<ValueType, Dimension, FarFieldMatrixKernel, Settings>>
    {
      public:
        static_assert(options::support(options::_s(Settings{}), options::_s(options::dense, options::low_rank)),
                      "unsupported uniform interpolator options!");
        using value_type = ValueType;
        static constexpr std::size_t dimension = Dimension;
        using matrix_kernel_type = FarFieldMatrixKernel;
        using size_type = std::size_t;
        using settings = Settings;

        using self_type = uniform_interpolator<value_type, dimension, matrix_kernel_type, settings>;
        using base_interpolator_type = impl::interpolator<self_type>;
        using base_m2l_handler_type = impl::m2l_handler<self_type>;

        using base_interpolator_type::base_interpolator_type;
        using base_m2l_handler_type::base_m2l_handler_type;

        /**
         * @brief Construct a new uniform interpolator object
         *
         */
        uniform_interpolator() = delete;

        /**
         * @brief Construct a new uniform interpolator object
         *
         * @param other
         */
        uniform_interpolator(uniform_interpolator const& other) = delete;

        /**
         * @brief Construct a new uniform interpolator object
         *
         */
        uniform_interpolator(uniform_interpolator&&) noexcept = delete;

        /**
         * @brief
         *
         * @return uniform_interpolator&
         */
        auto operator=(uniform_interpolator const&) noexcept -> uniform_interpolator& = delete;

        /**
         * @brief
         *
         * @return uniform_interpolator&
         */
        auto operator=(uniform_interpolator&&) noexcept -> uniform_interpolator& = delete;

        /**
         * @brief Destroy the uniform interpolator object
         *
         */
        ~uniform_interpolator() = default;

        /**
         * @brief Construct a new uniform interpolator object
         *
         * @param far_field
         * @param order
         * @param tree_height
         * @param root_cell_width
         * @param cell_width_extension
         */
        uniform_interpolator(matrix_kernel_type const& far_field, size_type order, size_type tree_height,
                             value_type root_cell_width, value_type cell_width_extension = value_type(0.))
          : base_interpolator_type(order, tree_height, root_cell_width, cell_width_extension, true)
          , base_m2l_handler_type(far_field, roots_impl(), tree_height, root_cell_width, cell_width_extension, true)
        {
            base_interpolator_type::initialize();
            base_m2l_handler_type::initialize(order, root_cell_width, tree_height);
        }

        /**
         * @brief Construct a new uniform interpolator object
         *
         * @param order
         * @param tree_height
         * @param root_cell_width
         * @param cell_width_extension
         */
        uniform_interpolator(size_type order, size_type tree_height, value_type root_cell_width,
                             value_type cell_width_extension = value_type(0.))
          : uniform_interpolator(matrix_kernel_type{}, order, tree_height, root_cell_width, cell_width_extension)
        {
        }

        /**
         * @brief
         *
         * @return auto
         */
        [[nodiscard]] inline auto roots_impl() const
        {
            return xt::linspace(value_type(-1.), value_type(1), this->order());
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
        [[nodiscard]] inline auto polynomials_impl(ComputationType x, std::size_t n) const -> ComputationType
        {
            const auto order{this->order()};
            using value_type = ComputationType;
            // assert(xsimd::any(xsimd::abs(x) - 1. < 10. * std::numeric_limits<value_type>::epsilon()));

            const value_type two_const{2.0};

            simd::compare(x);

            //  Specific precomputation of scale factor
            //  in order to minimize round-off errors
            //  NB: scale factor could be hardcoded (just as the roots)
            value_type scale{};
            const int omn{int(order) - int(n) - 1};

            if(omn % 2 != 0)
            {
                scale = value_type(-1.);   // (-1)^(n-1-(k+1)+1)=(-1)^(omn-1)
            }
            else
            {
                scale = value_type(1.);
            }

            scale /= xsimd::pow(two_const, int(order) - 1) * math::factorial<value_type>(int(n)) *
                     math::factorial<value_type>(omn);

            // compute L
            value_type L{1.};
            for(std::size_t m = 0; m < order; ++m)
            {
                if(m != n)
                {
                    // previous version with risks of round-off error
                    // L *= (x-FUnifRoots<order>::roots[m])/(FUnifRoots<order>::roots[n]-FUnifRoots<order>::roots[m]);

                    // new version (reducing round-off)
                    // regular grid on [-1,1] (h simplifies, only the size of the domain and a remains i.e. 2. and -1.)
                    L *= ((value_type(int(order) - 1) * (x + value_type(1.))) - (two_const * value_type(int(m))));
                    //                   L *= ((coeff * (x + value_type(1.))) - (two_const * value_type(int(m))));
                }
            }

            L *= scale;
            return L;
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
         */
        template<typename VectorType, typename ComputationType, std::size_t Dim>
        inline auto fill_all_polynomials_impl(VectorType& all_poly, container::point<ComputationType, Dim> x,
                                              std::size_t order) const -> void
        {
            // using point_type =  container::point<ComputationType, Dim>;
            // std::vector<point_type, XTENSOR_DEFAULT_ALLOCATOR(point_type)> all_poly(order);
            for(std::size_t d = 0; d < Dim; ++d)
            {
                for(std::size_t o = 0; o < order; ++o)
                {
                    all_poly[o][d] = this->polynomials_impl(x[d], o);
                }
            }
            // auto call_polynomials = [&](auto x, std::size_t n) { return this->polynomials_impl(x, n); };

            // for(std::size_t o = 0; o < order; ++o)
            // {
            //     all_poly[o] = simd::apply_f<point_type::dimension>(call_polynomials, x, o);
            // }
            //  return std::move(all_poly);
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
        [[nodiscard]] inline auto derivative_impl(ComputationType x, std::size_t n) const -> ComputationType
        {
            const auto order{this->order()};
            using value_type = ComputationType;
            // assert(xsimd::any(xsimd::abs(x) - 1. < 10. * std::numeric_limits<value_type>::epsilon()));

            simd::compare(x);
            // optimized variant
            value_type NdL(0.);   // init numerator
            value_type DdL(1.);   // init denominator
            value_type tmpNdL{};

            auto roots(roots_impl());
            const value_type roots_n(roots.at(n));

            for(unsigned int p = 0; p < order; ++p)
            {
                const value_type roots_p(roots.at(p));
                if(p != n)
                {
                    tmpNdL = value_type(1.);
                    for(unsigned int m = 0; m < order; ++m)
                    {
                        const value_type roots_m(roots.at(m));
                        if(m != n && m != p)
                        {
                            tmpNdL *= x - roots_m;
                        }
                    }
                    NdL += tmpNdL;
                    DdL *= roots_n - roots_p;
                }   // endif
            }   // p

            return NdL / DdL;
        }

        /**
         * @brief Compute the weights of the quadrature
         *
         * @param order
         * @return xt::xarray<value_type>
         */
        [[nodiscard]] inline auto generate_weights_impl(std::size_t order) const -> xt::xarray<value_type>
        {
            return xt::xarray<value_type>(std::vector{math::pow(order, dimension)}, value_type(1.));
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
            return "uniform " + std::string(s.value());
        }
    };

    /**
     * @brief Traits class to register types inside the interpolator generic class
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam FarFieldMatrixKernel
     * @tparam Settings
     */
    template<typename ValueType, std::size_t Dimension, typename FarFieldMatrixKernel, typename Settings>
    struct interpolator_traits<uniform_interpolator<ValueType, Dimension, FarFieldMatrixKernel, Settings>>
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
        static constexpr std::size_t kn = matrix_kernel_type::kn;

        /**
         * @brief
         *
         */
        static constexpr std::size_t km = matrix_kernel_type::km;

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
          (enable_symmetries && (matrix_kernel_type::symmetry_tag == matrix_kernels::symmetry::symmetric))};

        using settings = Settings;
        using self_type = uniform_interpolator<value_type, dimension, matrix_kernel_type, settings>;
        using base_interpolator_type = impl::interpolator<self_type>;
        using base_m2l_handler_type = impl::m2l_handler<self_type>;
        using storage_type = component::grid_storage<value_type, dimension, km, kn>;
        /// Temporary buffer to aggregate the multipole in order to perform matrix-matrix product
        using buffer_value_type = value_type;
        // the type of the element inside the buffer
        using buffer_inner_type = std::conditional_t<symmetry_support, xt::xarray<value_type>, meta::empty_inner>;
        // matrix of size 2 to store  the multipole and the local
        using buffer_shape_type = std::conditional_t<symmetry_support, xt::xshape<2>, meta::empty_shape>;
        using buffer_type =
          std::conditional_t<symmetry_support, xt::xtensor_fixed<buffer_inner_type, buffer_shape_type>, meta::empty>;

        // #endif
        using multipoles_container_type = typename storage_type::multipoles_container_type;
        using locals_container_type = typename storage_type::locals_container_type;
        using k_tensor_type = xt::xarray<value_type>;
    };

    /**
     * @brief
     *
     * uniform_interpolator is a CRTP based class
     * meaning that it implements the functions needed by the interpolator class
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam FarFieldMatrixKernel
     */
    template<typename ValueType, std::size_t Dimension, typename FarFieldMatrixKernel>
    struct uniform_interpolator<ValueType, Dimension, FarFieldMatrixKernel, options::fft_>
      : public impl::interpolator<uniform_interpolator<ValueType, Dimension, FarFieldMatrixKernel, options::fft_>>
      , public impl::m2l_handler<uniform_interpolator<ValueType, Dimension, FarFieldMatrixKernel, options::fft_>>
    {
      public:
        using value_type = ValueType;
        static constexpr std::size_t dimension = Dimension;
        using matrix_kernel_type = FarFieldMatrixKernel;
        using size_type = std::size_t;

        using settings = options::fft_;

        static constexpr std::size_t kn = matrix_kernel_type::kn;
        static constexpr std::size_t km = matrix_kernel_type::km;
        using self_type = uniform_interpolator<value_type, dimension, matrix_kernel_type, settings>;
        using base_interpolator_type = impl::interpolator<self_type>;
        using base_m2l_handler_type = impl::m2l_handler<self_type>;
        using storage_type = typename interpolator_traits<self_type>::storage_type;
        using buffer_value_type =
          typename interpolator_traits<self_type>::buffer_value_type;   // std::complex<value_type>;
        using buffer_inner_type =
          typename interpolator_traits<self_type>::buffer_inner_type;                       // xt::xarray<complex_type>;
        using k_tensor_type = typename interpolator_traits<self_type>::buffer_inner_type;   // xt::xarray<complex_type>;
        using interaction_matrix_type = typename base_m2l_handler_type::interaction_matrix_type;
        using buffer_shape_type = typename interpolator_traits<self_type>::buffer_shape_type;   // xt::xshape<kn>;
        using buffer_type =
          typename interpolator_traits<self_type>::buffer_type;   // xt::xtensor_fixed<buffer_inner_type,
        using multipoles_inner_type =
          typename memory::storage_traits<typename storage_type::multipoles_storage_type>::inner_type;
        using locals_inner_type =
          typename memory::storage_traits<typename storage_type::locals_storage_type>::inner_type;

        using base_interpolator_type::base_interpolator_type;
        using base_m2l_handler_type::base_m2l_handler_type;

        /**
         * @brief Construct a new uniform interpolator object
         *
         */
        uniform_interpolator() = delete;

        /**
         * @brief Construct a new uniform interpolator object
         *
         * @param other
         */
        uniform_interpolator(uniform_interpolator const& other) = delete;

        /**
         * @brief Construct a new uniform interpolator object
         *
         */
        uniform_interpolator(uniform_interpolator&&) noexcept = delete;

        /**
         * @brief
         *
         * @return uniform_interpolator&
         */
        auto operator=(uniform_interpolator const&) noexcept -> uniform_interpolator& = delete;

        /**
         * @brief
         *
         * @return uniform_interpolator&
         */
        auto operator=(uniform_interpolator&&) noexcept -> uniform_interpolator& = delete;

        /**
         * @brief Destroy the uniform interpolator object
         *
         */
        ~uniform_interpolator()
        {
            for(auto& fftw_ptr: fft_handler)
            {
                delete(fftw_ptr);
            }
        }

        /**
         * @brief Construct a new uniform interpolator object
         *
         * @param far_field
         * @param order
         * @param tree_height
         * @param root_cell_width
         * @param cell_width_extension
         */
        uniform_interpolator(matrix_kernel_type const& far_field, size_type order, size_type tree_height,
                             value_type root_cell_width, value_type cell_width_extension = value_type(0.))
          : base_interpolator_type(order, tree_height, root_cell_width, cell_width_extension, true)
          , base_m2l_handler_type(far_field, base_interpolator_type::roots(), tree_height, root_cell_width,
                                  cell_width_extension, true)
          , m_factorials(set_factorials(order))
          , m_omn(set_omn(order))

        {
#ifdef _OPENMP
#pragma omp parallel
            {
#pragma omp single
                {
                    fft_handler.resize(omp_get_num_threads());
                }
                fft_handler.at(omp_get_thread_num()) = new fft::fft<value_type, dimension>();
                fft_handler.at(omp_get_thread_num())->initialize(order);
            }
#else
            fft_handler.resize(1);
            fft_handler.at(0) = new fft::fft<value_type, dimension>();
            fft_handler.at(0)->initialize(order);
#endif
            base_interpolator_type::initialize();
            base_m2l_handler_type::initialize(order, root_cell_width, tree_height);
        }

        /**
         * @brief Construct a new uniform interpolator object
         *
         * @param order
         * @param tree_height
         * @param root_cell_width
         * @param cell_width_extension
         */
        uniform_interpolator(size_type order, size_type tree_height, value_type root_cell_width,
                             value_type cell_width_extension = value_type(0.))
          : uniform_interpolator(matrix_kernel_type{}, order, tree_height, root_cell_width, cell_width_extension)
        {
        }

        /**
         * @brief
         *
         * @return auto
         */
        [[nodiscard]] inline auto roots_impl() const
        {
            return xt::linspace(value_type(-1.), value_type(1), this->order());
        }

        /**
         * @brief compute  the nth lagrange function at point x
         *
         * The formula is developed for \f$ x_j = -1 + \frac{2}{N-1} j\f$
         *
         * @param x the points
         * @param n the order
         * @return ComputationType
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto polynomials_impl(ComputationType x, std::size_t n) const -> ComputationType
        {
            const auto order{this->order()};

            simd::compare(x);

            //  Specific precomputation of scale factor
            //  in order to minimize round-off errors
            //  NB: scale factor could be hardcoded (just as the roots)
            const int omn{int(order) - int(n) - 1};
            ComputationType scale{m_omn[omn]};

            // compute L
            ComputationType L{1.};
            const ComputationType x_coeff{(order - 1) * value_type(0.5) * (x + value_type(1.0))};
            scale /= ComputationType((m_factorials[n] * m_factorials[omn]));

            for(std::size_t m = 0; m < n; ++m)
            {
                L *= x_coeff - ComputationType(m);
            }
            for(std::size_t m = n + 1; m < order; ++m)
            {
                L *= x_coeff - ComputationType(m);
            }

            L *= scale;

            return L;
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
         */
        template<typename VectorType, typename ComputationType, std::size_t Dim>
        inline auto fill_all_polynomials_impl(VectorType& all_poly, container::point<ComputationType, Dim>& x,
                                              std::size_t order) const -> void
        {
            // using point_type =  container::point<ComputationType, Dim>;
            // std::vector<point_type, XTENSOR_DEFAULT_ALLOCATOR(point_type)> all_poly(order);
            for(std::size_t d = 0; d < Dim; ++d)
            {
                for(std::size_t o = 0; o < order; ++o)
                {
                    all_poly[o][d] = this->polynomials_impl(x[d], o);
                }
            }
            // auto call_polynomials = [&](auto x, std::size_t n) { return this->polynomials_impl(x, n); };

            // for(std::size_t o = 0; o < order; ++o)
            // {
            //     all_poly[o] = simd::apply_f<point_type::dimension>(call_polynomials, x, o);
            // }
            // return std::move(all_poly);
        }

        /**
         * @brief compute the derivative of the nth lagrange function at point x
         *
         * The derivative uses the following formula
         * \f$ L'_j(x) = \frac{ \sum_{i = 0, i \ne j}^{N-1}{\prod_{k = 0, k \ne i, k \ne j}^{N-1}{x - x_k} } }{\prod_{m
         * = 0, m \ne j}^{N - 1}{x_j - x_m}} \f$
         *
         * @tparam ComputationType
         * @param x the points
         * @param n the order
         * @return ComputationType
         */
        template<typename ComputationType>
        [[nodiscard]] inline auto derivative_impl(ComputationType x, std::size_t n) const -> ComputationType
        {
            const auto order{this->order()};
            using value_type = ComputationType;
            // assert(xsimd::any(xsimd::abs(x) - 1. < 10. * std::numeric_limits<value_type>::epsilon()));

            simd::compare(x);
            // optimized variant
            value_type NdL(0.);   // init numerator
            value_type DdL(1.);   // init denominator
            value_type tmpNdL{};

            auto roots(roots_impl());
            const value_type roots_n(roots.at(n));

            for(unsigned int p = 0; p < order; ++p)
            {
                const value_type roots_p(roots.at(p));
                if(p != n)
                {
                    tmpNdL = value_type(1.);
                    for(unsigned int m = 0; m < order; ++m)
                    {
                        const value_type roots_m(roots.at(m));
                        if(m != n && m != p)
                        {
                            tmpNdL *= x - roots_m;
                        }
                    }
                    NdL += tmpNdL;
                    DdL *= roots_n - roots_p;
                }   // endif
            }   // p

            return NdL / DdL;
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
        [[nodiscard]] inline auto derivative_impl1(ComputationType x, std::size_t n) const -> ComputationType
        {
            const auto order{this->order()};
            auto roots(roots_impl());

            ComputationType L{polynomials_impl(x, n)};
            ComputationType sum{0.0};
            const ComputationType one_const{1.0};
            for(std::size_t m = 0; m < n; ++m)
            {
                sum += one_const / (x - ComputationType(roots[m]));
            }
            for(std::size_t m = n + 1; m < order; ++m)
            {
                sum += one_const / (x - ComputationType(roots[m]));
            }
            return L * sum;
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
        /**
         * @brief Returns the buffers initialized for the optimized fft
         *
         * @return buffer_type
         */
        [[nodiscard]] inline auto name() const noexcept -> std::string const
        {
            settings s{};
            return "uniform " + std::string(s.value());
        }
        [[nodiscard]] inline auto buffer_initialization_impl() const -> buffer_type
        {
            // shape for the output fft is [2*order-1, ..., order]
            std::vector<std::size_t> shape(dimension, 2 * this->order() - 1);
            shape.at(dimension - 1) = this->order();
            // Here we return the tensors initialized with zeros
            // we have kn (number of outputs) tensors
            return buffer_type(buffer_shape_type{}, buffer_inner_type(shape, 0.));
        }

        /**
         * @brief
         *
         * @return std::vector<std::size_t>
         */
        [[nodiscard]] inline auto buffer_shape_impl() const -> std::vector<std::size_t>
        {
            std::vector<std::size_t> shape(dimension, 2 * this->order() - 1);
            shape.at(dimension - 1) = this->order();
            return shape;
        }

        /**
         * @brief
         *
         * @param buffers
         */
        inline auto buffer_reset_impl(buffer_type& buffers) const -> void
        {
            for(std::size_t n{0}; n < kn; ++n)
            {
                buffers.at(n).fill(buffer_value_type{0., 0.});
            }
        }

        /**
         * @brief Preprocessing function for applying ffts on multipole arrays
         *
         * This function is called as soon as the multipoles are computed and available just after the P2M or M2M
         * functions
         * @tparam Cell
         * @param current_cell the current cell containing the multipoles to preprocess
         * @param thread_id the thread_id needed for the fft
         */
        template<typename Cell>
        inline auto apply_multipoles_preprocessing_impl(Cell& current_cell,
                                                        [[maybe_unused]] size_type thread_id = 0) const -> void
        {
            using multipoles_container = typename Cell::storage_type::multipoles_storage_type;
            using fft_type = xt::xarray<typename multipoles_container::value_type>;
            const auto order{this->order()};
            // Get the multipoles and transformed mutlipoles tensors
            auto const& multipoles = current_cell.cmultipoles();
            auto& mtilde = current_cell.transformed_multipoles();

            // Input fft tensor shape i.e the multipoles padded with zeros
            std::vector<std::size_t> shape_transformed(dimension, std::size_t(2 * order - 1));

            // Creating padded tensors for the ffts inputs
            fft_type padded_tensor(shape_transformed, value_type(0.));

            for(std::size_t m{0}; m < km; ++m)
            {
                auto view_in_padded = tensor::get_view<dimension>(padded_tensor, xt::range(0, order));
                view_in_padded = multipoles.at(m);
                fft_handler.at(thread_id)->execute_plan(padded_tensor, mtilde.at(m));
            }
        }

        // m2l operator impl
        /**
         * @brief m2l function specialized for the FFT optimization
         *
         * @tparam Cell
         * @tparam ArrayScaleFactor
         * @param source_cell
         * @param target_cell
         * @param products the buffer to store the result in the spectral space
         * @param k the kernel matrix
         * @param scale_factor the scaling factor
         * @param n the nth output
         * @param m the mth input
         * @param thread_id the thread id
         */
        template<typename Cell, typename ArrayScaleFactor>
        inline auto apply_m2l_impl(Cell const& source_cell, [[maybe_unused]] Cell& target_cell,
                                   [[maybe_unused]] buffer_type& products, interaction_matrix_type const& k,
                                   ArrayScaleFactor scale_factor, [[maybe_unused]] std::size_t n,
                                   [[maybe_unused]] std::size_t m, [[maybe_unused]] size_type thread_id = 0) const
          -> void
        {
            // get the transformed multipoles (only available for uniform approximation)
            auto const& transformed_multipoles = source_cell.ctransformed_multipoles();
            // component-wise product in spectral space
            tensor::product(products.at(n), transformed_multipoles.at(m), k.at(n, m), scale_factor.at(n));
        }

        // postprocessing locals
        /**
         * @brief postprocessing locals
         *
         * We apply the inverse fft to construct the real local arrays (for all outputs) of the current_cell
         *
         * @tparam Cell
         * @param current_cell the cell containing the local array to post-process
         * @param products the spectral coefficients of the local array
         * @param thread_id the thread id
         */
        template<typename Cell>
        inline auto apply_multipoles_postprocessing_impl(Cell& current_cell,
                                                         [[maybe_unused]] buffer_type const& products,
                                                         [[maybe_unused]] size_type thread_id = 0) const -> void
        {
            const auto order{this->order()};
            auto& target_expansion = current_cell.locals();

            // applying kn inverse real ffts
            for(std::size_t n = 0; n < kn; ++n)
            {
                // we get the view [0,order] from the output of the ifft.
                this->fft_handler.at(thread_id)->execute_inverse_plan(products.at(n));
                auto view_for_update = xt::eval(
                  tensor::get_view<dimension>(this->fft_handler.at(thread_id)->creal_buffer(), xt::range(0, order)));
                // accumulate the results in local expansions
                // target_expansion.at(n) += view_for_update;
                cxxblas::axpy(static_cast<int>(view_for_update.size()), value_type(1.), view_for_update.data(),
                              static_cast<int>(1), target_expansion.at(n).data(), static_cast<int>(1));
            }
        }

        /**
         * @brief
         *
         * @return k_tensor_type
         */
        inline auto initialize_k_impl() const -> k_tensor_type { return k_tensor_type{}; }

        /**
         * @brief This function generates the circulant tensors for generating interaction matrixes.
         *
         * @tparam TensorViewX
         * @tparam TensorViewY
         * @param X
         * @param Y
         * @param n
         * @param m
         * @param thread_id
         * @return k_tensor_type
         */
        template<typename TensorViewX, typename TensorViewY>
        [[nodiscard]] inline auto generate_matrix_k_impl(TensorViewX&& X, TensorViewY&& Y, std::size_t n, std::size_t m,
                                                         [[maybe_unused]] size_type thread_id = 0) const
          -> k_tensor_type
        {
            build<value_type, dimension, dimension> build_c{};
            // generate the circulant tensor
            auto C = build_c(std::forward<TensorViewX>(X), std::forward<TensorViewY>(Y), this->order(),
                             this->matrix_kernel(), n, m);
            // return the fft
            //      using return_type = std::decay_t<decltype(xt::fftw::rfftn<dimension>(C))>;
            using return_type = xt::xarray<std::complex<value_type>>;

            std::vector<std::size_t> output_forward_shape(dimension, std::size_t(2 * this->order() - 1));
            output_forward_shape.at(dimension - 1) = this->order();
            return_type ctilde(output_forward_shape);
            this->fft_handler.at(thread_id)->execute_plan(C, ctilde);
            return ctilde;
        }

      private:
        /**
         * @brief Set the factorials object
         *
         * @param order
         * @return std::vector<value_type>
         */
        inline auto set_factorials(size_type order) -> std::vector<value_type>
        {
            std::vector<value_type> factorials(order);
            factorials[0] = factorials[1] = value_type(1.0);
            for(unsigned int o = 2; o < factorials.size(); ++o)
            {
                factorials[o] = o * factorials[o - 1];
            }

            return factorials;
        }

        /**
         * @brief Set the omn object
         *
         * @param order
         * @return std::vector<value_type>
         */
        inline auto set_omn(size_type order) -> std::vector<value_type>
        {
            std::vector<value_type> omn(order, value_type(1.0));
            for(unsigned int o = 1; o < omn.size(); ++o)
            {
                if(o % 2 != 0)
                {
                    omn[o] = value_type(-1.0);
                }
            }
            return omn;
        }

        /**
         * @brief
         *
         */
        std::vector<value_type> m_factorials;

        /**
         * @brief
         *
         */
        std::vector<value_type> m_omn;

        /**
         * @brief
         *
         */
        std::vector<fft::fft<value_type, dimension>*> fft_handler;
    };

    /**
     * @brief Traits class to register types inside the interpolator generic class.
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam FarFieldMatrixKernel
     */
    template<typename ValueType, std::size_t Dimension, typename FarFieldMatrixKernel>
    struct interpolator_traits<uniform_interpolator<ValueType, Dimension, FarFieldMatrixKernel, options::fft_>>
    {
        using value_type = ValueType;
        using matrix_kernel_type = FarFieldMatrixKernel;
        static constexpr std::size_t dimension = Dimension;
        static constexpr std::size_t kn = matrix_kernel_type::kn;
        static constexpr std::size_t km = matrix_kernel_type::km;
        static constexpr bool enable_symmetries = false;
        using settings = options::fft_;
        using self_type = uniform_interpolator<value_type, dimension, matrix_kernel_type, settings>;
        using base_interpolator_type = impl::interpolator<self_type>;
        using base_m2l_handler_type = impl::m2l_handler<self_type>;
        using storage_type = component::uniform_fft_storage<value_type, dimension, km, kn>;
        using buffer_value_type = typename storage_type::buffer_value_type;   // std::complex<value_type>;
        using buffer_inner_type = typename storage_type::buffer_inner_type;   // xt::xarray<complex_type>;
        using buffer_shape_type = typename storage_type::buffer_shape_type;   // xt::xshape<kn>;
        using buffer_type =
          typename storage_type::buffer_type;   /// xt::xtensor_fixed<buffer_inner_type, buffer_shape_type>;
        using multipoles_container_type = typename storage_type::multipoles_container_type;
        using locals_container_type = typename storage_type::locals_container_type;
        using k_tensor_type = buffer_inner_type;   // xt::xarray<complex_type>;
    };

}   // namespace scalfmm::interpolation

#endif   // SCALFMM_INTERPOLATION_UNIFORM_UNIFORM_INTERPOLATOR_HPP
