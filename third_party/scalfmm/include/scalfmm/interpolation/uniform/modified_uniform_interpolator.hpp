#pragma once

// A kernel-independent uniform fast multipole method based on barycentric rational interpolation
// https://doi.org/10.1007/s11075-022-01481-x

#include <array>
#include <vector>

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

namespace scalfmm::interpolation
{
    template<typename ValueType, std::size_t Dimension, typename FarFieldMatrixKernel>
    struct modified_uniform_interpolator
      : public impl::interpolator<modified_uniform_interpolator<ValueType, Dimension, FarFieldMatrixKernel>>
      , public impl::m2l_handler<modified_uniform_interpolator<ValueType, Dimension, FarFieldMatrixKernel>>
    {
      public:
        using value_type = ValueType;
        static constexpr std::size_t dimension = Dimension;
        using matrix_kernel_type = FarFieldMatrixKernel;
        using size_type = std::size_t;

        using settings = options::fft_;

        static constexpr std::size_t kn = matrix_kernel_type::kn;
        static constexpr std::size_t km = matrix_kernel_type::km;
        using self_type = modified_uniform_interpolator<value_type, dimension, matrix_kernel_type>;
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

        modified_uniform_interpolator() = delete;
        modified_uniform_interpolator(modified_uniform_interpolator const& other) = delete;
        modified_uniform_interpolator(modified_uniform_interpolator&&) noexcept = delete;
        auto operator=(modified_uniform_interpolator const&) noexcept -> modified_uniform_interpolator& = delete;
        auto operator=(modified_uniform_interpolator&&) noexcept -> modified_uniform_interpolator& = delete;
        ~modified_uniform_interpolator()
        {
            for(auto& fftw_ptr: fft_handler)
            {
                delete(fftw_ptr);
            }
        }

        modified_uniform_interpolator(matrix_kernel_type const& far_field, size_type order, size_type tree_height,
                                      value_type root_cell_width, int d)
          : base_interpolator_type(order, tree_height, root_cell_width, 0.0, true)
          , base_m2l_handler_type(far_field, base_interpolator_type::roots(), tree_height, root_cell_width, 0.0, true)
          , d_(d)
          , m_roots(set_roots(order))
          , m_weights(set_weights(order, d))
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

        [[nodiscard]] inline auto roots_impl() const
        {
            return xt::linspace(value_type(-1.), value_type(1), this->order());
        }

        template<typename ComputationType>
        [[nodiscard]] inline auto polynomials_impl(ComputationType x, std::size_t n_size) const -> ComputationType
        {
            auto p = static_cast<int>(this->order() - 1);
            auto n = static_cast<int>(n_size);

            if(d_ < 0)
            {
                ComputationType result = 1.0;
                for(auto m = 0; m <= p; ++m)
                {
                    if(m != n)
                    {
                        result *= p * (x + 1.0) - (2.0 * m);
                    }
                }

                result *= std::pow(-1.0, p - n) /
                          (std::pow(2.0, p) * math::factorial<value_type>(n) * math::factorial<value_type>(p - n));
                return result;
            }

            ComputationType num = m_weights.at(n) / (x - m_roots.at(n));
            ComputationType den = 0.0;
            for(auto m = 0; m <= p; ++m)
            {
                den += m_weights.at(m) / (x - m_roots.at(m));
            }

            ComputationType result = num / den;
            for(auto m = 0; m <= p; ++m)
            {
                auto mask = abs(x - m_roots.at(m)) < 1e-15;
                result = xsimd::select(mask, ComputationType(m == n ? 1.0 : 0.0), result);
            }

            return result;
        }

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

        template<typename ComputationType>
        [[nodiscard]] inline auto derivative_impl(ComputationType x, std::size_t n) const -> ComputationType
        {
            throw std::runtime_error("derivative_impl is not implemented");
        }

        template<typename ComputationType>
        [[nodiscard]] inline auto derivative_impl1(ComputationType x, std::size_t n) const -> ComputationType
        {
            throw std::runtime_error("derivative_impl1 is not implemented");
        }

        // Returns the buffers initialized for the optimized fft
        [[nodiscard]] inline auto buffer_initialization_impl() const -> buffer_type
        {
            // shape for the output fft is [2*order-1, ..., order]
            std::vector<std::size_t> shape(dimension, 2 * this->order() - 1);
            shape.at(dimension - 1) = this->order();
            // Here we return the tensors initialized with zeros
            // we have kn (number of outputs) tensors
            return buffer_type(buffer_shape_type{}, buffer_inner_type(shape, 0.));
        }

        [[nodiscard]] inline auto buffer_shape_impl() const -> std::vector<std::size_t>
        {
            std::vector<std::size_t> shape(dimension, 2 * this->order() - 1);
            shape.at(dimension - 1) = this->order();
            return shape;
        }

        inline auto buffer_reset_impl(buffer_type& buffers) const -> void
        {
            for(std::size_t n{0}; n < kn; ++n)
            {
                buffers.at(n).fill(buffer_value_type{0., 0.});
            }
        }

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

        template<typename Cell, typename ArrayScaleFactor>
        inline auto apply_m2l_impl(Cell const& source_cell, [[maybe_unused]] Cell& target_cell,
                                   [[maybe_unused]] buffer_type& products, interaction_matrix_type const& k,
                                   ArrayScaleFactor scale_factor, [[maybe_unused]] std::size_t n,
                                   [[maybe_unused]] std::size_t m,
                                   [[maybe_unused]] size_type thread_id = 0) const -> void
        {
            // get the transformed multipoles (only available for uniform approximation)
            auto const& transformed_multipoles = source_cell.ctransformed_multipoles();
            // component-wise product in spectral space
            tensor::product(products.at(n), transformed_multipoles.at(m), k.at(n, m), scale_factor.at(n));
        }

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

        inline auto initialize_k_impl() const -> k_tensor_type { return k_tensor_type{}; }

        // This function generates the circulant tensors for generating interaction matrixes.
        template<typename TensorViewX, typename TensorViewY>
        [[nodiscard]] inline auto
        generate_matrix_k_impl(TensorViewX&& X, TensorViewY&& Y, std::size_t n, std::size_t m,
                               [[maybe_unused]] size_type thread_id = 0) const -> k_tensor_type
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
        static int binomial(int i, int j)
        {
            if(j == 0 || j == i)
            {
                return 1;
            }
            return binomial(i - 1, j - 1) + binomial(i - 1, j);
        }

        static auto set_roots(int order) -> std::vector<value_type>
        {
            auto p = order - 1;
            std::vector<value_type> roots(p + 1, 0.0);
            for(auto n = 0; n <= p; ++n)
            {
                roots.at(n) = -1.0 + (2.0 * n) / p;
            }
            return roots;
        }

        static auto set_weights(int order, int d) -> std::vector<value_type>
        {
            auto p = order - 1;
            std::vector<value_type> weights(p + 1, 0.0);
            if(d < 0)
            {
                return weights;
            }

            for(auto n = 0; n <= p; ++n)
            {
                for(auto i = std::max(0, n - d); i <= std::min(p - d, n); ++i)
                {
                    weights.at(n) += binomial(d, n - i);
                }
                weights.at(n) *= std::pow(-1.0, n - d);
            }
            return weights;
        }

        int d_;
        std::vector<value_type> m_roots;
        std::vector<value_type> m_weights;
        std::vector<fft::fft<value_type, dimension>*> fft_handler;
    };

    // traits class to register types inside the interpolator generic class
    template<typename ValueType, std::size_t Dimension, typename FarFieldMatrixKernel>
    struct interpolator_traits<modified_uniform_interpolator<ValueType, Dimension, FarFieldMatrixKernel>>
    {
        using value_type = ValueType;
        using matrix_kernel_type = FarFieldMatrixKernel;
        static constexpr std::size_t dimension = Dimension;
        static constexpr std::size_t kn = matrix_kernel_type::kn;
        static constexpr std::size_t km = matrix_kernel_type::km;
        static constexpr bool enable_symmetries = false;
        using settings = options::fft_;
        using self_type = modified_uniform_interpolator<value_type, dimension, matrix_kernel_type>;
        using base_interpolator_type = impl::interpolator<self_type>;
        using base_m2l_handler_type = impl::m2l_handler<self_type>;
        using storage_type = component::uniform_fft_storage<value_type, dimension, km, kn>;
        using buffer_value_type = typename storage_type::buffer_value_type;   // std::complex<value_type>;
        using buffer_inner_type = typename storage_type::buffer_inner_type;   // xt::xarray<complex_type>;
        using buffer_shape_type = typename storage_type::buffer_shape_type;   // xt::xshape<kn>;
        using buffer_type = typename storage_type::buffer_type;               /// xt::xtensor_fixed<buffer_inner_type,
                                                                              /// buffer_shape_type>;
        using multipoles_container_type = typename storage_type::multipoles_container_type;
        using locals_container_type = typename storage_type::locals_container_type;
        using k_tensor_type = buffer_inner_type;   // xt::xarray<complex_type>;
    };

}   // namespace scalfmm::interpolation
