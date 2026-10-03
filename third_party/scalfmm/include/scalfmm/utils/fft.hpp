// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/fft.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_FFT_HPP
#define SCALFMM_UTILS_FFT_HPP

#include "scalfmm/utils/massert.hpp"

#include "xtensor/containers/xarray.hpp"

#include <pocketfft_hdronly.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <numeric>

namespace scalfmm::fft
{
    /**
     * @brief Real-to-complex FFTs of the (2 * order - 1)^dim grids used by the uniform interpolator.
     *
     * The forward transform is unnormalized; the inverse transform is scaled by 1 / (2 * order - 1)^dim.
     *
     * @tparam ValueType
     * @tparam dim
     */
    template<typename ValueType, std::size_t dim>
    struct fft
    {
        using value_type = ValueType;
        using fft_type = xt::xarray<value_type>;
        using transformed_fft_type = xt::xarray<std::complex<value_type>>;

        auto initialize(std::size_t order) -> void
        {
            m_real_shape.assign(dim, 2 * order - 1);
            m_complex_shape = m_real_shape;
            m_complex_shape.back() = order;
            m_axes.resize(dim);
            std::iota(m_axes.begin(), m_axes.end(), 0);
            m_real_buffer.resize(m_real_shape);
            m_inverse_scale = value_type(1) / static_cast<value_type>(m_real_buffer.size());
        }

        auto execute_plan(fft_type const& input, transformed_fft_type& output) const -> void
        {
            assertm(has_shape(input, m_real_shape), "Input does not have the shape of the fft handler!");
            assertm(has_shape(output, m_complex_shape), "Output does not have the shape of the fft handler!");
            pocketfft::r2c(m_real_shape, byte_strides(input), byte_strides(output), m_axes, pocketfft::FORWARD,
                           input.data(), output.data(), value_type(1));
        }

        auto execute_inverse_plan(transformed_fft_type const& input) -> void
        {
            assertm(has_shape(input, m_complex_shape), "Input does not have the shape of the fft handler!");
            pocketfft::c2r(m_real_shape, byte_strides(input), byte_strides(m_real_buffer), m_axes,
                           pocketfft::BACKWARD, input.data(), m_real_buffer.data(), m_inverse_scale);
        }

        [[nodiscard]] auto creal_buffer() const -> fft_type const& { return m_real_buffer; }

      private:
        template<typename Array>
        static auto has_shape(Array const& array, pocketfft::shape_t const& shape) -> bool
        {
            return std::equal(array.shape().begin(), array.shape().end(), shape.begin(), shape.end());
        }

        template<typename Array>
        static auto byte_strides(Array const& array) -> pocketfft::stride_t
        {
            pocketfft::stride_t strides(dim);
            for(std::size_t i = 0; i < dim; ++i)
            {
                strides[i] = static_cast<std::ptrdiff_t>(array.strides()[i]) *
                             static_cast<std::ptrdiff_t>(sizeof(typename Array::value_type));
            }
            return strides;
        }

        pocketfft::shape_t m_axes{};
        pocketfft::shape_t m_complex_shape{};
        value_type m_inverse_scale{1};
        fft_type m_real_buffer{};
        pocketfft::shape_t m_real_shape{};
    };
}   // namespace scalfmm::fft

#endif   // SCALFMM_UTILS_FFT_HPP
