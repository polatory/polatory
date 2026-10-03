// --------------------------------
// See LICENCE file at project root
// File : include/scalfmm/utils/tensor.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_TENSOR_HPP
#define SCALFMM_UTILS_TENSOR_HPP

#include "scalfmm/container/point.hpp"
#include "scalfmm/utils/math.hpp"
#include <scalfmm/meta/traits.hpp>
#include <scalfmm/utils/io_helpers.hpp>

#include "xflens/cxxblas/typedefs.h"
#include "xsimd/xsimd.hpp"
#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xblas_config.hpp"
#include "xtensor-blas/xblas_utils.hpp"
#include "xtensor-blas/xlinalg.hpp"

#include "xtensor/core/xeval.hpp"
#include "xtensor/views/xslice.hpp"
#include "xtensor/core/xtensor_config.hpp"
#include "xtensor/utils/xutils.hpp"

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/misc/xpad.hpp>
#include <xtensor/views/xstrided_view.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/views/xview.hpp>

#include <complex>
#include <cstddef>
#include <iterator>   // std::begin
#include <numeric>    // for std::iota
#include <scalfmm/meta/traits.hpp>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>   // for std::forward

#ifndef XTENSOR_SELECT_ALIGN
#define XTENSOR_SELECT_ALIGN(T) (XTENSOR_DEFAULT_ALIGNMENT != 0 ? XTENSOR_DEFAULT_ALIGNMENT : alignof(T))
#endif

#ifndef XTENSOR_FIXED_ALIGN
#define XTENSOR_FIXED_ALIGN XTENSOR_SELECT_ALIGN(void*)
#endif

namespace scalfmm::tensor
{
    /**
     * @brief
     *
     */
    struct row
    {
    };

    /**
     * @brief
     *
     */
    struct column
    {
    };

    namespace details
    {
        /**
         * @brief Get the view object
         *
         * @tparam Tensor
         * @tparam Range
         * @tparam Is
         * @param t
         * @param indice
         * @param range
         * @param r
         * @param seq
         * @return constexpr auto
         */
        template<typename Tensor, typename Range, std::size_t... Is>
        [[nodiscard]] inline constexpr auto get_view(Tensor&& t, std::size_t indice, Range&& range, row r,
                                                     std::index_sequence<Is...> seq)
        {
            return xt::view(std::forward<Tensor>(t), indice, meta::id<Is>(std::forward<Range>(range))...);
        }

        /**
         * @brief Get the view object
         *
         * @tparam Tensor
         * @tparam Range
         * @tparam Is
         * @param t
         * @param indice
         * @param range
         * @param c
         * @param seq
         * @return constexpr auto
         */
        template<typename Tensor, typename Range, std::size_t... Is>
        [[nodiscard]] inline constexpr auto get_view(Tensor&& t, std::size_t indice, Range&& range, column c,
                                                     std::index_sequence<Is...> seq)
        {
            return xt::view(std::forward<Tensor>(t), meta::id<Is>(std::forward<Range>(range))..., indice);
        }

        /**
         * @brief Get the view object
         *
         * @tparam Tensor
         * @tparam Slice
         * @tparam Is
         * @param t
         * @param s
         * @param seq
         * @return constexpr auto
         */
        template<typename Tensor, typename Slice, std::size_t... Is>
        [[nodiscard]] inline constexpr auto get_view(Tensor&& t, Slice&& s, std::index_sequence<Is...> seq)
        {
            return xt::view(std::forward<Tensor>(t), meta::id<Is>(std::forward<Slice>(s))...);
        }

        /**
         * @brief Get the inner view object
         *
         * @tparam Tensor
         * @tparam Slice
         * @tparam Is
         * @param t
         * @param s
         * @param seq
         * @return constexpr auto
         */
        template<typename Tensor, typename Slice, std::size_t... Is>
        [[nodiscard]] inline constexpr auto get_inner_view(Tensor&& t, Slice&& s, std::index_sequence<Is...> seq)
        {
            return xt::view(std::forward<Tensor>(t), xt::all(), meta::id<Is>(std::forward<Slice>(s))...);
        }

        /**
         * @brief
         *
         * @tparam Tensor
         * @tparam Is
         * @param t
         * @param indice
         * @param seq
         * @return constexpr auto
         */
        template<typename Tensor, std::size_t... Is>
        [[nodiscard]] inline constexpr auto gather(Tensor&& t, std::size_t indice, std::index_sequence<Is...> seq)
        {
            return xt::view(std::forward<Tensor>(t), xt::all(), indice, meta::id<Is>(xt::all())...);
        }

        /**
         * @brief
         *
         * @tparam Expression
         * @tparam Is
         * @param e
         * @param s
         * @return constexpr auto
         */
        template<typename Expression, std::size_t... Is>
        [[nodiscard]] inline constexpr auto generate_meshgrid(Expression const& e, std::index_sequence<Is...> s)
        {
            return xt::meshgrid(meta::id<Is>(e)...);
        }

    }   // namespace details

    /**
     * @brief Get the view object
     *
     * @tparam N
     * @tparam Tensor
     * @tparam Slice
     * @param t
     * @param s
     * @return constexpr auto
     */
    template<std::size_t N, typename Tensor, typename Slice>
    [[nodiscard]] inline constexpr auto get_view(Tensor&& t, Slice&& s)
    {
        return details::get_view(std::forward<Tensor>(t), std::forward<Slice>(s), std::make_index_sequence<N>{});
    }

    /**
     * @brief Get the inner view object
     *
     * @tparam N
     * @tparam Tensor
     * @tparam Slice
     * @param t
     * @param s
     * @return constexpr auto
     */
    template<std::size_t N, typename Tensor, typename Slice>
    [[nodiscard]] inline constexpr auto get_inner_view(Tensor&& t, Slice&& s)
    {
        return details::get_inner_view(std::forward<Tensor>(t), std::forward<Slice>(s), std::make_index_sequence<N>{});
    }

    /**
     * @brief Get the view object
     *
     * @tparam N
     * @tparam Tensor
     * @tparam Range
     * @tparam Tag
     * @param t
     * @param indice
     * @param range
     * @param tag
     * @return constexpr auto
     */
    template<std::size_t N, typename Tensor, typename Range, typename Tag>
    [[nodiscard]] inline constexpr auto get_view(Tensor&& t, std::size_t indice, Range&& range, Tag tag)
    {
        return details::get_view(std::forward<Tensor>(t), indice, std::forward<Range>(range), tag,
                                 std::make_index_sequence<N - 1>{});
    }

    /**
     * @brief
     *
     * @tparam N
     * @tparam Tensor
     * @param t
     * @param indice
     * @return constexpr auto
     */
    template<std::size_t N, typename Tensor>
    [[nodiscard]] inline constexpr auto gather(Tensor&& t, std::size_t indice)
    {
        if constexpr(N < 3)
        {
            return xt::view(std::forward<Tensor>(t), indice);
        }
        else
        {
            return details::gather(std::forward<Tensor>(t), indice, std::make_index_sequence<N - 2>{});
        }
    }

    /**
     * @brief
     *
     * @tparam Dimension
     * @tparam Expression
     * @param e
     * @return constexpr auto
     */
    template<std::size_t Dimension, typename Expression>
    [[nodiscard]] inline constexpr auto generate_meshgrid(Expression const& e)
    {
        return details::generate_meshgrid(e, std::make_index_sequence<Dimension>{});
    }

    /**
     * @brief
     *
     * @tparam Dimension
     * @tparam Expression
     * @tparam Repeats
     * @param e
     * @param r
     * @return auto
     */
    template<std::size_t Dimension, typename Expression, typename Repeats>
    [[nodiscard]] inline auto repeat(Expression&& e, Repeats&& r)
    {
        auto rep{r};
        auto t{e};
        for(std::size_t i = 0; i < Dimension; ++i)
        {
            t = xt::repeat(t, rep, i);
        }
        return t;
    }

    /**
     * @brief
     *
     * @tparam Tensor
     * @param t
     * @param source
     * @param destination
     * @return auto
     */
    template<typename Tensor>
    [[nodiscard]] inline auto moveaxis(Tensor&& t, std::size_t source, std::size_t destination)
    {
        using shape_type = typename std::decay_t<Tensor>::shape_type;

        if(source >= t.dimension() || destination >= t.dimension())
        {
            throw std::runtime_error("Cant't move axis, source has not the same length as destination.");
        }

        shape_type perm{};
        shape_type iota_{};

        if(!(xt::resize_container(perm, t.dimension()) && xt::resize_container(iota_, t.dimension())))
        {
            throw std::runtime_error("Cant't resize shape type.");
        }

        std::iota(std::begin(iota_), std::end(iota_), 0);
        std::iota(std::begin(perm), std::end(perm), 0);

        if(source > destination)
        {
            std::copy(std::begin(iota_) + destination, std::begin(iota_) + source, std::begin(perm) + destination + 1);
            perm.at(destination) = source;
            return xt::transpose(std::forward<Tensor>(t), perm);
        }

        std::copy(std::begin(iota_) + source + 1, std::begin(iota_) + destination + 1, std::begin(perm) + source);
        perm.at(destination) = source;
        return xt::transpose(std::forward<Tensor>(t), perm);
    }

    /**
     * @brief
     *
     * @tparam Tensor
     * @param t
     * @param axe
     * @return auto
     */
    template<typename Tensor>
    [[nodiscard]] inline auto unfold(Tensor&& t, std::size_t axe)
    {
        using tensor_type = std::decay_t<Tensor>;
        using size_type = typename tensor_type::size_type;

        if(axe >= t.dimension())
        {
            throw std::runtime_error("Unfold : Axe to unfold through is out of tensor dimension.");
        }

        auto shape = t.shape();
        size_type slice = shape.at(axe);

        std::swap(shape.at(0), shape.at(axe));

        size_type leading_dim =
          std::accumulate(std::begin(shape) + 1, std::end(shape), size_type(1), [](auto a, auto b) { return a * b; });

        using result_of_moveaxis = decltype(moveaxis(std::forward<Tensor>(t), axe, 0));

        return xt::reshape_view(std::forward<result_of_moveaxis>(moveaxis(std::forward<Tensor>(t), axe, 0)),
                                {slice, leading_dim});
    }

    /**
     * @brief
     *
     * @tparam Matrix
     * @tparam Shape
     * @param m
     * @param axe
     * @param shape
     * @return auto
     */
    template<typename Matrix, typename Shape>
    [[nodiscard]] inline auto fold(Matrix&& m, std::size_t axe, Shape shape)
    {
        using size_type = typename Shape::size_type;

        if(axe >= shape.size())
        {
            throw std::runtime_error("Fold : Folding axe out of shape range.");
        }

        Shape new_shape;

        if(!xt::resize_container(new_shape, shape.size()))
        {
            throw std::runtime_error("Fold : Cant't resize shape type.");
        }

        new_shape.at(0) = shape.at(axe);

        size_type j{1};
        for(size_type i = 0; i < shape.size(); ++i)
        {
            if(i != axe)
            {
                new_shape.at(j) = shape.at(i);
                j++;
            }
        }

        using result_of_reshaped = decltype(xt::reshape_view(std::forward<Matrix>(m), new_shape));

        return moveaxis(
          std::forward<result_of_reshaped>(xt::reshape_view(std::forward<Matrix>(m), std::move(new_shape))), 0, axe);
    }

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam ScalarType
     * @param accumulate
     * @param t
     * @param k
     * @param scalar
     * @return auto
     */
    template<typename ValueType, typename ScalarType>
    inline auto product(xt::xarray<ValueType>& accumulate, xt::xarray<ValueType> const& t,
                        xt::xarray<ValueType> const& k, ScalarType scalar)
    {
        using value_type = ValueType;
        using simd_type = xsimd::simd_type<value_type>;
        constexpr std::size_t inc = simd_type::size;
        const std::size_t size = accumulate.size();
        const std::size_t vec_size = size - size % inc;
        value_type scale_cp(scalar);
        simd_type splat(scale_cp);
        for(std::size_t i{0}; i < vec_size; i += inc)
        {
            auto t_ = xsimd::load_aligned(&t.data()[i]);
            auto k_ = xsimd::load_aligned(&k.data()[i]);
            auto acc_ = xsimd::load_aligned(&accumulate.data()[i]);
            auto times = k_ * splat;
            auto tmp = xsimd::fma(t_, times, acc_);
            //acc_ += t_ * k_ * splat;
            tmp.store_aligned(&accumulate.data()[i]);
        }
        for(std::size_t i{vec_size}; i < size; ++i)
        {
            accumulate.data()[i] += t.data()[i] * k.data()[i] * scale_cp;
            //auto tmp = accumulate.data()[i];
            //accumulate.data()[i] = xsimd::fma(t.data()[i],k.data()[i]*scale_cp, tmp);
        }
    }

    /**
     * @brief
     *
     * @tparam T
     * @param a
     * @param b
     * @return T
     */
    template<typename T>
    inline T simd_complex_prod(T a, T b)
    {
        return T(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
    }

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam ScalarType
     * @param accumulate
     * @param t
     * @param k
     * @param scalar
     * @return auto
     */
    template<typename ValueType, typename ScalarType>
    inline auto product(xt::xarray<std::complex<ValueType>>& accumulate, xt::xarray<std::complex<ValueType>> const& t,
                        xt::xarray<std::complex<ValueType>> const& k, ScalarType scalar)
    {
        using simd_type = xsimd::simd_type<std::complex<ValueType>>;
        constexpr std::size_t inc = simd_type::size;
        const std::size_t size = accumulate.size();
        const std::size_t vec_size = size - size % inc;
        ValueType scale_cp(scalar);
        simd_type splat(scale_cp);
        auto ptr_t = t.data();
        auto ptr_k = k.data();
        auto ptr_acc = accumulate.data();
        for(std::size_t i{0}; i < vec_size; i += inc)
        {
            const auto t_ = xsimd::load_aligned(&ptr_t[i]);
            const auto k_ = xsimd::load_aligned(&ptr_k[i]);
            auto acc_ = xsimd::load_aligned(&ptr_acc[i]);
            //auto times = k_*splat;
            //auto tmp = xsimd::fma(t_, times, acc_);
            auto tmp1 = simd_complex_prod(t_, k_);
            auto tmp2 = simd_complex_prod(tmp1, splat);
            acc_ += tmp2;
            acc_.store_aligned(&ptr_acc[i]);
        }
        for(std::size_t i{vec_size}; i < size; ++i)
        {
            ptr_acc[i] += ptr_t[i] * ptr_k[i] * scale_cp;
            //auto tmp = accumulate.data()[i];
            //accumulate.data()[i] = xsimd::fma(t.data()[i],k.data()[i]*scale_cp, tmp);
        }
    }

    /**
     * @brief Generate a tensor of points from the interpolator roots.
     *
     * @tparam Dimension
     * @tparam RootsExpression
     * @param roots
     * @return auto
     */
    template<std::size_t Dimension, typename RootsExpression>
    inline auto generate_grid_of_points(RootsExpression const& roots)
    {
        static constexpr std::size_t dimension{Dimension};
        using value_type = typename RootsExpression::value_type;

        auto nnodes{math::pow(roots.size(), dimension)};
        auto X_gen = tensor::generate_meshgrid<dimension>(roots);
        // we evaluate the generator
        auto X = std::apply([](auto&... xs) { return std::make_tuple(xt::eval(xs)...); }, X_gen);
        // get an array of points
        auto X_points = xt::xarray<container::point<value_type, dimension>>::from_shape(std::get<0>(X).shape());

        // we flatten the grids
        auto X_flatten_views =
          std::apply([](auto&&... xs) { return std::make_tuple(xt::flatten(std::forward<decltype(xs)>(xs))...); }, X);

        // we flatten the tensor of points
        auto X_points_flatten_views = xt::flatten(X_points);

        // here, we reconstruct the points for the grids.
        // i.e p(x,y,z)[i] <- x[i], y[i], z[i]
        for(std::size_t i = 0; i < nnodes; ++i)
        {
            X_points_flatten_views[i] =
              std::apply([&i](auto&&... xs)
                         { return container::point<value_type, dimension>{std::forward<decltype(xs)>(xs)[i]...}; },
                         X_flatten_views);
        }
        return X_points;
    }

    /**
     * @brief  Perform matrix-vector product
     *
     *  Perform the matrix-vector product with gemv
     *    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
     *
     * @tparam TensorType
     * @tparam InteractionMatrixType
     * @tparam ValueType
     * @param multipoles the x vector
     * @param locals  the y vector
     * @param A  The interaction matrix (KNM)
     * @param scale_factor the scale factor alpha
     * @param acc if true beta is 1 otherwise 0.
     * @param trans boolean to check if we transpose the matrix (true) or not (false)
     */
    template<typename TensorType, typename TensorType1, typename InteractionMatrixType, typename ValueType>
    inline auto blas2_product(TensorType const& multipoles, TensorType1& locals, InteractionMatrixType const& A,
                              ValueType scale_factor, bool acc, bool trans = false) -> void
    {
        using value_type = ValueType;
        auto const& shapeA = A.shape();
        //     locals = xt::linalg::dot(A, multipoles);

        cxxblas::gemv<xt::blas_index_t>(
          xt::get_blas_storage_order(locals),   // locals.at(n)
          (trans ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans), static_cast<xt::blas_index_t>(shapeA[0]),
          static_cast<xt::blas_index_t>(shapeA[1]), scale_factor, A.data() + A.data_offset(),
          static_cast<xt::blas_index_t>(xt::get_leading_stride(A)),
          multipoles.data() + multipoles.data_offset(),   // multipoles.at(m)
          static_cast<xt::blas_index_t>(1), (acc ? value_type(1.) : value_type(0.)),
          locals.data() + locals.data_offset(), static_cast<xt::blas_index_t>(1));
    }

    /**
     * @brief  Perform matrix-vector product
     *
     *  Perform the matrix-matrix product with gemm
     *     C := alpha*A*B + beta*C
     *
     * @tparam TensorType1
     * @tparam TensorType2
     * @tparam InteractionMatrixType
     * @tparam ValueType
     * @param multipoles the B matrix
     * @param locals  the C matrix
     * @param A  The interaction matrix (KNM)
     * @param scale_factor the scale factor alpha
     * @param acc if true beta = 1 otherwise 0.
     */
    template<typename TensorType1, typename TensorType2, typename InteractionMatrixType, typename ValueType>
    inline auto blas3_product(TensorType1& B, TensorType2& C, InteractionMatrixType const& A, int nb_mult,
                              ValueType scale_factor, bool acc, bool transA = false) -> void
    {
        using value_type = ValueType;
        // see
        // https://netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html
        auto& shapeA = A.shape();
        // //  m specifies the number of rows
        // //  N specifies the number  of columns of the matrix B
        // io::print(std::clog, " shape(A) ", shapeA);
        // auto& shape = B.shape();
        // scalfmm::io::print(std::clog, " shape(B) ", shape);
        // std::clog << "xt::get_blas_storage_order(A), " << xt::get_blas_storage_order(A)
        //           << " xt::get_blas_storage_order(B), " << xt::get_blas_storage_order(B)
        //           << " xt::get_blas_storage_order(C), " << xt::get_blas_storage_order(C) << std::endl;
        // auto& shape1 = C.shape();
        // scalfmm::io::print(std::clog, " shape(C) ", shape1);
        // std::clog << " nb_mult " << nb_mult << "  LDA " << xt::get_leading_stride(A) << "  LDB "
        //           << xt::get_leading_stride(B) << "  LDC " << xt::get_leading_stride(C) << "  A.data_offset() "
        //           << A.data_offset() << std::endl;
        // auto LDA = xt::get_leading_stride(A);
        // std::exit(-1);
        // meta::td<TensorType1> u;
        // meta::td<TensorType2> u1;
        // C = xt::linalg::dot(A, B);
        cxxblas::gemm<xt::blas_index_t>(
          xt::get_blas_storage_order(A),   // locals.at(n)
                                           // OP en A
          (transA ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans),
          // OP en B
          cxxblas::Transpose::NoTrans,
          // M the number  of rows  of the  matrix op(A) and of the matrix C
          (transA ? static_cast<xt::blas_index_t>(shapeA[1]) : static_cast<xt::blas_index_t>(shapeA[0])),
          // N the number  of columns of the matrix op(B) and of the matrix C
          static_cast<xt::blas_index_t>(nb_mult),
          // K the number of columns of the matrix op(A) and the number of rows of the matrix op(B)
          (transA ? static_cast<xt::blas_index_t>(shapeA[0]) : static_cast<xt::blas_index_t>(shapeA[1])),
          // alpha coefficient
          scale_factor,
          // matrix A, ldA
          A.data() + A.data_offset(), xt::get_leading_stride(A),
          // Set of vector matrix B, ldB
          B.data(), static_cast<xt::blas_index_t>(xt::get_leading_stride(B)),
          // beta coefficient
          (acc ? value_type(1.) : value_type(0.)),
          //  Set of vector matrix C, ldC
          C.data(), static_cast<xt::blas_index_t>(xt::get_leading_stride(C)));
    }

}   // namespace scalfmm::tensor

#endif   // SCALFMM_UTILS_TENSOR_HPP
