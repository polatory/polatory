// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/low_rank.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_LOW_RANK_HPP
#define SCALFMM_UTILS_LOW_RANK_HPP

#include <array>
#include <cpp_tools/colors/colorized.hpp>
#include <xtensor/containers/xtensor.hpp>

namespace scalfmm::low_rank
{
    /**
     * @brief Partial Adaptive cross approximation
     *
     *  The partially pivoted adaptive cross approximation (pACA) compresses a
     *   matrix kernel interaction as \f$K\sim UV^\top\f$. The pACA computes the matrix
     *   entries on the fly, as they are needed. The compression follows in
     *   \f$\mathcal{O}(2\ell^3k)\f$ operations based on the required accuracy
     *   \f$\varepsilon\f$.
     *
     * @tparam ValueType
     * @tparam MatrixKernelType
     * @tparam TensorViewX
     * @tparam TensorViewY
     * @param[in] mk the matrix kernel to be approximated
     * @param[in] X the points involved in the rows
     * @param[in] Y  the points involved in the column
     * @param[in] weights  the weights  
     * @param[in] epsilon the prescribed accuracy for low-rank approximation
     * @param[in] kn the number of row of the matrix kernel
     * @param[in] km the number of column of the matrix kernel
     * @return std::tuple<xt::xtensor<ValueType, 2>, xt::xtensor<ValueType, 2>>  containing \a k column vectors and  k row vectors
     */
    template<typename ValueType, typename MatrixKernelType, typename TensorViewX, typename TensorViewY>
    inline auto paca(MatrixKernelType const& mk, TensorViewX&& X, TensorViewY&& Y, xt::xarray<ValueType> const& weights,
                     ValueType epsilon, std::size_t kn, std::size_t km)
      -> std::tuple<xt::xtensor<ValueType, 2>, xt::xtensor<ValueType, 2>>
    {
        const std::size_t nnodes{X.size()};
        std::vector<bool> row_bools(nnodes, true);
        std::vector<bool> col_bools(nnodes, true);
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> U;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> V;

        // initialize rank r
        std::size_t r{0};
        const auto max_r{nnodes};

        // resize U V
        // TODO kn km
        U.resize({nnodes, max_r});
        V.resize({nnodes, max_r});

        // initialize norm
        ValueType norm2s{0.};
        ValueType norm2uv{0.};
        // start partially pivoted aca
        auto evaluate = [&weights, &mk, &X, &Y, kn, km](auto& view, auto Ib, auto Ie, auto Jb, auto Je)
        {
            std::size_t idx{0};
            for(std::size_t j = Jb; j < Je; ++j)
            {
                for(std::size_t i = Ib; i < Ie; ++i)
                {
#ifdef LOW_RANK_NO_WEIGHTS
                    view.at(idx) = mk.evaluate(X[i], Y[j]).at(kn * MatrixKernelType::km + km);
#else
                    view.at(idx) = weights[i] * weights[j] * mk.evaluate(X[i], Y[j]).at(kn * MatrixKernelType::km + km);

#endif
                    ++idx;
                }
            }
        };
        const auto epsilon2 = epsilon * epsilon;
        std::size_t I{0};
        std::size_t J{0};
        do
        {
            // compute row I and its residual
            auto V_ = xt::view(V, xt::all(), r);
            evaluate(V_, I, I + 1, 0, nnodes);
            row_bools[I] = false;
            for(std::size_t l = 0; l < r; ++l)
            {
                auto col_u = xt::view(U, xt::all(), l);
                auto col_v = xt::view(V, xt::all(), l);
                V_ -= col_u.at(I) * col_v;
            }
            // find max of residual and argmax
            ValueType val_max{0.};
            for(std::size_t j = 0; j < nnodes; ++j)
            {
                const auto val_abs = std::abs(V_.at(j));
                if(col_bools[j] && val_max < val_abs)
                {
                    val_max = val_abs;
                    J = j;
                }
            }
            // find pivot and scale column of V
            const ValueType pivot = ValueType(1.) / V_.at(J);
            V_ *= pivot;

            // compute col J and its residual
            auto U_ = xt::view(U, xt::all(), r);
            evaluate(U_, 0, nnodes, J, J + 1);
            col_bools[J] = false;
            for(std::size_t l = 0; l < r; ++l)
            {
                auto col_u = xt::view(U, xt::all(), l);
                auto col_v = xt::view(V, xt::all(), l);
                U_ -= col_v.at(J) * col_u;
            }
            // find max of residual and argmax
            val_max = 0.;
            for(std::size_t i = 0; i < nnodes; ++i)
            {
                const auto val_abs = std::abs(U_.at(i));
                if(row_bools[i] && val_max < val_abs)
                {
                    val_max = val_abs;
                    I = i;
                }
            }
            // increment Frobenius norm: |Sk|^2 += |uk|^2 |vk|^2 + 2 sumj ukuj vjvk
            ValueType normuuvv{0.};
            for(std::size_t l = 0; l < r; ++l)
            {
                auto col_u = xt::view(U, xt::all(), l);
                auto col_v = xt::view(V, xt::all(), l);
                normuuvv += xt::linalg::vdot(U_, col_u) * xt::linalg::vdot(col_v, V_);
            }
            norm2uv = xt::linalg::vdot(U_, U_) * xt::linalg::vdot(V_, V_);
            norm2s += norm2uv + ValueType(2.) * normuuvv;
            // increment low-rank
            ++r;
        } while(norm2uv > epsilon2 * norm2s && r < max_r);

        auto UU = xt::eval(xt::view(U, xt::all(), xt::range(0, r)));
        auto VV = xt::eval(xt::view(V, xt::all(), xt::range(0, r)));
        return std::make_tuple(UU, VV);
    }
    template<typename MatrixType, typename ValueType, typename IntType>
    static IntType get_numerical_rank(MatrixType const& S, const IntType size, const ValueType eps)
    {
        const ValueType nrm2 = xt::linalg::vdot(S, S);
        const ValueType neta2 = eps * eps * nrm2;

        ValueType cumulativeSum{nrm2};
        int rank = 0;

        for(int i = 0; i < S.size(); i++)
        {
            cumulativeSum -= S[i] * S[i];
            ++rank;

            if(cumulativeSum < neta2)
            {
                // std::cout << i << "   threshold " << neta2 << "  cumulativeSum " << cumulativeSum << std::endl;
                // std::cout << "              rank " << rank << "  S.size() " << S.size() << std::endl;
                break;
            }
        }
        return rank;   // std::min(rank , int(S.size()));
    }

    template<typename MatrixType, typename ValueType>
    void check_reconstruction(MatrixType const& K, const MatrixType& U, const MatrixType& V, const ValueType epsilon,
                              std::string help, bool stop = false)
    {
        auto nnodes = K.shape()[0];
        auto normK{xt::linalg::norm(K)};
        xt::xarray<ValueType> prod(std::vector(2, nnodes));
        xt::blas::gemm(U, V, prod, false, true);

        auto error = K - prod;
        auto norm{xt::linalg::norm(error)};
        auto error_recom{norm / normK};

        if(error_recom > epsilon)
        {
            std::cout << cpp_tools::colors::red << help << " error rank " << U.shape()[1] << " rel error "
                      << error_recom << " epsilon " << epsilon << "  ratio  " << error_recom / epsilon << std::endl
                      << cpp_tools::colors::reset;
            if(stop)
            {
                std::stringstream ss{};
                ss << error_recom << " > " << epsilon;
                throw std::runtime_error("error in svd: " + ss.str());
            }
        }
        // else
        // {
        //     std::cout << "error rank " << U.shape()[1] << " rel error " << error_recom << " epsilon " << epsilon
        //               << std::endl;
        // }
    }
    /**
     * @brief truncated SVD approximation
     *
     *  The singular value decomposition) compresses a
     *   matrix kernel interaction as \f$K\sim UV^\top\f$ with the required accuracy
     *   \f$\varepsilon\f$.
     *
     * @tparam ValueType
     * @tparam MatrixKernelType
     * @tparam TensorViewX
     * @tparam TensorViewY
     * @param[in] mk the matrix kernel to be approximated
     * @param[in] X the points involved in the rows
     * @param[in] Y  the points involved in the column
     * @param[in] weights  the weights  
     * @param[in] epsilon the prescribed accuracy for low-rank approximation
     * @param[in] kn the number of row of the matrix kernel
     * @param[in] km the number of column of the matrix kernel
     * @return std::tuple<xt::xtensor<ValueType, 2>, xt::xtensor<ValueType, 2>>  containing \a k column vectors and  k row vectors
     */

    template<typename ValueType, typename MatrixKernelType, typename TensorViewX, typename TensorViewY>
    inline auto tsvd(MatrixKernelType const& mk, TensorViewX&& X, TensorViewY&& Y, xt::xarray<ValueType> const& weights,
                     ValueType epsilon, std::size_t kn, std::size_t km)
      -> std::tuple<xt::xtensor<ValueType, 2>, xt::xtensor<ValueType, 2>>
    {
        using matrix_tye = xt::xtensor<ValueType, 2, xt::layout_type::column_major>;
        const std::size_t nnodes{X.size()};
        matrix_tye K;
        K.resize({nnodes, nnodes});

        // xt::xarray<ValueType> K(std::vector<int>(2, nnodes));
        ValueType normW{1};

        for(std::size_t i{0}; i < nnodes; ++i)
        {
            for(std::size_t j{0}; j < nnodes; ++j)
            {
// normW += weights[i] * weights[i];
#ifdef LOW_RANK_NO_WEIGHTS
                K.at(i, j) = mk.evaluate(X[i], Y[j]).at(kn * MatrixKernelType::km + km);
#else
                K.at(i, j) = weights[i] * weights[j] * mk.evaluate(X[i], Y[j]).at(kn * MatrixKernelType::km + km);

#endif
            }
        }
        // std::cout << " normW " << normW << " epsilon " << epsilon << " new eps/normW2 " << epsilon / normW << std::endl;
        auto eps{epsilon / normW};

        // res_svd = (U, S, Vt)
        auto res_svd = xt::linalg::svd(K, false, true);
        auto& S = std::get<1>(res_svd);
        // std::cout << "S " << S << std::endl;
        auto rank = get_numerical_rank(S, nnodes, eps);
        //
        // std::cout << "numerical_rank: " << rank << std::endl;
        auto& UU = std::get<0>(res_svd);
        auto& VT = std::get<2>(res_svd);   // is VT

        auto U = xt::eval(xt::view(UU, xt::all(), xt::range(0, rank)));
        auto V = xt::eval(xt::view(VT, xt::range(0, rank), xt::all()));
        for(std::size_t j = 0; j < rank; ++j)
        {
            for(std::size_t i = 0; i < nnodes; ++i)
            {
                U.at(i, j) *= S.at(j);
            }
        }
        return std::make_tuple(U, xt::eval(xt::transpose(V)));
    }

    /**
     * @brief Generate the low rank approximation of the matix kernel
     *
     *  use a partial adaptive cross approximation
     *   followed by  QR+SVD recompression of the U and V returned by ACA.
     *   This allows to avoid potential redundancies in U and V,
     *   since orthogonality is not ensured by ACA.

     * @tparam ValueType
     * @tparam MatrixKernelType
     * @tparam TensorViewX
     * @tparam TensorViewY
     * @param[in] mk the matrix kernel to be approximated
     * @param[in] X the points involved in the rows
     * @param[in] Y  the points involved in the column
     * @param[in] weights  the weights  
     * @param[in] epsilon the prescribed accuracy for low-rank approximation
     * @param[in] kn the number of row of the matrix kernel
     * @param[in] km the number of column of the matrix kernel

     * @return std::tuple<xt::xtensor<ValueType, 2>, xt::xtensor<ValueType, 2>> U, V the low rank approximation of mk
     */
    template<typename ValueType, typename MatrixKernelType, typename TensorViewX, typename TensorViewY>
    inline auto generate(MatrixKernelType const& mk, TensorViewX&& X, TensorViewY&& Y,
                         xt::xarray<ValueType> const& weights, ValueType epsilon, std::size_t kn, std::size_t km)
      -> std::tuple<xt::xtensor<ValueType, 2>, xt::xtensor<ValueType, 2>>
    {
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> U;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> V;
        //
#ifdef LOW_RANK_USE_SVD
        //  std::cout << " SVD " << std::endl;

        //  return U and V such that K = U VT
        std::tie(U, V) = tsvd(mk, std::forward<TensorViewX>(X), std::forward<TensorViewY>(Y), weights, epsilon, kn, km);

        auto nnodes = U.shape()[0];
        auto rank = U.shape()[1];

#else
        //  std::cout << " ACA " << std::endl;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> QU;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> QV;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> RU;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> RV;
        std::tie(U, V) = paca(mk, std::forward<TensorViewX>(X), std::forward<TensorViewY>(Y), weights, epsilon, kn, km);

        //  Perform QR+SVD recompression of the U and V returned by ACA.
        // This allows to avoid potential redundancies in U and V,
        // since orthogonality is not ensured by ACA.

        std::tie(QU, RU) = xt::linalg::qr(U);
        std::tie(QV, RV) = xt::linalg::qr(V);

        auto nnodes = U.shape()[0];
        auto rank = U.shape()[1];

        xt::xtensor<ValueType, 2, xt::layout_type::column_major> phi({rank, rank});
        xt::blas::gemm(RU, RV, phi, false, true);

        auto res_svd = xt::linalg::svd(phi);

        auto U_ = std::get<0>(res_svd);
        auto S = std::get<1>(res_svd);
        auto V_ = std::get<2>(res_svd);

        xt::blas::gemm(QV, V_, V, false, true);

        for(std::size_t j = 0; j < rank; ++j)
        {
            for(std::size_t i = 0; i < rank; ++i)
            {
                U_.at(i, j) *= S.at(j);
            }
        }
        xt::blas::gemm(QU, U_, U, false, false);

#endif
        // std::cout << " compression rank " << rank << " epsilon " << epsilon << std::endl;   // unweightening
        for(std::size_t j = 0; j < rank; ++j)
        {
            for(std::size_t i = 0; i < nnodes; ++i)
            {
                U.at(i, j) /= weights[i];
                V.at(i, j) /= weights[i];
            }
        }

        xt::xtensor<ValueType, 2> U_row(U);
        xt::xtensor<ValueType, 2> V_row(V);
#ifdef LOW_RANK_CHECK
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> K;
        K.resize({nnodes, nnodes});
        for(std::size_t i{0}; i < nnodes; ++i)
        {
            for(std::size_t j{0}; j < nnodes; ++j)
            {
                K.at(i, j) = mk.evaluate(X[i], Y[j]).at(kn * MatrixKernelType::km + km);
            }
        }
        check_reconstruction(K, U, V, epsilon, "kn=" + std::to_string(kn) + "km=" + std::to_string(km));
#endif
        return std::make_tuple(U_row, V_row);
    }

}   // namespace scalfmm::low_rank

#endif   //SCALFMM_UTILS_LOW_RANK_HPP
