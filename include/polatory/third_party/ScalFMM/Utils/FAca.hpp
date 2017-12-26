// See LICENCE file at project root
#ifndef FACA_HPP
#define FACA_HPP

#include "FGlobal.hpp"

#include "FSvd.hpp"


/**
 * @class FAca
 * 
 * @brief This class provides the function to perform fully or partially pivoted ACA.
 *  
 */
namespace FAca {


    /*! The fully pivoted adaptive cross approximation (fACA) compresses a
        far-field interaction as \f$K\sim UV^\top\f$. The fACA requires all entries
        to be computed beforehand, then the compression follows in
        \f$\mathcal{O}(2\ell^3k)\f$ operations based on the required accuracy
        \f$\varepsilon\f$. The matrix K will be destroyed as a result.

        @param[in] K far-field to be approximated
        @param[in] nx number of rows
        @param[in] ny number of cols
        @param[in] eps prescribed accuracy
        @param[out] U matrix containing \a k column vectors
        @param[out] V matrix containing \a k row vectors
        @param[out] k final low-rank depends on prescribed accuracy \a eps
     */
    template <class FReal>
    void fACA(FReal *const K,
              const unsigned int nx, const unsigned int ny,
              const double eps, FReal* &U, FReal* &V, unsigned int &k)
    {
        // control vectors (true if not used, false if used)
        bool *const r = new bool[nx];
        bool *const c = new bool[ny];
        for (unsigned int i=0; i<nx; ++i) r[i] = true;
        for (unsigned int j=0; j<ny; ++j) c[j] = true;

        // compute Frobenius norm of original Matrix K
        FReal norm2K = 0;
        for (unsigned int j=0; j<ny; ++j) {
            const FReal *const colK = K + j*nx;
            norm2K += FBlas::scpr(nx, colK, colK);
        }

        // initialize rank k and UV'
        k = 0;
        const unsigned int maxk = (nx + ny) / 2;
        U = new FReal[nx * maxk];
        V = new FReal[ny * maxk];
        FBlas::setzero(nx*maxk, U);
        FBlas::setzero(ny*maxk, V);
        FReal norm2R;

        ////////////////////////////////////////////////
        // start fully pivoted ACA
        do {

            // find max(K) and argmax(K)
            FReal maxK = 0.;
            unsigned int pi=0, pj=0;
            for (unsigned int j=0; j<ny; ++j)
                if (c[j]) {
                    const FReal *const colK = K + j*nx;
                    for (unsigned int i=0; i<nx; ++i)
                        if (r[i] && maxK < FMath::Abs(colK[i])) {
                            maxK = FMath::Abs(colK[i]);
                            pi = i; 
                            pj = j;
                        }
                }

            // copy pivot cross into U and V
            FReal *const colU = U + k*nx;
            FReal *const colV = V + k*ny;
            const FReal pivot = K[pj*nx + pi];
            for (unsigned int i=0; i<nx; ++i) if (r[i]) colU[i] = K[pj*nx + i];
            for (unsigned int j=0; j<ny; ++j) if (c[j]) colV[j] = K[j *nx + pi] / pivot;

            // don't use these cols and rows anymore
            c[pj] = false;
            r[pi] = false;

            // subtract k-th outer product from K
            for (unsigned int j=0; j<ny; ++j)
                if (c[j]) {
                    FReal *const colK = K + j*nx;
                    FBlas::axpy(nx, FReal(-1. * colV[j]), colU, colK);
                }

            // compute Frobenius norm of updated K
            norm2R = 0.0;
            for (unsigned int j=0; j<ny; ++j)
                if (c[j]) {
                    const FReal *const colK = K + j*nx;
                    norm2R += FBlas::scpr(nx, colK, colK);
                }

            // increment rank k
            ++k ;

        } while (norm2R > eps*eps * norm2K);
        ////////////////////////////////////////////////

        delete [] r;
        delete [] c;
    }











    /*!  The partially pivoted adaptive cross approximation (pACA) compresses a
        far-field interaction as \f$K\sim UV^\top\f$. The pACA computes the matrix
        entries on the fly, as they are needed. The compression follows in
        \f$\mathcal{O}(2\ell^3k)\f$ operations based on the required accuracy
        \f$\varepsilon\f$. 

        @tparam ComputerType the functor type which allows to compute matrix entries

        @param[in] Computer the entry-computer functor
        @param[in] eps prescribed accuracy
        @param[in] nx number of rows
        @param[in] ny number of cols
        @param[out] U matrix containing \a k column vectors
        @param[out] V matrix containing \a k row vectors
        @param[out] k final low-rank depends on prescribed accuracy \a eps
     */
    template <class FReal, typename ComputerType>
    void pACA(const ComputerType& Computer,
            const unsigned int nx, const unsigned int ny,
            const FReal eps, FReal* &U, FReal* &V, unsigned int &k)
    {
        // control vectors (true if not used, false if used)
        bool *const r = new bool[nx];
        bool *const c = new bool[ny];
        for (unsigned int i=0; i<nx; ++i) r[i] = true;
        for (unsigned int j=0; j<ny; ++j) c[j] = true;

        // initialize rank k and UV'
        k = 0;
        const FReal eps2 = eps * eps;
        const unsigned int maxk = (nx + ny) / 2;
        U = new FReal[nx * maxk];
        V = new FReal[ny * maxk];

        // initialize norm
        FReal norm2S(0.);
        FReal norm2uv(0.);

        ////////////////////////////////////////////////
        // start partially pivoted ACA
        unsigned int J = 0, I = 0;

        do {
            FReal *const colU = U + nx*k;
            FReal *const colV = V + ny*k;

            ////////////////////////////////////////////
            // compute row I and its residual
            Computer(I, I+1, 0, ny, colV);
            r[I] = false;
            for (unsigned int l=0; l<k; ++l) {
                FReal *const u = U + nx*l;
                FReal *const v = V + ny*l;
                FBlas::axpy(ny, FReal(-1. * u[I]), v, colV);
            }

            // find max of residual and argmax
            FReal maxval = 0.;
            for (unsigned int j=0; j<ny; ++j) {
                const FReal abs_val = FMath::Abs(colV[j]);
                if (c[j] && maxval < abs_val) {
                    maxval = abs_val;
                    J = j;
                }
            }
            // find pivot and scale column of V
            const FReal pivot = FReal(1.) / colV[J];
            FBlas::scal(ny, pivot, colV);

            ////////////////////////////////////////////
            // compute col J and its residual
            Computer(0, nx, J, J+1, colU);
            c[J] = false;
            for (unsigned int l=0; l<k; ++l) {
                FReal *const u = U + nx*l;
                FReal *const v = V + ny*l;
                FBlas::axpy(nx, FReal(-1. * v[J]), u, colU);
            }

            // find max of residual and argmax
            maxval = 0.0;
            for (unsigned int i=0; i<nx; ++i) {
                const FReal abs_val = FMath::Abs(colU[i]);
                if (r[i] && maxval < abs_val) {
                    maxval = abs_val;
                    I = i;
                }
            }

            ////////////////////////////////////////////
            // increment Frobenius norm: |Sk|^2 += |uk|^2 |vk|^2 + 2 sumj ukuj vjvk
            FReal normuuvv(0.);
            for (unsigned int l=0; l<k; ++l)
                normuuvv += FBlas::scpr(nx, colU, U + nx*l) * FBlas::scpr(ny, V + ny*l, colV);
            norm2uv = FBlas::scpr(nx, colU, colU) * FBlas::scpr(ny, colV, colV);
            norm2S += norm2uv + 2*normuuvv;

            ////////////////////////////////////////////
            // increment low-rank
            ++k;

        } while (norm2uv > eps2 * norm2S);

        delete [] r;
        delete [] c;
    }


    /*! Perform QR+SVD recompression of the U and V returned by ACA. 
        This allows to avoid potential redundancies in U and V, 
        since orthogonality is not ensured by ACA.

        @param[in] UU nx \times rank array returned by ACA
        @param[in] VV ny \times rank array returned by ACA
        @param[in] nx number of rows
        @param[in] ny number of cols
        @param[in] eps prescribed accuracy
        @param[out] K rank\times (nx+ny) array containing U and V after recompression
        @param[out] rank final low-rank depends on prescribed accuracy \a eps
     */
    template <class FReal>
    void recompress(FReal *const UU, FReal *const VV, 
                  const unsigned int nx, const unsigned int ny,
                  const double eps, FReal* &U, FReal* &VT, unsigned int &rank)
    {

        // needed for the SVD
        int INFO;

        // QR decomposition (UU=QuRu & VV=QvRv, thus UUVV^t=Qu(RuRv^t)Qv^t)
        FReal* phi = new FReal [rank*rank];
        {
            // needed for QR
            const unsigned int LWORK = 2*4*rank; // for square matrices
            //const unsigned int LWORK = 2*std::max(3*minMN+maxMN, 5*minMN);
            FReal *const WORK = new FReal [LWORK];

            // QR of U and V
            FReal* tauU = new FReal [rank];
            INFO = FBlas::geqrf(nx, rank, UU, tauU, LWORK, WORK);
            assert(INFO==0);
            FReal* tauV = new FReal [rank];
            INFO = FBlas::geqrf(ny, rank, VV, tauV, LWORK, WORK);
            assert(INFO==0);
            // phi = Ru Rv'
            FReal* rU = new FReal [2 * rank*rank];
            FReal* rV = rU + rank*rank;
            FBlas::setzero(2 * rank*rank, rU);
            for (unsigned int l=0; l<rank; ++l) {
                FBlas::copy(l+1, UU + l*nx, rU + l*rank);
                FBlas::copy(l+1, VV + l*ny, rV + l*rank);
            }
            FBlas::gemmt(rank, rank, rank, FReal(1.), rU, rank, rV, rank, phi, rank);
            delete [] rU;

            // get Qu and Qv
            INFO = FBlas::orgqr(nx, rank, UU, tauU, LWORK, WORK); // Qu -> UU
            assert(INFO==0);
            INFO = FBlas::orgqr(ny, rank, VV, tauV, LWORK, WORK); // Qv -> VV
            assert(INFO==0);
            delete [] tauU;
            delete [] tauV;
        }

        const unsigned int aca_rank = rank;

        // SVD (of phi=RuRv^t, thus phi=UrSVr^t)
        FReal *const VrT = new FReal [aca_rank*aca_rank];
        FReal *const Sr = new FReal [aca_rank];
        {
            // needed for SVD
            const unsigned int LWORK = 2*4*aca_rank; // for square matrices
            //const unsigned int LWORK = 2*std::max(3*minMN+maxMN, 5*minMN);
            FReal *const WORK = new FReal [LWORK];

            INFO = FBlas::gesvd(aca_rank, aca_rank, phi, Sr, VrT, aca_rank, LWORK, WORK); // Ur -> phi // Sr -> Sr // Vr^t -> VrT
            if (INFO!=0){
                std::stringstream stream;
                stream << INFO;
                throw std::runtime_error("SVD did not converge with " + stream.str());
            }
            rank = FSvd::getRank(Sr, aca_rank, eps);
        }                   

        // store Qu(UrS) in U and Vr^t(Qv^t) in VT
        {
            // allocate
            assert(U ==nullptr);
            assert(VT==nullptr);
            U  = new FReal[rank*nx];
            VT = new FReal[rank*ny];

            // (Ur Sigma)
            for (unsigned int r=0; r<rank; ++r)
                FBlas::scal(aca_rank, Sr[r], phi + r*aca_rank);
            delete [] Sr;

            // Qu (Ur Sr) in U
            FBlas::gemm(nx, aca_rank, rank, FReal(1.), UU, nx, phi, aca_rank, U, nx);
            delete [] phi;

            // VT=Vr^t Qv^t
            FBlas::gemmt(rank, aca_rank, ny, FReal(1.), VrT, aca_rank, VV, ny, VT, rank);
            delete [] VrT;
            
        }

    }





};

#endif /* FACA_HPP */