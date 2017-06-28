// ===================================================================================
// Copyright ScalFmm 2016 INRIA, Olivier Coulaud, Bérenger Bramas,
// Matthias Messner olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the
// FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
// An extension to the license is given to allow static linking of scalfmm
// inside a proprietary application (no matter its license).
// See the main license file for more details.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================
#ifndef FCHEBTENSORIALM2LHANDLER_HPP
#define FCHEBTENSORIALM2LHANDLER_HPP

#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>
#include <typeinfo>

#include "../../Utils/FBlas.hpp"
#include "../../Utils/FTic.hpp"

#include "./FChebTensor.hpp"

/**
 * Computes and compresses all \f$K_t\f$.
 *
 * @param[in] epsilon accuracy
 * @param[out] U matrix of size \f$\ell^3\times r\f$
 * @param[out] C matrix of size \f$r\times 316 r\f$ storing \f$[C_1,\dots,C_{316}]\f$
 * @param[out] B matrix of size \f$\ell^3\times r\f$
 */
template <class FReal, int ORDER, class MatrixKernelClass>
unsigned int ComputeAndCompress(const MatrixKernelClass *const MatrixKernel, 
                                const FReal CellWidth, 
                                const FReal epsilon,
                                FReal* &U,
                                FReal** &C,
                                FReal* &B);

//template <int ORDER>
//unsigned int Compress(const FReal epsilon, const unsigned int ninteractions,
//                                          FReal* &U,  FReal* &C, FReal* &B);


/**
 * @author Matthias Messner (matthias.messner@inria.fr)
 * @class FChebM2LHandler
 * Please read the license
 *
 * This class precomputes and compresses the M2L operators
 * \f$[K_1,\dots,K_{316}]\f$ for all (\f$7^3-3^3 = 316\f$ possible interacting
 * cells in the far-field) interactions for the Chebyshev interpolation
 * approach. The class uses the compression via a truncated SVD and represents
 * the compressed M2L operator as \f$K_t \sim U C_t B^\top\f$ with
 * \f$t=1,\dots,316\f$. The truncation rank is denoted by \f$r\f$ and is
 * determined by the prescribed accuracy \f$\varepsilon\f$. Hence, the
 * originally \f$K_t\f$ of size \f$\ell^3\times\ell^3\f$ times \f$316\f$ for
 * all interactions is reduced to only one \f$U\f$ and one \f$B\f$, each of
 * size \f$\ell^3\times r\f$, and \f$316\f$ \f$C_t\f$, each of size \f$r\times
 * r\f$.
 *
 * PB: BEWARE! Homogeneous matrix kernels do not support cell width extension
 * yet. Is it possible to find a reference width and a scale factor such that
 * only 1 set of M2L ops can be used for all levels?? 
 *
 * @tparam ORDER interpolation order \f$\ell\f$
 */
template <class FReal, int ORDER, class MatrixKernelClass, KERNEL_FUNCTION_TYPE TYPE> class FChebTensorialM2LHandler;

template <class FReal, int ORDER, class MatrixKernelClass>
class FChebTensorialM2LHandler<FReal, ORDER,MatrixKernelClass,HOMOGENEOUS> : FNoCopyable
{
    enum {order = ORDER,
          nnodes = TensorTraits<ORDER>::nnodes,
          ninteractions = 316,// 7^3 - 3^3 (max num cells in far-field)
          ncmp = MatrixKernelClass::NCMP};

    FReal *U, *B;
    FReal** C;

    const FReal CellWidthExtension; //<! extension of cells width

    const FReal epsilon; //<! accuracy which determines trucation of SVD
    unsigned int rank;   //<! truncation rank, satisfies @p epsilon


    static const std::string getFileName(FReal epsilon)
    {
        const char precision_type = (typeid(FReal)==typeid(double) ? 'd' : 'f');
        std::stringstream stream;
        stream << "m2l_k"<< MatrixKernelClass::getID() << "_" << precision_type
                     << "_o" << order << "_e" << epsilon << ".bin";
        return stream.str();
    }

    
public:
    FChebTensorialM2LHandler(const MatrixKernelClass *const MatrixKernel, const unsigned int, const FReal, const FReal inCellWidthExtension, const FReal _epsilon)
        : U(nullptr), B(nullptr), CellWidthExtension(inCellWidthExtension), 
          epsilon(_epsilon), rank(0)
    {
        // measure time
        FTic time; time.tic();
        // check if already set
        if (U||B) throw std::runtime_error("U or B operator already set");

        // allocate C
        C = new FReal*[ncmp];
        for (unsigned int d=0; d<ncmp; ++d) C[d]=nullptr;

        for (unsigned int d=0; d<ncmp; ++d)
            if (C[d]) throw std::runtime_error("Compressed M2L operator already set");

        // Compute matrix of interactions
        // reference cell width is arbitrarly set to 2. 
        // but it NEEDS to match the numerator of the scale factor in matrix kernel!
        // Therefore box width extension is not yet supported for homog kernels
        const FReal ReferenceCellWidth = FReal(2.);
        rank = ComputeAndCompress<order>(MatrixKernel, ReferenceCellWidth, 0., epsilon, U, C, B);

        unsigned long sizeM2L = 343*ncmp*rank*rank*sizeof(FReal);


        // write info
        std::cout << "Compressed and set M2L operators (" << long(sizeM2L) << " B) in "
                  << time.tacAndElapsed() << "sec."   << std::endl;
    }

    ~FChebTensorialM2LHandler()
    {
        if (U != nullptr) delete [] U;
        if (B != nullptr) delete [] B;
        for (unsigned int d=0; d<ncmp; ++d)
            if (C[d] != nullptr) delete [] C[d];
    }

    /**
     * @return rank of the SVD compressed M2L operators
     */
    unsigned int getRank() const {return rank;}

    /**
     * Expands potentials \f$x+=UX\f$ of a target cell. This operation can be
     * seen as part of the L2L operation.
     *
     * @param[in] X compressed local expansion of size \f$r\f$
     * @param[out] x local expansion of size \f$\ell^3\f$
     */
    void applyU(const FReal *const X, FReal *const x) const
    {
//    FBlas::gemva(nnodes, rank, 1., U, const_cast<FReal*>(X), x);
        FBlas::add(nnodes, const_cast<FReal*>(X), x);
    }

    /**
     * Compressed M2L operation \f$X+=C_tY\f$, where \f$Y\f$ is the compressed
     * multipole expansion and \f$X\f$ is the compressed local expansion, both
     * of size \f$r\f$. The index \f$t\f$ denotes the transfer vector of the
     * target cell to the source cell.
     *
     * @param[in] transfer transfer vector
     * @param[in] Y compressed multipole expansion
     * @param[out] X compressed local expansion
     * @param[in] CellWidth needed for the scaling of the compressed M2L operators which are based on a homogeneous matrix kernel computed for the reference cell width \f$w=2\f$, ie in \f$[-1,1]^3\f$.
     */
    void applyC(const unsigned int idx, const unsigned int , 
                const FReal scale, const unsigned int d,
                const FReal *const Y, FReal *const X) const
    {
        FBlas::gemva(rank, rank, scale, C[d] + idx*rank*rank, const_cast<FReal*>(Y), X);
    }

    /**
     * Compresses densities \f$Y=B^\top y\f$ of a source cell. This operation
     * can be seen as part of the M2M operation.
     *
     * @param[in] y multipole expansion of size \f$\ell^3\f$
     * @param[out] Y compressed multipole expansion of size \f$r\f$
     */
    void applyB(FReal *const y, FReal *const Y) const
    {
//    FBlas::gemtv(nnodes, rank, 1., B, y, Y);
        FBlas::copy(nnodes, y, Y);
    }


};


template <class FReal, int ORDER, class MatrixKernelClass>
class FChebTensorialM2LHandler<FReal,ORDER,MatrixKernelClass,NON_HOMOGENEOUS> : FNoCopyable
{
    enum {order = ORDER,
          nnodes = TensorTraits<ORDER>::nnodes,
          ninteractions = 316,// 7^3 - 3^3 (max num cells in far-field)
          ncmp = MatrixKernelClass::NCMP};

    // Tensorial MatrixKernel and homogeneity specific
    FReal **U, **B;
    FReal*** C;

    const unsigned int TreeHeight; //<! number of levels
    const FReal RootCellWidth; //<! width of root cell
    const FReal CellWidthExtension; //<! extension of cells width

    const FReal epsilon; //<! accuracy which determines trucation of SVD
    unsigned int *rank;   //<! truncation rank, satisfies @p epsilon


    static const std::string getFileName(FReal epsilon)
    {
        const char precision_type = (typeid(FReal)==typeid(double) ? 'd' : 'f');
        std::stringstream stream;
        stream << "m2l_k"<< MatrixKernelClass::getID() << "_" << precision_type
                     << "_o" << order << "_e" << epsilon << ".bin";
        return stream.str();
    }

    
public:
    FChebTensorialM2LHandler(const MatrixKernelClass *const MatrixKernel, const unsigned int inTreeHeight, const FReal inRootCellWidth, const FReal inCellWidthExtension, const FReal _epsilon)
        : TreeHeight(inTreeHeight),
          RootCellWidth(inRootCellWidth),
          CellWidthExtension(inCellWidthExtension),
          epsilon(_epsilon)
    {
        // measure time
        FTic time; time.tic();

        // allocate rank
        rank = new unsigned int[TreeHeight];

        // allocate U and B
        U = new FReal*[TreeHeight]; 
        B = new FReal*[TreeHeight]; 
        for (unsigned int l=0; l<TreeHeight; ++l){
           B[l]=nullptr; U[l]=nullptr;
        }

        // allocate C
        C = new FReal**[TreeHeight];
        for (unsigned int l=0; l<TreeHeight; ++l){ 
            C[l] = new FReal*[ncmp];
            for (unsigned int d=0; d<ncmp; ++d)
                C[l][d]=nullptr;
        }

        for (unsigned int l=0; l<TreeHeight; ++l) {
            if (U[l] || B[l]) throw std::runtime_error("Operator U or B already set");
            for (unsigned int d=0; d<ncmp; ++d)
                if (C[l][d]) throw std::runtime_error("Compressed M2L operator already set");
        }

        // Compute matrix of interactions at each level !! (since non homog)
        FReal CellWidth = RootCellWidth / FReal(2.); // at level 1
        CellWidth /= FReal(2.);                      // at level 2
        rank[0]=rank[1]=0;
        for (unsigned int l=2; l<TreeHeight; ++l) {
            // compute m2l operator on extended cell
            rank[l] = ComputeAndCompress<FReal, order>(MatrixKernel, CellWidth, CellWidthExtension, epsilon, U[l], C[l], B[l]);
            // update cell width
            CellWidth /= FReal(2.);                    // at level l+1 
        }
        unsigned long sizeM2L = (TreeHeight-2)*343*ncmp*rank[2]*rank[2]*sizeof(FReal);

        // write info
        std::cout << "Compute and Set M2L operators of " << TreeHeight-2 << " levels ("<< long(sizeM2L/**1e-6*/) <<" Bytes) in "
                                << time.tacAndElapsed() << "sec."   << std::endl;
    }

    ~FChebTensorialM2LHandler()
    {
        if (rank != nullptr) delete [] rank;
        for (unsigned int l=0; l<TreeHeight; ++l) {
            if (U[l] != nullptr) delete [] U[l];
            if (B[l] != nullptr) delete [] B[l];
            for (unsigned int d=0; d<ncmp; ++d)
                if (C[l][d] != nullptr) delete [] C[l][d];
        }
    }

    /**
     * @return rank of the SVD compressed M2L operators
     */
    unsigned int getRank(unsigned int l = 2) const {return rank[l];}

    /**
     * Expands potentials \f$x+=UX\f$ of a target cell. This operation can be
     * seen as part of the L2L operation.
     *
     * @param[in] X compressed local expansion of size \f$r\f$
     * @param[out] x local expansion of size \f$\ell^3\f$
     */
    void applyU(const FReal *const X, FReal *const x) const
    {
//    FBlas::gemva(nnodes, rank, 1., U, const_cast<FReal*>(X), x);
        FBlas::add(nnodes, const_cast<FReal*>(X), x);
    }

    /**
     * Compressed M2L operation \f$X+=C_tY\f$, where \f$Y\f$ is the compressed
     * multipole expansion and \f$X\f$ is the compressed local expansion, both
     * of size \f$r\f$. The index \f$t\f$ denotes the transfer vector of the
     * target cell to the source cell.
     *
     * @param[in] transfer transfer vector
     * @param[in] Y compressed multipole expansion
     * @param[out] X compressed local expansion
     * @param[in] CellWidth needed for the scaling of the compressed M2L operators which are based on a homogeneous matrix kernel computed for the reference cell width \f$w=2\f$, ie in \f$[-1,1]^3\f$.
     */
    void applyC(const unsigned int idx, const unsigned int l, 
                const FReal, const unsigned int d,
                const FReal *const Y, FReal *const X) const
    {
        FBlas::gemva(rank[l], rank[l], 1., C[l][d] + idx*rank[l]*rank[l], const_cast<FReal*>(Y), X);
    }

    /**
     * Compresses densities \f$Y=B^\top y\f$ of a source cell. This operation
     * can be seen as part of the M2M operation.
     *
     * @param[in] y multipole expansion of size \f$\ell^3\f$
     * @param[out] Y compressed multipole expansion of size \f$r\f$
     */
    void applyB(FReal *const y, FReal *const Y) const
    {
//    FBlas::gemtv(nnodes, rank, 1., B, y, Y);
        FBlas::copy(nnodes, y, Y);
    }


};





//////////////////////////////////////////////////////////////////////
// definition ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////






template <class FReal, int ORDER, class MatrixKernelClass>
unsigned int ComputeAndCompress(const MatrixKernelClass *const MatrixKernel, 
                                const FReal CellWidth, 
                                const FReal CellWidthExtension, 
                                const FReal /*epsilon*/,
                                FReal* &U,
                                FReal** &C,
                                FReal* &B)
{
    // PB: need to redefine some constant since not function from m2lhandler class
    const unsigned int order = ORDER;
    const unsigned int nnodes = TensorTraits<ORDER>::nnodes;
    const unsigned int ninteractions = 316;
    const unsigned int ncmp = MatrixKernelClass::NCMP;

    // allocate memory and store compressed M2L operators
    for (unsigned int d=0; d<ncmp; ++d) 
    if (C[d]) throw std::runtime_error("Compressed M2L operators are already set");

    // interpolation points of source (Y) and target (X) cell
    FPoint<FReal> X[nnodes], Y[nnodes];
    // set roots of target cell (X)
    const FReal ExtendedCellWidth(CellWidth+CellWidthExtension);
    FChebTensor<FReal,order>::setRoots(FPoint<FReal>(0.,0.,0.), ExtendedCellWidth, X);

    // allocate memory and compute 316 m2l operators
    FReal** _C; 
    _C  = new FReal* [ncmp];
    for (unsigned int d=0; d<ncmp; ++d) 
    _C[d] = new FReal [nnodes*nnodes * ninteractions];

    unsigned int counter = 0;
    for (int i=-3; i<=3; ++i) {
        for (int j=-3; j<=3; ++j) {
            for (int k=-3; k<=3; ++k) {
                if (abs(i)>1 || abs(j)>1 || abs(k)>1) {
                    // set roots of source cell (Y)
                    const FPoint<FReal> cy(CellWidth*FReal(i), CellWidth*FReal(j), CellWidth*FReal(k));
                    FChebTensor<FReal,order>::setRoots(cy, ExtendedCellWidth, Y);

                    // evaluate m2l operator
                    for (unsigned int n=0; n<nnodes; ++n)
                        for (unsigned int m=0; m<nnodes; ++m){

                            // Compute current M2L interaction (block matrix)
                            FReal* block; 
                            block = new FReal[ncmp]; 
                            MatrixKernel->evaluateBlock(X[m], Y[n], block);

                            // Copy block in C
                            for (unsigned int d=0; d<ncmp; ++d) 
                                _C[d][counter*nnodes*nnodes + n*nnodes + m] = block[d];

                            delete [] block;
              
                        }

                    // increment interaction counter
                    counter++;
                }
            }
        }
    }
    if (counter != ninteractions)
        throw std::runtime_error("Number of interactions must correspond to 316");


    //  //////////////////////////////////////////////////////////      
    //  FReal weights[nnodes];
    //  FChebTensor<FReal,order>::setRootOfWeights(weights);
    //  for (unsigned int i=0; i<316; ++i)
    //      for (unsigned int n=0; n<nnodes; ++n) {
    //          FBlas::scal(nnodes, weights[n], _C+i*nnodes*nnodes + n,  nnodes); // scale rows
    //          FBlas::scal(nnodes, weights[n], _C+i*nnodes*nnodes + n * nnodes); // scale cols
    //      }
    //  //////////////////////////////////////////////////////////      

    // svd compression of M2L
    //  const unsigned int rank = Compress<ORDER>(epsilon, ninteractions, _U, _C, _B);
    const unsigned int rank   = nnodes; //PB: dense Chebyshev
    if (!(rank>0)) throw std::runtime_error("Low rank must be larger then 0!");

    // store U
    U = new FReal [nnodes * rank];
    //  FBlas::copy(rank*nnodes, _U, U);
    FBlas::setzero(rank*nnodes, U);
    //  delete [] _U;
    // store B
    B = new FReal [nnodes * rank];
    //  FBlas::copy(rank*nnodes, _B, B);
    FBlas::setzero(rank*nnodes, B);
    //  delete [] _B;

    // store C
    counter = 0;
    for (unsigned int d=0; d<ncmp; ++d) 
    C[d] = new FReal [343 * rank*rank];
    for (int i=-3; i<=3; ++i)
        for (int j=-3; j<=3; ++j)
            for (int k=-3; k<=3; ++k) {
                const unsigned int idx = (i+3)*7*7 + (j+3)*7 + (k+3);
                if (abs(i)>1 || abs(j)>1 || abs(k)>1) {
                    for (unsigned int d=0; d<ncmp; ++d) 
                        FBlas::copy(rank*rank, _C[d] + counter*rank*rank, C[d] + idx*rank*rank);
                    counter++;
                } else {
                    for (unsigned int d=0; d<ncmp; ++d) 
                        FBlas::setzero(rank*rank, C[d] + idx*rank*rank);
                }
            }
    if (counter != ninteractions)
        throw std::runtime_error("Number of interactions must correspond to 316");
    for (unsigned int d=0; d<ncmp; ++d) 
    delete [] _C[d];


    //  //////////////////////////////////////////////////////////      
    //  for (unsigned int n=0; n<nnodes; ++n) {
    //      FBlas::scal(rank, FReal(1.) / weights[n], U+n, nnodes); // scale rows
    //      FBlas::scal(rank, FReal(1.) / weights[n], B+n, nnodes); // scale rows
    //  }
    //  //////////////////////////////////////////////////////////      

    // return low rank
    return rank;
}




#endif // FCHEBTENSORIALM2LHANDLER_HPP

// [--END--]
