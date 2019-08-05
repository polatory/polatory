// See LICENCE file at project root
#ifndef FINTERPMATRIXKERNEL_TENSORIAL_INTERACTIONS_HPP
#define FINTERPMATRIXKERNEL_TENSORIAL_INTERACTIONS_HPP

#include <iostream>
#include <stdexcept>

#include "Utils/FPoint.hpp"
#include "Utils/FNoCopyable.hpp"
#include "Utils/FMath.hpp"
#include "Utils/FGlobal.hpp"

#include "FInterpMatrixKernel.hpp"


/**
 *
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FInterpMatrixKernels_TensorialInteractions
 * Please read the license
 *
 * This class provides the evaluators and scaling functions of the matrix
 * kernels. A matrix kernel should be understood in the sense of a kernel
 * of interaction (or the fundamental solution of a given equation).
 * It can either be scalar (NCMP=1) or tensorial (NCMP>1) depending on the
 * dimension of the equation considered. NCMP denotes the number of components
 * that are actually stored (e.g. 6 for a \f$3\times3\f$ symmetric tensor).
 * 
 * Notes on the application scheme:
 * Let there be a kernel \f$K\f$ such that \f$P_i(X)X=K_{ij}(X,Y)W_j(Y)\f$
 * with \f$P\f$ the lhs of size NLHS and \f$W\f$ the rhs of size NRHS.
 * The table applyTab provides the indices in the reduced storage table
 * corresponding to the application scheme depicted earlier.
 *
 * \warning BEWARE! Homogeneous matrix kernels do not support cell width extension
 * yet. Is it possible to find a reference width and a scale factor such that
 * only 1 set of M2L opt can be used for all levels??
 * The definition of the potential p and force f are extended to the case
* of tensorial interaction kernels:
* 
*\f$ p_i(x) = K_{ip}(x,y)w_p(y),\f$    \f$ \forall i=1..NPOT, p=1..NPV\f$
*
* \f$f_{ik}= w_p(x)K_{ip,k}(x,y)w_p(y)\f$
*
* Since the interpolation scheme is such that
*
*\f$ p_i(x) \approx S^m(x) L^{m}_{ip}\f$
*
* \f$f_{ik}= w_p(x) \nabla_k S^m(x) L^{m}_{ip}\f$
*
* with
*
* \f$  L^{m}_{ip} = K^{mn}_{ip} S^n(y) w_p(y)\f$  (local expansion) 
*
 *\f$ M^{m}_{p} = S^n(y) w_p(y)\f$  (multipole expansion)  
*
* then the multipole exp have NPV components and the local exp NPOT*NPV.
*
* NB1: Only the computation of forces requires that the sum over p is 
* performed at L2P step. It could be done at M2L step for the potential.
*
* NB2: An efficient application of the matrix kernel is highly kernel 
* dependent, we recommand overriding the P2M/M2L/L2P function of the kernel 
* you are using in order to have opsimal performances + set your own NRHS/NLHS.*
 */


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
// Tensorial Matrix Kernels (NCMP>1)
//
// The definition of the potential p and force f are extended to the case
// of tensorial interaction kernels:
// p_i(x) = K_{ip}(x,y)w_p(y), \forall i=1..NPOT, p=1..NPV
// f_{ik}= w_p(x)K_{ip,k}(x,y)w_p(y) "
//
// Since the interpolation scheme is such that
// p_i(x) \approx S^m(x) L^{m}_{ip}
// f_{ik}= w_p(x) \nabla_k S^m(x) L^{m}_{ip}
// with
// L^{m}_{ip} = K^{mn}_{ip} S^n(y) w_p(y) (local expansion)
// M^{m}_{p} = S^n(y) w_p(y) (multipole expansion)
// then the multipole exp have NPV components and the local exp NPOT*NPV.
//
// NB1: Only the computation of forces requires that the sum over p is 
// performed at L2P step. It could be done at M2L step for the potential.
//
// NB2: An efficient application of the matrix kernel is highly kernel 
// dependent, we recommand overriding the P2M/M2L/L2P function of the kernel 
// you are using in order to have opsimal performances + set your own NRHS/NLHS.
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/// R_{,ij}
// PB: IMPORTANT! This matrix kernel does not present the symmetries 
// required by ChebSym kernel => only suited for Unif interpolation
template <class FReal>
struct FInterpMatrixKernel_R_IJ : FInterpAbstractMatrixKernel<FReal>
{
    static const KERNEL_FUNCTION_TYPE Type = NON_HOMOGENEOUS;
    static const unsigned int NK   = 3*3; //< total number of components
    static const unsigned int NCMP = 6;   //< number of independent components
    static const unsigned int NPV  = 3;   //< dim of physical values
    static const unsigned int NPOT = 3;   //< dim of potentials
    static const unsigned int NRHS = NPV; //< dim of mult exp
    static const unsigned int NLHS = NPOT*NPV; //< dim of loc exp (no csummation over j during M2L)

    // store indices (i,j) corresponding to sym idx
    static const unsigned int indexTab[/*2*NCMP=12*/];

    // store positions in sym tensor (when looping over NRHSxNLHS)
    static const unsigned int applyTab[/*NK=9*/];

    // indices to be set at construction if component-wise evaluation is performed
    const unsigned int _i,_j;

    // Material Parameters
    const FReal _CoreWidth2; // if >0 then kernel is NON homogeneous

    FInterpMatrixKernel_R_IJ(const FReal CoreWidth = 0.0, const unsigned int d = 0)
        : _i(indexTab[d]), _j(indexTab[d+NCMP]), _CoreWidth2(CoreWidth*CoreWidth)
    {}

    // copy ctor
    FInterpMatrixKernel_R_IJ(const FInterpMatrixKernel_R_IJ& other)
        : _i(other._i), _j(other._j), _CoreWidth2(other._CoreWidth2)
    {}

    static const char* getID() { return "R_IJ"; }

    static void printInfo() { 
        std::cout << "K_ij(x,y)=r_{,ij}=d^2/dx_idx_j(r), with r=|x-y|_a=sqrt(a^2 + (x_i-y_i)^2)." << std::endl; 
        std::cout << "Compute potentials p_i(x)=sum_{y,j}(K_ij(x,y)w_j(y))" << std::endl; 
        std::cout << "        forces f_{ik}(x)=-d/dx_k(sum_{y,j}(w_j(x)K_ij(x,y)w_j(y)))" << std::endl; 
        std::cout << std::endl;
    }

    // returns position in reduced storage from position in full 3x3 matrix
    unsigned  int getPosition(const unsigned int n) const
    {return applyTab[n];}

    // returns Core Width squared
    FReal getCoreWidth2() const
    {return _CoreWidth2;}

    // returns coefficient of mutual interaction
    // 1 for symmetric kernels
    // -1 for antisymmetric kernels
    // somethings else if other property of symmetry
    FReal getMutualCoefficient() const{ return FReal(1.); }

    // evaluate interaction
    template <class ValueClass>
    FReal evaluate(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                   const ValueClass& xs, const ValueClass& ys, const ValueClass& zs) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        const ValueClass r2 = diffx*diffx+diffy*diffy+diffz*diffz;
        const ValueClass one_over_r = FMath::One<ValueClass>()/FMath::Sqrt(r2 + FMath::ConvertTo<ValueClass,FReal>(_CoreWidth2));
        const ValueClass one_over_r3 = one_over_r*one_over_r*one_over_r;
        ValueClass ri,rj;

        if(_i==0) ri=diffx;
        else if(_i==1) ri=diffy;
        else if(_i==2) ri=diffz;
        else throw std::runtime_error("Update i!");

        if(_j==0) rj=diffx;
        else if(_j==1) rj=diffy;
        else if(_j==2) rj=diffz;
        else throw std::runtime_error("Update j!");

        if(_i==_j)
            return one_over_r - ri * ri * one_over_r3;
        else
            return - ri * rj * one_over_r3;

    }

    // evaluate interaction (blockwise)
    template <class ValueClass>
    void evaluateBlock(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                       const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                       ValueClass* block) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        const ValueClass r2 = diffx*diffx+diffy*diffy+diffz*diffz;
        const ValueClass one_over_r = FMath::One<ValueClass>()/FMath::Sqrt(r2 + FMath::ConvertTo<ValueClass,FReal>(_CoreWidth2));
        const ValueClass one_over_r3 = one_over_r*one_over_r*one_over_r;

        const ValueClass r[3] = {diffx,diffy,diffz};

        for(unsigned int d=0;d<NCMP;++d){
            unsigned int i = indexTab[d];
            unsigned int j = indexTab[d+NCMP];

            if(i==j)
                block[d] = one_over_r - r[i] * r[i] * one_over_r3;
            else
                block[d] = - r[i] * r[j] * one_over_r3;
        }
    }

    // evaluate interaction and derivative (blockwise)
    // [TODO] Fix! Add corewidth!
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt,
                                    const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                                    ValueClass block[NCMP], ValueClass blockDerivative[NCMP][3]) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        const ValueClass r2[3] = {diffx*diffx,diffy*diffy,diffz*diffz};
        const ValueClass one_over_r2 = FMath::One<ValueClass>() / (r2[0] + r2[1] + r2[2] + FMath::ConvertTo<ValueClass,FReal>(_CoreWidth2));
        const ValueClass one_over_r  = FMath::Sqrt(one_over_r2);
        const ValueClass one_over_r3 = one_over_r2*one_over_r;

        const ValueClass r[3] = {diffx,diffy,diffz};

        const ValueClass Three = FMath::ConvertTo<ValueClass,FReal>(3.);
        const ValueClass MinusOne = - FMath::One<ValueClass>();

        for(unsigned int d=0;d<NCMP;++d){
            unsigned int i = indexTab[d];
            unsigned int j = indexTab[d+NCMP];

            // evaluate kernel
            if(i==j)
                block[d] = one_over_r - r2[i] * one_over_r3;
            else
                block[d] = - r[i] * r[j] * one_over_r3;

            // evaluate derivative
            for(unsigned int k = 0 ; k < 3 ; ++k){
              if(i==j){
                if(j==k) //i=j=k
                  blockDerivative[d][k] = Three * ( MinusOne + r2[i] * one_over_r2 ) * r[i] * one_over_r3;
                else //i=j!=k
                  blockDerivative[d][k] = ( MinusOne + Three * r2[i] * one_over_r2 ) * r[k] * one_over_r3;
              }
              else{ //(i!=j)
                if(i==k) //i=k!=j
                  blockDerivative[d][k] = ( MinusOne + Three * r2[i] * one_over_r2 ) * r[j] * one_over_r3;
                else if(j==k) //i!=k=j
                  blockDerivative[d][k] = ( MinusOne + Three * r2[j] * one_over_r2 ) * r[i] * one_over_r3;
                else //i!=k!=j
                  blockDerivative[d][k] = Three * r[i] * r[j] * r[k] * one_over_r2 * one_over_r3;
              }
            }// k

        }// NCMP
    }

    FReal getScaleFactor(const FReal RootCellWidth, const int TreeLevel) const
    {
        const FReal CellWidth(RootCellWidth / FReal(FMath::pow(2, TreeLevel)));
        return getScaleFactor(CellWidth);
    }

    // R_{,ij} is homogeneous to [L]/[L]^{-2}=[L]^{-1}
    // => scales like ONE_OVER_R
    FReal getScaleFactor(const FReal CellWidth) const
    {
        return FReal(2.) / CellWidth;
    }



    FReal evaluate(const FPoint<FReal>& pt, const FPoint<FReal>& ps) const{
        return evaluate<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ());
    }
    void evaluateBlock(const FPoint<FReal>& pt, const FPoint<FReal>& ps, FReal* block) const{
        evaluateBlock<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block);
    }
    void evaluateBlockAndDerivative(const FPoint<FReal>& pt, const FPoint<FReal>& ps,
                                    FReal block[NCMP], FReal blockDerivative[NCMP][3]) const {
        evaluateBlockAndDerivative<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block, blockDerivative);
    }
};

/// R_IJ
template <class FReal>
const unsigned int FInterpMatrixKernel_R_IJ<FReal>::indexTab[]={0,0,0,1,1,2,
                                                                0,1,2,1,2,2};

template <class FReal>
const unsigned int FInterpMatrixKernel_R_IJ<FReal>::applyTab[]={0,1,2,
                                                                1,3,4,
                                                                2,4,5};








#endif // FINTERPMATRIXKERNEL_TENSORIAL_INTERACTIONS_HPP

// [--END--]

