// See LICENCE file at project root
// Keep in private GIT

#ifndef FUNIFTENSORIALKERNEL_HPP
#define FUNIFTENSORIALKERNEL_HPP

#include "../../Utils/FGlobal.hpp"

#include "../../Utils/FSmartPointer.hpp"

#include "./FAbstractUnifKernel.hpp"
#include "./FUnifM2LHandler.hpp"
#include "./FUnifTensorialM2LHandler.hpp" //PB: temporary version

class FTreeCoordinate;

/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FUnifTensorialKernel
 * @brief
 * Please read the license
 *
 * This kernels implement the Lagrange interpolation based FMM operators. It
 * implements all interfaces (P2P,P2M,M2M,M2L,L2L,L2P) which are required by
 * the FFmmAlgorithm and FFmmAlgorithmThread.
 *
 * PB: 3 IMPORTANT remarks !!!
 *
 * 1) Handling tensorial kernels (DIM,NRHS,NLHS) and having multiple rhs 
 * (NVALS) are considered 2 distinct features and are currently combined.
 *
 * 2) When it comes to applying M2L it is NOT much faster to loop over 
 * NRHSxNLHS inside applyM2L (at least for the Lagrange case).
 * 2-bis) During precomputation the tensorial matrix kernels are evaluated 
 * blockwise, but this is not always possible. 
 * In fact, in the ChebyshevSym variant the matrix kernel needs to be 
 * evaluated compo-by-compo since we currently use a scalar ACA.
 *
 * 3) We currently use multiple 1D FFT instead of multidim FFT since embedding
 * is circulant. Multidim FFT could be used if embedding were block circulant.
 * TODO investigate possibility of block circulant embedding
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 * @tparam ORDER Lagrange interpolation order
 */
template < class FReal, class CellClass, class ContainerClass,   class MatrixKernelClass, int ORDER, int NVALS = 1>
class FUnifTensorialKernel
    : public FAbstractUnifKernel<FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>
{
    enum {nRhs = MatrixKernelClass::NRHS,
          nLhs = MatrixKernelClass::NLHS,
          nPot = MatrixKernelClass::NPOT,
          nPV = MatrixKernelClass::NPV};

protected://PB: for OptiDis

    // private types
    typedef FUnifTensorialM2LHandler<FReal, ORDER,MatrixKernelClass,MatrixKernelClass::Type> M2LHandlerClass;

    // using from
    typedef FAbstractUnifKernel< FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>
    AbstractBaseClass;

    /// Needed for P2P and M2L operators
    const MatrixKernelClass *const MatrixKernel;

    /// Needed for M2L operator
    const M2LHandlerClass M2LHandler;

    /// Leaf level separation criterion
    const int LeafLevelSeparationCriterion;

public:
    /**
     * The constructor initializes all constant attributes and it reads the
     * precomputed and compressed M2L operators from a binary file (an
     * runtime_error is thrown if the required file is not valid).
     */
    FUnifTensorialKernel(const int inTreeHeight,
                         const FReal inBoxWidth,
                         const FPoint<FReal>& inBoxCenter,
                         const MatrixKernelClass *const inMatrixKernel,
                         const FReal inBoxWidthExtension,
                         const int inLeafLevelSeparationCriterion = 1)
    : FAbstractUnifKernel< FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>(inTreeHeight,inBoxWidth,inBoxCenter,inBoxWidthExtension),
      MatrixKernel(inMatrixKernel),
      M2LHandler(MatrixKernel,
                 inTreeHeight,
                 inBoxWidth,
                 inBoxWidthExtension,
                 inLeafLevelSeparationCriterion), 
      LeafLevelSeparationCriterion(inLeafLevelSeparationCriterion)
    { }


    void P2M(CellClass* const LeafCell,
             const ContainerClass* const SourceParticles)
    {
        const FPoint<FReal> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate()));
        const FReal ExtendedLeafCellWidth(AbstractBaseClass::BoxWidthLeaf 
                                          + AbstractBaseClass::BoxWidthExtension);

        for(int idxV = 0 ; idxV < NVALS ; ++idxV){

            // 1) apply Sy
            AbstractBaseClass::Interpolator->applyP2M(LeafCellCenter, ExtendedLeafCellWidth,
                                                      LeafCell->getMultipole(idxV*nRhs), SourceParticles);

            for(int idxRhs = 0 ; idxRhs < nRhs ; ++idxRhs){
                // update multipole index
                int idxMul = idxV*nRhs + idxRhs;

                // 2) apply Discrete Fourier Transform
                M2LHandler.applyZeroPaddingAndDFT(LeafCell->getMultipole(idxMul), 
                                                  LeafCell->getTransformedMultipole(idxMul));

            }
        }// NVALS
    }


    void M2M(CellClass* const FRestrict ParentCell,
             const CellClass*const FRestrict *const FRestrict ChildCells,
             const int TreeLevel)
    {
        for(int idxV = 0 ; idxV < NVALS ; ++idxV){
            for(int idxRhs = 0 ; idxRhs < nRhs ; ++idxRhs){
                // update multipole index
                int idxMul = idxV*nRhs + idxRhs;

                // 1) apply Sy
                FBlas::scal(AbstractBaseClass::nnodes, FReal(0.), ParentCell->getMultipole(idxMul));
                for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex){
                    if (ChildCells[ChildIndex]){
                        AbstractBaseClass::Interpolator->applyM2M(ChildIndex, 
                                                                  ChildCells[ChildIndex]->getMultipole(idxMul),
                                                                  ParentCell->getMultipole(idxMul), 
                                                                  TreeLevel/*Cell width extension specific*/);
                    }
                }
                // 2) Apply Discete Fourier Transform
                M2LHandler.applyZeroPaddingAndDFT(ParentCell->getMultipole(idxMul), 
                                                  ParentCell->getTransformedMultipole(idxMul));
            }
        }// NVALS
    }


    void M2L(CellClass* const FRestrict TargetCell, const CellClass* SourceCells[],
             const int neighborPositions[], const int inSize, const int TreeLevel)  override {
        const FReal CellWidth(AbstractBaseClass::BoxWidth / FReal(FMath::pow(2, TreeLevel)));
        const FReal ExtendedCellWidth(CellWidth + AbstractBaseClass::BoxWidthExtension);
        const FReal scale(MatrixKernel->getScaleFactor(ExtendedCellWidth));

        for(int idxV = 0 ; idxV < NVALS ; ++idxV){
            for (int idxLhs=0; idxLhs < nLhs; ++idxLhs){

                // update local index
                const int idxLoc = idxV*nLhs + idxLhs;

                // load transformed local expansion
                FComplex<FReal> *const TransformedLocalExpansion = TargetCell->getTransformedLocal(idxLoc);

                // update idxRhs
                const int idxRhs = idxLhs % nPV; 

                // update multipole index
                const int idxMul = idxV*nRhs + idxRhs;

                // get index in matrix kernel
                const unsigned int d = MatrixKernel->getPosition(idxLhs);

                for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
                    const int idx = neighborPositions[idxExistingNeigh];

                    M2LHandler.applyFC(idx, TreeLevel, scale, d,
                                       SourceCells[idxExistingNeigh]->getTransformedMultipole(idxMul),
                                       TransformedLocalExpansion);

                }
            }// NLHS=NPOT*NPV
        }// NVALS
    }


    void L2L(const CellClass* const FRestrict ParentCell,
             CellClass* FRestrict *const FRestrict ChildCells,
             const int TreeLevel)
    {
        for(int idxV = 0 ; idxV < NVALS ; ++idxV){
            for(int idxLhs = 0 ; idxLhs < nLhs ; ++idxLhs){
                int idxLoc = idxV*nLhs + idxLhs;
                // 1) Apply Inverse Discete Fourier Transform
                M2LHandler.unapplyZeroPaddingAndDFT(ParentCell->getTransformedLocal(idxLoc),
                                                    const_cast<CellClass*>(ParentCell)->getLocal(idxLoc));
                // 2) apply Sx
                for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex){
                    if (ChildCells[ChildIndex]){
                        AbstractBaseClass::Interpolator->applyL2L(ChildIndex, 
                                                                  ParentCell->getLocal(idxLoc), 
                                                                  ChildCells[ChildIndex]->getLocal(idxLoc),
                                                                  TreeLevel/*Cell width extension specific*/);
                    }
                }
            }
        }// NVALS
    }

    void L2P(const CellClass* const LeafCell,
             ContainerClass* const TargetParticles)
    {
        const FPoint<FReal> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate()));
        const FReal ExtendedLeafCellWidth(AbstractBaseClass::BoxWidthLeaf 
                                          + AbstractBaseClass::BoxWidthExtension);

        for(int idxV = 0 ; idxV < NVALS ; ++idxV){
            for(int idxLhs = 0 ; idxLhs < nLhs ; ++idxLhs){
                int idxLoc = idxV*nLhs + idxLhs;
                // 1)  Apply Inverse Discete Fourier Transform
                M2LHandler.unapplyZeroPaddingAndDFT(LeafCell->getTransformedLocal(idxLoc), 
                                                    const_cast<CellClass*>(LeafCell)->getLocal(idxLoc));

            }

            // 2.a) apply Sx
            AbstractBaseClass::Interpolator->applyL2P(LeafCellCenter, ExtendedLeafCellWidth,
                                                      LeafCell->getLocal(idxV*nLhs), TargetParticles);

            // 2.b) apply Px (grad Sx)
            AbstractBaseClass::Interpolator->applyL2PGradient(LeafCellCenter, ExtendedLeafCellWidth,
                                                              LeafCell->getLocal(idxV*nLhs), TargetParticles);

        }// NVALS
    }

    void P2P(const FTreeCoordinate& inPosition,
             ContainerClass* const FRestrict inTargets, const ContainerClass* const FRestrict inSources,
             ContainerClass* const inNeighbors[], const int neighborPositions[],
             const int inSize) override {
        // Standard FMM separation criterion, i.e. max 27 neighbor clusters per leaf
        if(LeafLevelSeparationCriterion==1) {
            if(inTargets == inSources){
                P2POuter(inPosition, inTargets, inNeighbors, neighborPositions, inSize);
                DirectInteractionComputer<FReal, MatrixKernelClass::NCMP, NVALS>::P2PInner(inTargets,MatrixKernel);
            }
            else{
                const ContainerClass* const srcPtr[1] = {inSources};
                DirectInteractionComputer<FReal, MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets,srcPtr,1,MatrixKernel);
                DirectInteractionComputer<FReal, MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets,inNeighbors,inSize,MatrixKernel);
            }
        }
        // Nearfield interactions are only computed within the target leaf
        else if(LeafLevelSeparationCriterion==0){
            DirectInteractionComputer<FReal,MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets,inNeighbors,inSize,MatrixKernel);
        }
        // If criterion equals -1 then no P2P need to be performed.
    }

    void P2POuter(const FTreeCoordinate& /*inLeafPosition*/,
             ContainerClass* const FRestrict inTargets,
             ContainerClass* const inNeighbors[], const int neighborPositions[],
             const int inSize) override {
        int nbNeighborsToCompute = 0;
        while(nbNeighborsToCompute < inSize
              && neighborPositions[nbNeighborsToCompute] < 14){
            nbNeighborsToCompute += 1;
        }
        DirectInteractionComputer<FReal, MatrixKernelClass::NCMP, NVALS>::P2P(inTargets,inNeighbors,nbNeighborsToCompute,MatrixKernel);
    }


    void P2PRemote(const FTreeCoordinate& /*inPosition*/,
                   ContainerClass* const FRestrict inTargets, const ContainerClass* const FRestrict /*inSources*/,
                   const ContainerClass* const inNeighbors[], const int /*neighborPositions*/[],
                   const int inSize) override {
        // Standard FMM separation criterion, i.e. max 27 neighbor clusters per leaf
        if(LeafLevelSeparationCriterion==1) 
            DirectInteractionComputer<FReal, MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets,inNeighbors,inSize,MatrixKernel);
        // Nearfield interactions are only computed within the target leaf
        if(LeafLevelSeparationCriterion==0) 
            DirectInteractionComputer<FReal, MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets,inNeighbors,0,MatrixKernel);
        // If criterion equals -1 then no P2P need to be performed.        
    }

};


#endif //FUNIFTENSORIALKERNEL_HPP

// [--END--]
