// See LICENCE file at project root
#ifndef FCHEBTENSORIALKERNEL_HPP
#define FCHEBTENSORIALKERNEL_HPP

#include "../../Utils/FGlobal.hpp"

#include "../../Utils/FSmartPointer.hpp"

#include "./FAbstractChebKernel.hpp"
#include "./FChebTensorialM2LHandler.hpp" //PB: temporary version

class FTreeCoordinate;

/**
 * @author Matthias Messner(matthias.messner@inria.fr)
 * @class FChebTensorialKernel
 * @brief
 * Please read the license
 *
 * This kernels implement the Chebyshev interpolation based FMM operators. It
 * implements all interfaces (P2P, P2M, M2M, M2L, L2L, L2P) which are required by
 * the FFmmAlgorithm and FFmmAlgorithmThread.
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 * @tparam ORDER Chebyshev interpolation order
 */
template < class FReal, class CellClass, class ContainerClass,   class MatrixKernelClass, int ORDER, int NVALS = 1>
class FChebTensorialKernel
    : public FAbstractChebKernel< FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>
{
    enum {nRhs = MatrixKernelClass::NRHS,
          nLhs = MatrixKernelClass::NLHS,
          nPot = MatrixKernelClass::NPOT,
          nPv = MatrixKernelClass::NPV};

protected://PB: for OptiDis

    // private types
    typedef FChebTensorialM2LHandler<FReal, ORDER,MatrixKernelClass,MatrixKernelClass::Type> M2LHandlerClass;

    // using from 
    typedef FAbstractChebKernel< FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>
    AbstractBaseClass;

    /// Needed for P2P and M2L operators
    const MatrixKernelClass *const MatrixKernel;

    /// Needed for M2L operator
    FSmartPointer<  M2LHandlerClass,FSmartPointerMemory> M2LHandler;

public:
    /**
     * The constructor initializes all constant attributes and it reads the
     * precomputed and compressed M2L operators from a binary file (an
     * runtime_error is thrown if the required file is not valid).
     */
    FChebTensorialKernel(const int inTreeHeight,
                         const FReal inBoxWidth,
                         const FPoint<FReal>& inBoxCenter,
                         const MatrixKernelClass *const inMatrixKernel,
                         const FReal inBoxWidthExtension)
    : FAbstractChebKernel< FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>(inTreeHeight,inBoxWidth,inBoxCenter,inBoxWidthExtension),
      MatrixKernel(inMatrixKernel),
      M2LHandler(new M2LHandlerClass(MatrixKernel,
                                     inTreeHeight,
                                     inBoxWidth,
                                     inBoxWidthExtension))
    { }


    void P2M(CellClass* const LeafCell,
                     const ContainerClass* const SourceParticles)
    {
        const FPoint<FReal> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate()));
        const FReal ExtendedLeafCellWidth(AbstractBaseClass::BoxWidthLeaf 
                                          + AbstractBaseClass::BoxWidthExtension);

        for(int idxV = 0 ; idxV < NVALS ; ++idxV){
            // apply Sy
            AbstractBaseClass::Interpolator->applyP2M(LeafCellCenter, ExtendedLeafCellWidth,
                                                      LeafCell->getMultipole(idxV*nRhs), SourceParticles);
        }
    }


    void M2M(CellClass* const FRestrict ParentCell,
             const CellClass*const FRestrict *const FRestrict ChildCells,
             const int TreeLevel)
    {
        for(int idxV = 0 ; idxV < NVALS ; ++idxV){
            for(int idxRhs = 0 ; idxRhs < nRhs ; ++idxRhs){
                // update multipole index
                int idxMul = idxV*nRhs + idxRhs;
                FBlas::scal(AbstractBaseClass::nnodes*2, FReal(0.), ParentCell->getMultipole(idxMul));
                for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex){
                    if (ChildCells[ChildIndex]){
                        AbstractBaseClass::Interpolator->applyM2M(ChildIndex, 
                                                                  ChildCells[ChildIndex]->getMultipole(idxMul),
                                                                  ParentCell->getMultipole(idxMul), 
                                                                  TreeLevel/*Cell width extension specific*/);
                    }
                }

            }
        }
    }

    void M2L(CellClass* const FRestrict TargetCell, const CellClass* SourceCells[],
             const int neighborPositions[], const int inSize, const int TreeLevel)  override {
        // scale factor for homogeneous case
        const FReal CellWidth(AbstractBaseClass::BoxWidth / FReal(FMath::pow(2, TreeLevel)));
        const FReal ExtendedCellWidth(CellWidth + AbstractBaseClass::BoxWidthExtension);
        const FReal scale(MatrixKernel->getScaleFactor(ExtendedCellWidth));

        for(int idxV = 0 ; idxV < NVALS ; ++idxV){
            for (int idxLhs=0; idxLhs < nLhs; ++idxLhs){
                // update local index
                const int idxLoc = idxV*nLhs + idxLhs;

                FReal *const localExpansion = TargetCell->getLocal(idxLoc);

                // update idxRhs
                const int idxRhs = idxLhs % nPv; 
                // update multipole index
                const int idxMul = idxV*nRhs + idxRhs;

                // get index in matrix kernel
                const unsigned int d = MatrixKernel->getPosition(idxLhs);

                for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
                    const int idx = neighborPositions[idxExistingNeigh];
                    M2LHandler->applyC(idx, TreeLevel, scale, d,
                                       SourceCells[idxExistingNeigh]->getMultipole(idxMul),
                                       localExpansion);
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
                for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex){
                    if (ChildCells[ChildIndex]){
                      AbstractBaseClass::Interpolator->applyL2L(ChildIndex, 
                                                                ParentCell->getLocal(idxLoc), 
                                                                ChildCells[ChildIndex]->getLocal(idxLoc), 
                                                                TreeLevel/*Cell width extension specific*/);
                    }
                }
            }//NLHS
        }// NVALS
    }


    void L2P(const CellClass* const LeafCell,
                     ContainerClass* const TargetParticles)
    {
        const FPoint<FReal> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate()));
        const FReal ExtendedLeafCellWidth(AbstractBaseClass::BoxWidthLeaf 
                                          + AbstractBaseClass::BoxWidthExtension);

        for(int idxV = 0 ; idxV < NVALS ; ++idxV){
            AbstractBaseClass::Interpolator->applyL2PTotal(LeafCellCenter, ExtendedLeafCellWidth,
                                                           LeafCell->getLocal(idxV*nLhs), TargetParticles);
        }
    }

    void P2P(const FTreeCoordinate& inPosition,
             ContainerClass* const FRestrict inTargets, const ContainerClass* const FRestrict inSources,
             ContainerClass* const inNeighbors[], const int neighborPositions[],
             const int inSize) override {
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
        DirectInteractionComputer<FReal, MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets,inNeighbors,inSize,MatrixKernel);
    }

};


#endif //FCHEBTENSORIALKERNELS_HPP

// [--END--]
