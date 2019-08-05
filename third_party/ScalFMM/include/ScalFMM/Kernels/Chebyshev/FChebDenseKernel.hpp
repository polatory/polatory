// See LICENCE file at project root

#ifndef FCHEBDENSEKERNEL_HPP
#define FCHEBDENSEKERNEL_HPP

#include "../../Utils/FGlobal.hpp"

#include "../../Utils/FSmartPointer.hpp"

#include "./FAbstractChebKernel.hpp"

#include "./FChebDenseM2LHandler.hpp"

class FTreeCoordinate;

/**
 * @author Matthias Messner(matthias.messner@inria.fr)
 * @class FChebDenseKernel
 * @brief  Chebyshev interpolation based FMM operators for general non oscillatory kernels..
 *
 * This class implements the Chebyshev interpolation based FMM operators. It
 * implements all interfaces (P2P, P2M, M2M, M2L, L2L, L2P) which are required by
 * the FFmmAlgorithm, FFmmAlgorithmThread ...
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 * @tparam ORDER Chebyshev interpolation order
 *
 * The ORDER sets the accuracy of the Chebyshev FMM while the EPSILON parameter introduces extra error but optimize the M2L step.
 *  In fact, in the Chebyshev FMM we perform compression on the M2L operators using various low rank approximation techniques
 *  (see https://arxiv.org/abs/1210.7292 for further details). Therefore we use a second accuracy criterion, namely EPSILON,
 *  in order to set the accuracy of these methods. For most kernels that we tested and in particular for 1/r, setting EPSILON=10^-ORDER d
 *  oes not introduce extra error in the FMM and captures the rank efficiently. If you think that for your kernel you need a better
 *  approximation of the M2L operators, then you can try to set EPSILON to 10 ^- (ORDER+{1,2,...}).
 */
template < class FReal, class CellClass, class ContainerClass,   class MatrixKernelClass, int ORDER, int NVALS = 1>
class FChebDenseKernel
    : public FAbstractChebKernel< FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>
{
    // private types
    typedef FChebDenseM2LHandler<FReal, ORDER,MatrixKernelClass> M2LHandlerClass;

    // using from
    typedef FAbstractChebKernel<FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>
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
     *
     * @param[in] epsilon  The compression parameter for M2L operator.
     *
     * The M2L optimized Chebyshev FMM implemented in ScalFMM are kernel dependent, but keeping EPSILON=10^-ORDER is usually fine.
     *  On the other hand you can short-circuit this feature by setting EPSILON to the machine accuracy,
     *  but this will significantly slow down the computations.
     *
     */
    FChebDenseKernel(const int inTreeHeight,  const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter, const MatrixKernelClass *const inMatrixKernel,
                const FReal Epsilon)
    : FAbstractChebKernel< FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>(inTreeHeight,inBoxWidth,inBoxCenter),
      MatrixKernel(inMatrixKernel),
      M2LHandler(new M2LHandlerClass(MatrixKernel,Epsilon))
    {
        // read precomputed compressed m2l operators from binary file
        //M2LHandler->ReadFromBinaryFileAndSet();
        M2LHandler->ComputeAndCompressAndSet();
    }


    /**
     * The constructor initializes all constant attributes and it reads the
     * precomputed and compressed M2L operators from a binary file (an
     * runtime_error is thrown if the required file is not valid).
     * Same as \see above constructor  but the epsilon is automatically set to EPSILON=10^-ORDER
     */

    FChebDenseKernel(const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter, const MatrixKernelClass *const inMatrixKernel)
        :   FChebDenseKernel(inTreeHeight, inBoxWidth,inBoxCenter,inMatrixKernel,FMath::pow(10.0,static_cast<FReal>(-ORDER)))
    {

    }


    void P2M(CellClass* const LeafCell,
             const ContainerClass* const SourceParticles)
    {
        const FPoint<FReal> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate()));
        AbstractBaseClass::Interpolator->applyP2M(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                  LeafCell->getMultipole(0), SourceParticles);
    }


    void M2M(CellClass* const FRestrict ParentCell,
             const CellClass*const FRestrict *const FRestrict ChildCells,
             const int /*TreeLevel*/)
    {
        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){
            for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex){
                if (ChildCells[ChildIndex]){
                    AbstractBaseClass::Interpolator->applyM2M(ChildIndex, ChildCells[ChildIndex]->getMultipole(idxRhs),
                                                              ParentCell->getMultipole(idxRhs));
                }
            }
        }
    }

    void M2L(CellClass* const FRestrict TargetCell, const CellClass* SourceCells[],
             const int neighborPositions[], const int inSize, const int TreeLevel)  override {
        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){
            FReal *const localExpansion = TargetCell->getLocal(idxRhs);
            const FReal CellWidth(AbstractBaseClass::BoxWidth / FReal(FMath::pow(2, TreeLevel)));
            for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
                const int idx = neighborPositions[idxExistingNeigh];
                M2LHandler->applyC(idx, CellWidth, SourceCells[idxExistingNeigh]->getMultipole(idxRhs),
                                   localExpansion);
            }
        }
    }


    void L2L(const CellClass* const FRestrict ParentCell,
             CellClass* FRestrict *const FRestrict ChildCells,
             const int /*TreeLevel*/)
    {
        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){
            for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex){
                if (ChildCells[ChildIndex]){
                    AbstractBaseClass::Interpolator->applyL2L(ChildIndex, ParentCell->getLocal(idxRhs), ChildCells[ChildIndex]->getLocal(idxRhs));
                }
            }
        }
    }

    void L2P(const CellClass* const LeafCell,
             ContainerClass* const TargetParticles)
    {
        const FPoint<FReal> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate()));

        //// 2.a) apply Sx
        //AbstractBaseClass::Interpolator->applyL2P(LeafCellCenter,
        //                                          AbstractBaseClass::BoxWidthLeaf,
        //                                          LeafCell->getLocal(0),
        //                                          TargetParticles);
        //// 2.b) apply Px (grad Sx)
        //AbstractBaseClass::Interpolator->applyL2PGradient(LeafCellCenter,
        //                                                  AbstractBaseClass::BoxWidthLeaf,
        //                                                  LeafCell->getLocal(0),
        //                                                  TargetParticles);

        // 2.c) apply Sx and Px (grad Sx)
        AbstractBaseClass::Interpolator->applyL2PTotal(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                        LeafCell->getLocal(0), TargetParticles);

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


#endif //FCHEBKERNELS_HPP

// [--END--]
