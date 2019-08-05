// See LICENCE file at project root
// Keep in private GIT

#ifndef FUNIFDENSEKERNEL_HPP
#define FUNIFDENSEKERNEL_HPP

#include "../../Utils/FGlobal.hpp"

#include "../../Utils/FSmartPointer.hpp"

#include "./FAbstractUnifKernel.hpp"
#include "./FUnifM2LHandler.hpp"

class FTreeCoordinate;

/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FUnifDenseKernel
 * @brief
 * Please read the license
 *
 * This kernels implement the Lagrange interpolation based FMM operators. It
 * implements all interfaces (P2P,P2M,M2M,M2L,L2L,L2P) which are required by
 * the FFmmAlgorithm and FFmmAlgorithmThread. 
 * This is the dense version of the kernel. The transfer are done in real space
 * and not in Fourier space. 
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 * @tparam ORDER Lagrange interpolation order
 */
template < class FReal, class CellClass,	class ContainerClass,	class MatrixKernelClass, int ORDER, int NVALS = 1>
class FUnifDenseKernel
  : public FAbstractUnifKernel<FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>
{
    // private types
    typedef FUnifM2LHandler<FReal, ORDER,MatrixKernelClass::Type> M2LHandlerClass;

    // using from
    typedef FAbstractUnifKernel<FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>
    AbstractBaseClass;

    /// Needed for P2P and M2L operators
    const MatrixKernelClass *const MatrixKernel;

    /// Needed for M2L operator
    const M2LHandlerClass M2LHandler;


public:
    /**
    * The constructor initializes all constant attributes and it reads the
    * precomputed and compressed M2L operators from a binary file (an
    * runtime_error is thrown if the required file is not valid).
    */
    FUnifDenseKernel(const int inTreeHeight,
                     const FReal inBoxWidth,
                     const FPoint<FReal>& inBoxCenter,
                     const MatrixKernelClass *const inMatrixKernel)
    : FAbstractUnifKernel< FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>(inTreeHeight,inBoxWidth,inBoxCenter),
      MatrixKernel(inMatrixKernel),
      M2LHandler(MatrixKernel,
                 inTreeHeight,
                 inBoxWidth) 
    { }


    void P2M(CellClass* const LeafCell,
             const ContainerClass* const SourceParticles)
    {
        const FPoint<FReal> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate()));
        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){
            // 1) apply Sy
            AbstractBaseClass::Interpolator->applyP2M(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                      LeafCell->getMultipole(idxRhs), SourceParticles);
        }
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
             const int /*neighborPositions*/[], const int inSize, const int TreeLevel)  override {
        const FReal CellWidth(AbstractBaseClass::BoxWidth / FReal(FMath::pow(2, TreeLevel)));

        // interpolation points of source (Y) and target (X) cell
        FPoint<FReal> X[AbstractBaseClass::nnodes], Y[AbstractBaseClass::nnodes];
        FUnifTensor<FReal,ORDER>::setRoots(AbstractBaseClass::getCellCenter(TargetCell->getCoordinate(),TreeLevel), CellWidth, X);

        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){

            for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
                FUnifTensor<FReal,ORDER>::setRoots(AbstractBaseClass::getCellCenter(SourceCells[idxExistingNeigh]->getCoordinate(),TreeLevel), CellWidth, Y);

                for (unsigned int m=0; m<AbstractBaseClass::nnodes; ++m)
                    for (unsigned int n=0; n<AbstractBaseClass::nnodes; ++n){
                        TargetCell->getLocal(idxRhs)[m]+=MatrixKernel->evaluate(X[m], Y[n]) * SourceCells[idxExistingNeigh]->getMultipole(idxRhs)[n];
                    }

            }
        }
    }


    void L2L(const CellClass* const FRestrict ParentCell,
             CellClass* FRestrict *const FRestrict ChildCells,
             const int /*TreeLevel*/)
    {
        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){
            // 2) apply Sx
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

        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){

            // 2.a) apply Sx
            AbstractBaseClass::Interpolator->applyL2P(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                      LeafCell->getLocal(idxRhs), TargetParticles);

            // 2.b) apply Px (grad Sx)
            AbstractBaseClass::Interpolator->applyL2PGradient(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                              LeafCell->getLocal(idxRhs), TargetParticles);

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


#endif //FUNIFDENSEKERNEL_HPP

// [--END--]
