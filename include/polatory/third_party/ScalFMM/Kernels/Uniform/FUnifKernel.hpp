// See LICENCE file at project root
// Keep in private GIT

#ifndef FUNIFKERNEL_HPP
#define FUNIFKERNEL_HPP

#include "Utils/FGlobal.hpp"

#include "Utils/FSmartPointer.hpp"

#include "FAbstractUnifKernel.hpp"
#include "FUnifM2LHandler.hpp"

class FTreeCoordinate;

/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FUnifKernel
 * @brief
 * Please read the license
 *
 * This kernels implement the Lagrange interpolation based FMM operators. It
 * implements all interfaces (P2P,P2M,M2M,M2L,L2L,L2P) which are required by
 * the FFmmAlgorithm and FFmmAlgorithmThread.
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 * @tparam ORDER Lagrange interpolation order
 */
template < class FReal, class CellClass, class ContainerClass,   class MatrixKernelClass, int ORDER, int NVALS = 1>
class FUnifKernel
  : public FAbstractUnifKernel<FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>
{
    // private types
    typedef FUnifM2LHandler<FReal, ORDER,MatrixKernelClass::Type> M2LHandlerClass;

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
    FUnifKernel(const int inTreeHeight,
                const FReal inBoxWidth,
                const FPoint<FReal>& inBoxCenter,
                const MatrixKernelClass *const inMatrixKernel,
                const int inLeafLevelSeparationCriterion = 1)
    : FAbstractUnifKernel< FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER, NVALS>(inTreeHeight,inBoxWidth,inBoxCenter),
      MatrixKernel(inMatrixKernel),
      M2LHandler(MatrixKernel,
                 inTreeHeight,
                 inBoxWidth,
                 inLeafLevelSeparationCriterion),
      LeafLevelSeparationCriterion(inLeafLevelSeparationCriterion)
    { }


    void P2M(CellClass* const LeafCell,
             const ContainerClass* const SourceParticles)
    {
        const FPoint<FReal> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate()));
        // 1) apply Sy
        AbstractBaseClass::Interpolator->applyP2M(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                  LeafCell->getMultipole(0), SourceParticles);

        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){

            // 2) apply Discrete Fourier Transform
            M2LHandler.applyZeroPaddingAndDFT(LeafCell->getMultipole(idxRhs), 
                                              LeafCell->getTransformedMultipole(idxRhs));

        }
    }


    void M2M(CellClass* const FRestrict ParentCell,
             const CellClass*const FRestrict *const FRestrict ChildCells,
             const int /*TreeLevel*/)
    {
        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){
            // 1) apply Sy
            //FBlas::scal(AbstractBaseClass::nnodes, FReal(0.), ParentCell->getMultipole(idxRhs));
            for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex){
                if (ChildCells[ChildIndex]){
                    AbstractBaseClass::Interpolator->applyM2M(ChildIndex, ChildCells[ChildIndex]->getMultipole(idxRhs),
                                                              ParentCell->getMultipole(idxRhs));
                }
            }
            // 2) Apply Discete Fourier Transform
            M2LHandler.applyZeroPaddingAndDFT(ParentCell->getMultipole(idxRhs), 
                                              ParentCell->getTransformedMultipole(idxRhs));
        }
    }


    void M2L(CellClass* const FRestrict TargetCell, const CellClass* SourceCells[],
             const int neighborPositions[], const int inSize, const int TreeLevel)  override {
        const FReal CellWidth(AbstractBaseClass::BoxWidth / FReal(FMath::pow(2, TreeLevel)));
        const FReal scale(MatrixKernel->getScaleFactor(CellWidth));

        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){
            FComplex<FReal> *const TransformedLocalExpansion = TargetCell->getTransformedLocal(idxRhs);

            for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
                const int idxNeigh = neighborPositions[idxExistingNeigh];
                M2LHandler.applyFC(idxNeigh, TreeLevel, scale,
                                   SourceCells[idxExistingNeigh]->getTransformedMultipole(idxRhs),
                                   TransformedLocalExpansion);
            }
        }
    }


    void L2L(const CellClass* const FRestrict ParentCell,
             CellClass* FRestrict *const FRestrict ChildCells,
             const int /*TreeLevel*/)
    {
        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){

            // 1) Apply Inverse Discete Fourier Transform
            FReal localExp[AbstractBaseClass::nnodes];
            M2LHandler.unapplyZeroPaddingAndDFT(ParentCell->getTransformedLocal(idxRhs),
                                                localExp);
            FBlas::add(AbstractBaseClass::nnodes,const_cast<FReal*>(ParentCell->getLocal(idxRhs)),localExp);

            // 2) apply Sx
            for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex){
                if (ChildCells[ChildIndex]){
                    AbstractBaseClass::Interpolator->applyL2L(ChildIndex, localExp, ChildCells[ChildIndex]->getLocal(idxRhs));
                }
            }
        }
    }

    void L2P(const CellClass* const LeafCell,
             ContainerClass* const TargetParticles)
    {
        const FPoint<FReal> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate()));

        FReal localExp[NVALS*AbstractBaseClass::nnodes];

        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){

            // 1)  Apply Inverse Discete Fourier Transform
            M2LHandler.unapplyZeroPaddingAndDFT(LeafCell->getTransformedLocal(idxRhs), 
                                                localExp + idxRhs*AbstractBaseClass::nnodes);
            FBlas::add(AbstractBaseClass::nnodes,const_cast<FReal*>(LeafCell->getLocal(idxRhs)),localExp + idxRhs*AbstractBaseClass::nnodes);

        }

        // 2.a) apply Sx
        AbstractBaseClass::Interpolator->applyL2P(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                  localExp, TargetParticles);

        // 2.b) apply Px (grad Sx)
        AbstractBaseClass::Interpolator->applyL2PGradient(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                          localExp, TargetParticles);


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
            DirectInteractionComputer<FReal,MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets,inNeighbors,inSize,MatrixKernel);
        // Nearfield interactions are only computed within the target leaf
        if(LeafLevelSeparationCriterion==0) 
            DirectInteractionComputer<FReal,MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets,inNeighbors,0,MatrixKernel);
        // If criterion equals -1 then no P2P need to be performed.
    }

};


#endif //FUNIFKERNEL_HPP

// [--END--]
