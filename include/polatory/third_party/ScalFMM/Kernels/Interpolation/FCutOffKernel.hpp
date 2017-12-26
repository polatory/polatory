// See LICENCE file at project root
// Keep in private GIT

#ifndef FCutOffKernel_HPP
#define FCutOffKernel_HPP

#include "Utils/FGlobal.hpp"
#include "Components/FAbstractKernels.hpp"

#include "Kernels/Interpolation/FInterpP2PKernels.hpp"

class FTreeCoordinate;

/**
 * @author Olivier Coulaud (Olivier.coulaud@inria.fr)
 * @class FCutOffKernel
 * @brief
 * A small kernel to perform only P2P with aMatrixKernel
 *
 * This kernels implement the Lagrange interpolation based FMM operators. It
 * implements all interfaces (P2P,P2M,M2M,M2L,L2L,L2P) which are required by
 * the FFmmAlgorithm and FFmmAlgorithmThread.
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 */
template < class FReal, class CellClass, class ContainerClass,   class MatrixKernelClass , int NVALS = 1>
class FCutOffKernel  : public FAbstractKernels< CellClass, ContainerClass>
{
  // private types
  
  /// Needed for P2P and M2L operators
  const MatrixKernelClass *const MatrixKernel;
  
  /// Leaf level separation criterion
  const int LeafLevelSeparationCriterion;
  
public:
  /**
   * The constructor initializes all constant attributes and it reads the
   * precomputed and compressed M2L operators from a binary file (an
   * runtime_error is thrown if the required file is not valid).
   */
  FCutOffKernel(const int inTreeHeight,
		const FReal inBoxWidth,
		const FPoint<FReal>& inBoxCenter,
		const MatrixKernelClass *const inMatrixKernel,
		const int inLeafLevelSeparationCriterion = 1)
    : MatrixKernel(inMatrixKernel),LeafLevelSeparationCriterion(inLeafLevelSeparationCriterion)
  { }
  
  
  void P2M(CellClass* const LeafCell,
	   const ContainerClass* const SourceParticles)
  {
    std::cout << "P2M call not needed" << std::endl;
  }


  void M2M(CellClass* const FRestrict ParentCell,
	   const CellClass*const FRestrict *const FRestrict ChildCells,
	   const int /*TreeLevel*/)
  {
    std::cout << "M2M call not needed" << std::endl;
    }


  void M2L(CellClass* const FRestrict TargetCell, const CellClass* SourceCells[],
	   const int neighborPositions[], const int inSize, const int TreeLevel)  override  {
    std::cout << "M2L call not needed" << std::endl;
    }


  void L2L(const CellClass* const FRestrict ParentCell,
	   CellClass* FRestrict *const FRestrict ChildCells,
	   const int /*TreeLevel*/)
  {
    std::cout << "L2L call not needed" << std::endl;
  }

  void L2P(const CellClass* const LeafCell,
	   ContainerClass* const TargetParticles)
  {
    std::cout << "L2P call not needed" << std::endl;

  }

  void P2P(const FTreeCoordinate& inPosition,
	   ContainerClass* const FRestrict inTargets, const ContainerClass* const FRestrict inSources,
	   ContainerClass* const inNeighbors[], const int neighborPositions[],
	   const int inSize) /*override */{
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
		const int inSize) /*override */{
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
		 const int inSize) /*override */{
    // Standard FMM separation criterion, i.e. max 27 neighbor clusters per leaf
    if(LeafLevelSeparationCriterion==1) 
      DirectInteractionComputer<FReal,MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets,inNeighbors,inSize,MatrixKernel);
    // Nearfield interactions are only computed within the target leaf
    if(LeafLevelSeparationCriterion==0) 
      DirectInteractionComputer<FReal,MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets,inNeighbors,0,MatrixKernel);
    // If criterion equals -1 then no P2P need to be performed.
  }

};


#endif //FCutOffKernel_HPP

// [--END--]
