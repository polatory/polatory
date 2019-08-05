// See LICENCE file at project root
#ifndef FABSTRACTCHEBKERNEL_HPP
#define FABSTRACTCHEBKERNEL_HPP

#include "../../Utils/FGlobal.hpp"

#include "../../Utils/FSmartPointer.hpp"

#include "../../Components/FAbstractKernels.hpp"

#include "../Interpolation/FInterpP2PKernels.hpp"
#include "./FChebInterpolator.hpp"

#include "../../Containers/FTreeCoordinate.hpp"

/**
 * @author Matthias Messner(matthias.messner@inria.fr)
 * @class FAbstractChebKernel
 * @brief
 * This kernels implement the Chebyshev interpolation based FMM operators. It
 * implements all interfaces (P2P, P2M, M2M, M2L, L2L, L2P) which are required by
 * the FFmmAlgorithm and FFmmAlgorithmThread.
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 * @tparam ORDER Chebyshev interpolation order
 */
template < class FReal, class CellClass,	class ContainerClass,	class MatrixKernelClass, int ORDER, int NVALS = 1>
class FAbstractChebKernel : public FAbstractKernels< CellClass, ContainerClass>
{
protected:
  enum {nnodes = TensorTraits<ORDER>::nnodes};
  typedef FChebInterpolator<FReal, ORDER,MatrixKernelClass,NVALS> InterpolatorClass;

  /// Needed for P2M, M2M, L2L and L2P operators
  const FSmartPointer<InterpolatorClass,FSmartPointerMemory> Interpolator;
  /// Height of the entire oct-tree
  const unsigned int TreeHeight;
  /// Corner of oct-tree box
  const FPoint<FReal> BoxCorner;
  /// Width of oct-tree box   
  const FReal BoxWidth;    
  /// Width of a leaf cell box 
  const FReal BoxWidthLeaf;
  /// Extension of the box width ( same for all level! )
  const FReal BoxWidthExtension;

  /**
   * Compute center of leaf cell from its tree coordinate.
   * @param[in] Coordinate tree coordinate
   * @return center of leaf cell
   */
  const FPoint<FReal> getLeafCellCenter(const FTreeCoordinate& Coordinate) const
  {
    return FPoint<FReal>(BoxCorner.getX() + (FReal(Coordinate.getX()) + FReal(.5)) * BoxWidthLeaf,
		  BoxCorner.getY() + (FReal(Coordinate.getY()) + FReal(.5)) * BoxWidthLeaf,
		  BoxCorner.getZ() + (FReal(Coordinate.getZ()) + FReal(.5)) * BoxWidthLeaf);
  }

  /** 
   * @brief Return the position of the center of a cell from its tree
   *  coordinate 
   * @param FTreeCoordinate
   * @param inLevel the current level of Cell
   */
  FPoint<FReal> getCellCenter(const FTreeCoordinate coordinate, int inLevel)
  {

    //Set the boxes width needed
    FReal widthAtCurrentLevel = BoxWidthLeaf*FReal(1 << (TreeHeight-(inLevel+1)));   
    FReal widthAtCurrentLevelDiv2 = widthAtCurrentLevel/FReal(2.);

    //Set the center real coordinates from box corner and widths.
    FReal X = BoxCorner.getX() + FReal(coordinate.getX())*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
    FReal Y = BoxCorner.getY() + FReal(coordinate.getY())*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
    FReal Z = BoxCorner.getZ() + FReal(coordinate.getZ())*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
    
    return FPoint<FReal>(X,Y,Z);
  }

public:
  /**
   * The constructor initializes all constant attributes and it reads the
   * precomputed and compressed M2L operators from a binary file (an
   * runtime_error is thrown if the required file is not valid).
   */
  FAbstractChebKernel(const int inTreeHeight,
                      const FReal inBoxWidth,
                      const FPoint<FReal>& inBoxCenter,
                      const FReal inBoxWidthExtension = 0.0)
    : Interpolator(new InterpolatorClass(inTreeHeight,
                                         inBoxWidth,
                                         inBoxWidthExtension)),
      TreeHeight(inTreeHeight),
      BoxCorner(inBoxCenter - inBoxWidth / FReal(2.)),
      BoxWidth(inBoxWidth),
      BoxWidthLeaf(BoxWidth / FReal(FMath::pow(2, inTreeHeight - 1))),
      BoxWidthExtension(inBoxWidthExtension)
  {
    /* empty */
  }

  virtual ~FAbstractChebKernel(){
    // should not be used
  }

  const InterpolatorClass * getPtrToInterpolator() const
  { return Interpolator.getPtr(); }


  virtual void P2M(CellClass* const LeafCell,
		   const ContainerClass* const SourceParticles) = 0;


  virtual void M2M(CellClass* const FRestrict ParentCell,
		   const CellClass*const FRestrict *const FRestrict ChildCells,
		   const int TreeLevel) = 0;


  virtual void M2L(CellClass* const FRestrict TargetCell,
           const CellClass* SourceCells[],
            const int SourcePositions[],
		   const int NumSourceCells,
		   const int TreeLevel) = 0;


  virtual void L2L(const CellClass* const FRestrict ParentCell,
		   CellClass* FRestrict *const FRestrict ChildCells,
		   const int TreeLevel) = 0;


  virtual void L2P(const CellClass* const LeafCell,
		   ContainerClass* const TargetParticles) = 0;
	
	

  virtual void P2P(const FTreeCoordinate& /* LeafCellCoordinate */, // needed for periodic boundary conditions
		   ContainerClass* const FRestrict TargetParticles,
		   const ContainerClass* const FRestrict /*SourceParticles*/,
           ContainerClass* const NeighborSourceParticles[],
            const int SourcePositions[],
		   const int /* size */) = 0;

  virtual void P2POuter(const FTreeCoordinate& inLeafPosition,
           ContainerClass* const FRestrict targets,
           ContainerClass* const directNeighborsParticles[], const int neighborPositions[],
           const int size) = 0;


  virtual void P2PRemote(const FTreeCoordinate& /*inPosition*/,
			 ContainerClass* const FRestrict inTargets, const ContainerClass* const FRestrict /*inSources*/,
             const ContainerClass* const inNeighbors[], const int SourcePositions[], const int /*inSize*/) = 0;

};





#endif //FCHEBKERNELS_HPP

// [--END--]
