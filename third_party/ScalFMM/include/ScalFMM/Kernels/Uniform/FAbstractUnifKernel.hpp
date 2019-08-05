// See LICENCE file at project root
// Keep in private GIT

#ifndef FABSTRACTUNIFKERNEL_HPP
#define FABSTRACTUNIFKERNEL_HPP

#include "Utils/FGlobal.hpp"

#include "Utils/FSmartPointer.hpp"

#include "Components/FAbstractKernels.hpp"

#include "../Interpolation/FInterpP2PKernels.hpp"
#include "./FUnifInterpolator.hpp"

#include "Containers/FTreeCoordinate.hpp"

/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FAbstractUnifKernel
 * @brief
 * This kernels implement the Lagrange interpolation based FMM operators. It
 * implements all interfaces (P2P, P2M, M2M, M2L, L2L, L2P) which are required by
 * the FFmmAlgorithm and FFmmAlgorithmThread.
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 * @tparam ORDER Lagrange interpolation order
 *
 * Related publications:
 * Fast hierarchical algorithms for generating Gaussian random fields
 * (https://hal.inria.fr/hal-01228519)
 */
template < class FReal, class CellClass,	class ContainerClass,	class MatrixKernelClass, int ORDER, int NVALS = 1>
class FAbstractUnifKernel : public FAbstractKernels< CellClass, ContainerClass>
{
protected:
  enum {nnodes = TensorTraits<ORDER>::nnodes};
  typedef FUnifInterpolator<FReal, ORDER,MatrixKernelClass,NVALS> InterpolatorClass;

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
  FAbstractUnifKernel(const int inTreeHeight,
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

  virtual ~FAbstractUnifKernel(){
    // should not be used
  }

  const InterpolatorClass * getPtrToInterpolator() const
  { return Interpolator.getPtr(); }


  virtual void P2M(CellClass* const LeafCell,
                   const ContainerClass* const SourceParticles) = 0;


  virtual void M2M(CellClass* const FRestrict ParentCell,
                   const CellClass*const FRestrict *const FRestrict ChildCells,
                   const int TreeLevel) = 0;


  virtual void M2L(CellClass* const FRestrict TargetCell, const CellClass* SourceCells[],
                   const int neighborPositions[], const int inSize, const int TreeLevel) = 0;


  virtual void L2L(const CellClass* const FRestrict ParentCell,
                   CellClass* FRestrict *const FRestrict ChildCells,
                   const int TreeLevel) = 0;


  virtual void L2P(const CellClass* const LeafCell,
                   ContainerClass* const TargetParticles) = 0;



  virtual void P2P(const FTreeCoordinate& /* LeafCellCoordinate */, // needed for periodic boundary conditions
                   ContainerClass* const FRestrict TargetParticles,
                   const ContainerClass* const FRestrict /*SourceParticles*/,
                   ContainerClass* const NeighborSourceParticles[],
                   const int neighborPositions[],
                   const int /* size */) = 0;


  virtual void P2PRemote(const FTreeCoordinate& /*inPosition*/,
                         ContainerClass* const FRestrict inTargets,
                         const ContainerClass* const FRestrict /*inSources*/,
                         const ContainerClass* const inNeighbors[],
                         const int neighborPositions[],
                         const int /*inSize*/) = 0;

};





#endif //FABSTRACTUNIFKERNEL_HPP

// [--END--]
