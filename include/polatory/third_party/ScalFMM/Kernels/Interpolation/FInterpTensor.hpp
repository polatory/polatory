// See LICENCE file at project root
#ifndef FINTERPTENSOR_HPP
#define FINTERPTENSOR_HPP

#include "../../Utils/FMath.hpp"

#include "./FInterpMapping.hpp"


/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * Please read the license
 */

/**
 * @class TensorTraits
 *
 * The class @p TensorTraits gives the number of interpolation nodes per
 * cluster in 3D, depending on the interpolation order.
 *
 * @tparam ORDER interpolation order
 */
template <int ORDER> struct TensorTraits
{
	enum {nnodes = ORDER*ORDER*ORDER};
};


/**
 * @class FInterpTensor
 *
 * The class FInterpTensor provides function considering the tensor product
 * interpolation.
 *
 * @tparam ORDER interpolation order \f$\ell\f$
 * @tparam RootsClass class containing the roots choosen for the interpolation 
 * (e.g. FChebRoots, FUnifRoots...)

 */
template <class FReal, int ORDER, typename RootsClass>
class FInterpTensor : FNoCopyable
{
  enum {nnodes = TensorTraits<ORDER>::nnodes};
  typedef RootsClass BasisType;

public:

  /**
   * Sets the ids of the coordinates of all \f$\ell^3\f$ interpolation
   * nodes
   *
   * @param[out] NodeIds ids of coordinates of interpolation nodes
   */
  static
  void setNodeIds(unsigned int NodeIds[nnodes][3])
  {
    for (unsigned int n=0; n<nnodes; ++n) {
      NodeIds[n][0] =  n         % ORDER;
      NodeIds[n][1] = (n/ ORDER) % ORDER;
      NodeIds[n][2] =  n/(ORDER  * ORDER);
    }
  }


  /**
   * Sets the interpolation points in the cluster with @p center and @p width
   *
   * PB: tensorial version
   *
   * @param[in] center of cluster
   * @param[in] width of cluster
   * @param[out] rootPositions coordinates of interpolation points
   */
  static
  void setRoots(const FPoint<FReal>& center, const FReal width, FPoint<FReal> rootPositions[nnodes])
  {
    unsigned int node_ids[nnodes][3];
    setNodeIds(node_ids);
    const map_loc_glob<FReal> map(center, width);
    FPoint<FReal> localPosition;
    for (unsigned int n=0; n<nnodes; ++n) {
      localPosition.setX(FReal(BasisType::roots[node_ids[n][0]]));
      localPosition.setY(FReal(BasisType::roots[node_ids[n][1]]));
      localPosition.setZ(FReal(BasisType::roots[node_ids[n][2]]));
      map(localPosition, rootPositions[n]);
    }
  }

  /**
   * Sets the equispaced roots in the cluster with @p center and @p width
   *
   * @param[in] center of cluster
   * @param[in] width of cluster
   * @param[out] roots coordinates of equispaced roots
   */
  static
  void setPolynomialsRoots(const FPoint<FReal>& center, const FReal width, FReal roots[3][ORDER])
  {
    const map_loc_glob<FReal> map(center, width);
    FPoint<FReal> lPos, gPos;
    for (unsigned int n=0; n<ORDER; ++n) {
      lPos.setX(FReal(BasisType::roots[n]));
      lPos.setY(FReal(BasisType::roots[n]));
      lPos.setZ(FReal(BasisType::roots[n]));
      map(lPos, gPos);
      roots[0][n] = gPos.getX();
      roots[1][n] = gPos.getY();
      roots[2][n] = gPos.getZ();
    }
  }

  /**
   * Set the relative child (width = 1) center according to the Morton index.
   *
   * @param[in] ChildIndex index of child according to Morton index
   * @param[out] center
   * @param[in] ExtendedCellRatio ratio between extended child and parent widths
   */
  static
  void setRelativeChildCenter(const unsigned int ChildIndex,
                              FPoint<FReal>& ChildCenter,
                              const FReal ExtendedCellRatio=FReal(.5))
  {
    const int RelativeChildPositions[][3] = { {-1, -1, -1},
                                              {-1, -1,  1},
                                              {-1,  1, -1},
                                              {-1,  1,  1},
                                              { 1, -1, -1},
                                              { 1, -1,  1},
                                              { 1,  1, -1},
                                              { 1,  1,  1} };
    // Translate center if cell widths are extended
    const FReal frac = (FReal(1.) - ExtendedCellRatio); 

    ChildCenter.setX(FReal(RelativeChildPositions[ChildIndex][0]) * frac);
    ChildCenter.setY(FReal(RelativeChildPositions[ChildIndex][1]) * frac);
    ChildCenter.setZ(FReal(RelativeChildPositions[ChildIndex][2]) * frac);
  }
};





#endif /*FINTERPTENSOR_HPP*/
