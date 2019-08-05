#ifndef FP2PEXCLUSION_HPP
#define FP2PEXCLUSION_HPP

#include "../Containers/FTreeCoordinate.hpp"

/**
 * This class gives is responsible of the separation of the leaves
 * using the coloring algorithm.
 * In case of classic P2P and mutual interaction the BoxSeparations = 2 should be used.
 * For our current mutual P2P is is a little more complicated because we need
 * 2 boxes of separation but only in some directions.
 */
template <int BoxSeparations = 2>
class FP2PExclusion{
public:
    static const int BoxesPerDim = (BoxSeparations+1);
    static const int SizeShape = BoxesPerDim*BoxesPerDim*BoxesPerDim;

    static int GetShapeIdx(const int inX, const int inY, const int inZ){
        return (inX%BoxesPerDim)*(BoxesPerDim*BoxesPerDim) + (inY%BoxesPerDim)*BoxesPerDim + (inZ%BoxesPerDim);
    }

    static int GetShapeIdx(const FTreeCoordinate& coord){
        return GetShapeIdx(coord.getX(), coord.getY(), coord.getZ());
    }
};

/**
 * Here the formula is related to the octree construction of neighbors list:
 * const int index = (((idxX + 1) * 3) + (idxY +1)) * 3 + idxZ + 1;
 * If go from 0 to 27,
 * if we loop from 0 to 14, then we need "x" in [0;2[
 * "y" "z" in [0;3[
 */
class FP2PMiddleExclusion{
public:
    static const int SizeShape = 3*3*2;

    static int GetShapeIdx(const int inX, const int inY, const int inZ){
        return (inX%2)*9 + (inY%3)*3 + (inZ%3);
    }

    static int GetShapeIdx(const FTreeCoordinate& coord){
        return GetShapeIdx(coord.getX(), coord.getY(), coord.getZ());
    }
};


#endif // FP2PEXCLUSION_HPP

