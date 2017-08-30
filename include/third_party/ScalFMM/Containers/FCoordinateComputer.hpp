#ifndef FCOORDINATECOMPUTER_HPP
#define FCOORDINATECOMPUTER_HPP

#include "../Utils/FGlobal.hpp"
#include "FTreeCoordinate.hpp"
#include "../Utils/FPoint.hpp"
#include "../Utils/FMath.hpp"
#include "../Utils/FAssert.hpp"

/**
 * @brief The FCoordinateComputer struct is the main class to get the tree coordinate
 * from the simulation box properties.
 */
struct FCoordinateComputer {
    template <class FReal>
    static inline int GetTreeCoordinate(const FReal inRelativePosition, const FReal boxWidth,
                          const FReal boxWidthAtLeafLevel, const int treeHeight) {
        FAssertLF( (inRelativePosition >= 0 && inRelativePosition <= boxWidth), "inRelativePosition : ",inRelativePosition, " boxWidth ", boxWidth );
        if(inRelativePosition == boxWidth){
            return FMath::pow2(treeHeight-1)-1;
        }
        const FReal indexFReal = inRelativePosition / boxWidthAtLeafLevel;
        return static_cast<int>(indexFReal);
    }


    template <class FReal>
    static inline FTreeCoordinate GetCoordinateFromPosition(const FPoint<FReal>& centerOfBox, const FReal boxWidth, const int treeHeight,
                                              const FPoint<FReal>& pos) {
        const FPoint<FReal> boxCorner(centerOfBox,-(boxWidth/2));
        const FReal boxWidthAtLeafLevel(boxWidth/FReal(1<<(treeHeight-1)));

        // box coordinate to host the particle
        FTreeCoordinate host;
        // position has to be relative to corner not center
        host.setX( GetTreeCoordinate<FReal>( pos.getX() - boxCorner.getX(), boxWidth, boxWidthAtLeafLevel, treeHeight));
        host.setY( GetTreeCoordinate<FReal>( pos.getY() - boxCorner.getY(), boxWidth, boxWidthAtLeafLevel, treeHeight));
        host.setZ( GetTreeCoordinate<FReal>( pos.getZ() - boxCorner.getZ(), boxWidth, boxWidthAtLeafLevel, treeHeight));
        return host;
    }

    template <class FReal>
    static inline FTreeCoordinate GetCoordinateFromPositionAndCorner(const FPoint<FReal>& cornerOfBox, const FReal boxWidth, const int treeHeight,
                                              const FPoint<FReal>& pos) {
        const FReal boxWidthAtLeafLevel(boxWidth/FReal(1<<(treeHeight-1)));

        // box coordinate to host the particle
        FTreeCoordinate host;
        // position has to be relative to corner not center
        host.setX( GetTreeCoordinate<FReal>( pos.getX() - cornerOfBox.getX(), boxWidth, boxWidthAtLeafLevel, treeHeight));
        host.setY( GetTreeCoordinate<FReal>( pos.getY() - cornerOfBox.getY(), boxWidth, boxWidthAtLeafLevel, treeHeight));
        host.setZ( GetTreeCoordinate<FReal>( pos.getZ() - cornerOfBox.getZ(), boxWidth, boxWidthAtLeafLevel, treeHeight));
        return host;
    }


    template <class FReal>
    static inline FPoint<FReal> GetPositionFromCoordinate(const FPoint<FReal>& centerOfBox, const FReal boxWidth, const int treeHeight,
                                              const FTreeCoordinate& pos) {
        const FPoint<FReal> boxCorner(centerOfBox,-(boxWidth/2));
        const FReal boxWidthAtLeafLevel(boxWidth/FReal(1<<(treeHeight-1)));

        // box coordinate to host the particle
        FPoint<FReal> host;
        // position has to be relative to corner not center
        host.setX( pos.getX()*boxWidthAtLeafLevel + boxCorner.getX() );
        host.setY( pos.getY()*boxWidthAtLeafLevel + boxCorner.getY() );
        host.setZ( pos.getZ()*boxWidthAtLeafLevel + boxCorner.getZ() );
        return host;
    }

};


#endif // FCOORDINATECOMPUTER_HPP

