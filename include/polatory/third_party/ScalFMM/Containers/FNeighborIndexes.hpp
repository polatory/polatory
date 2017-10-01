#ifndef FNEIGHBORINDEXES_HPP
#define FNEIGHBORINDEXES_HPP

#include "../Utils/FGlobal.hpp"

/**
 * @brief The FAbstractNeighborIndex class is an abstraction of
 * what should propose classes that can find the neighbor morton indexes
 * from a given index.
 * These kind of classes are useful in the P2P and the M2L
 * (in the find neighbors functions)
 */
class FAbstractNeighborIndex{
public:
    /* will return -1 or 0 if the cell is at the border */
    virtual int minX() = 0;
    /* will return -1 or 0 if the cell is at the border */
    virtual int minY() = 0;
    /* will return -1 or 0 if the cell is at the border */
    virtual int minZ() = 0;

    /* will return 1 or 0 if the cell is at the border */
    virtual int maxX() = 0;
    /* will return 1 or 0 if the cell is at the border */
    virtual int maxY() = 0;
    /* will return 1 or 0 if the cell is at the border */
    virtual int maxZ() = 0;

    /* given the neighbor relative position returns the morton index */
    virtual MortonIndex getIndex(const int inX, const int inY, const int inZ) = 0;
};




#include "FTreeCoordinate.hpp"

/**
 * This class compute the neigh position from the tree coordinates
 */
class FCoordinateNeighborIndex : public FAbstractNeighborIndex{
    int currentMinX;
    int currentMinY;
    int currentMinZ;
    int currentMaxX;
    int currentMaxY;
    int currentMaxZ;

    const MortonIndex mindex;
    const int level;

    const FTreeCoordinate coord;

public:
    FCoordinateNeighborIndex(const MortonIndex inMindex, const int inLevel)
        : mindex(inMindex), level(inLevel), coord(mindex) {

        currentMinX = (coord.getX()==0? 0 : -1);
        currentMinY = (coord.getY()==0? 0 : -1);
        currentMinZ = (coord.getZ()==0? 0 : -1);

        const int limite = FMath::pow2(level);

        currentMaxX = (coord.getX()== (limite-1)? 0 : 1);
        currentMaxY = (coord.getY()== (limite-1)? 0 : 1);
        currentMaxZ = (coord.getZ()== (limite-1)? 0 : 1);
    }

    int minX() override{
        return currentMinX;
    }
    int minY() override{
        return currentMinY;
    }
    int minZ() override{
        return currentMinZ;
    }

    int maxX() override{
        return currentMaxX;
    }
    int maxY() override{
        return currentMaxY;
    }
    int maxZ() override{
        return currentMaxZ;
    }

    MortonIndex getIndex(const int inX, const int inY, const int inZ) override{
        return FTreeCoordinate(inX+coord.getX(), inY+coord.getY(), inZ+coord.getZ()).getMortonIndex();
    }
};



/**
 * This class returns the neigh indexes from bitwise operations.
 */
class FBitsNeighborIndex : public FAbstractNeighborIndex{
    int currentMinX;
    int currentMinY;
    int currentMinZ;
    int currentMaxX;
    int currentMaxY;
    int currentMaxZ;

    const MortonIndex mindex;
    const int level;

    const MortonIndex flagZ;
    const MortonIndex flagY;
    const MortonIndex flagX;

    MortonIndex mindexes[3];


    static const int NbBitsFlag = MaxTreeHeight;

    /*constexpr*/ static MortonIndex BuildFlag(){
        MortonIndex flag = 0;
        int counter = NbBitsFlag;
        while(counter--){
            flag = (flag<<3) | 1;
        }
        return flag;
    }

public:
    FBitsNeighborIndex(const MortonIndex inMindex, const int inLevel)
        : mindex(inMindex), level(inLevel),
            flagZ(BuildFlag()), flagY(flagZ<<1), flagX(flagZ<<2){

        currentMinX = (((mindex&flagX)^flagX) == flagX ? 0 : -1);
        currentMinY = (((mindex&flagY)^flagY) == flagY ? 0 : -1);
        currentMinZ = (((mindex&flagZ)^flagZ) == flagZ ? 0 : -1);

        const int nbShiftLimit = (NbBitsFlag-level)*3;
        const MortonIndex limiteX = (flagX>>nbShiftLimit);
        const MortonIndex limiteY = (flagY>>nbShiftLimit);
        const MortonIndex limiteZ = (flagZ>>nbShiftLimit);

        currentMaxX = ((mindex&limiteX) == limiteX? 0 : 1);
        currentMaxY = ((mindex&limiteY) == limiteY? 0 : 1);
        currentMaxZ = ((mindex&limiteZ) == limiteZ? 0 : 1);

        {
            MortonIndex mindex_minus = mindex;
            MortonIndex flag_minus =   (((mindex&flagX)^flagX) == flagX ? 0 : 4)
                                     | (((mindex&flagY)^flagY) == flagY ? 0 : 2)
                                     | (((mindex&flagZ)^flagZ) == flagZ ? 0 : 1);

            while(flag_minus){
                const MortonIndex prevflag_minus = flag_minus;
                flag_minus = (flag_minus& ~mindex_minus)<<3;
                mindex_minus = (mindex_minus^prevflag_minus);
            }
            mindexes[0] = mindex_minus;
        }
        {
            MortonIndex mindex_plus = mindex;
            MortonIndex flag_plus =   ((mindex&limiteX) == limiteX? 0 : 4)
                                    | ((mindex&limiteY) == limiteY? 0 : 2)
                                    | ((mindex&limiteZ) == limiteZ? 0 : 1);

            while(flag_plus){
                const MortonIndex prevflag_plus = flag_plus;
                flag_plus = (flag_plus& mindex_plus)<<3;
                mindex_plus = (mindex_plus^prevflag_plus);
            }
            mindexes[2] = mindex_plus;
        }
        mindexes[1] = mindex;
    }

    int minX() override{
        return currentMinX;
    }
    int minY() override{
        return currentMinY;
    }
    int minZ() override{
        return currentMinZ;
    }

    int maxX() override{
        return currentMaxX;
    }
    int maxY() override{
        return currentMaxY;
    }
    int maxZ() override{
        return currentMaxZ;
    }

    MortonIndex getIndex(const int inX, const int inY, const int inZ) override{
        return (mindexes[inX+1]&flagX) | (mindexes[inY+1]&flagY) | (mindexes[inZ+1]&flagZ);
    }

    bool areNeighbors(const MortonIndex mv1, const MortonIndex mv2){
        bool cellsAreNeighbor = true;

        const MortonIndex flags[3] = { flagX, flagY, flagZ };

        for(int idx = 0; idx < 3; ++idx){
            const MortonIndex v1 = (mv1 & flags[idx]);
            const MortonIndex v2 = (mv2 & flags[idx]);

            if( (v1 == v2) || ((v1^v2) == 1) ){
                // Are neighbor
            }
            else{
                MortonIndex firstBit = 0;
                asm("bsf %1,%0" : "=r"(firstBit) : "r"(FMath::Max(v1,v2)));

                const MortonIndex highMask = ((((~MortonIndex(0))>>(firstBit+1))<<(firstBit+1)) & flags[idx]);

                if((v1&highMask) != (v2&highMask)){
                    cellsAreNeighbor = false;
                    break;
                }
                const MortonIndex lowMask = ((~((~MortonIndex(0))<<firstBit)) & flags[idx]);

                if((FMath::Min(v1,v2)&lowMask) != lowMask){
                    cellsAreNeighbor = false;
                    break;
                }
                // Are neighbors
            }
        }

        return cellsAreNeighbor;
    }
};

#endif // FNEIGHBORINDEXES_HPP
