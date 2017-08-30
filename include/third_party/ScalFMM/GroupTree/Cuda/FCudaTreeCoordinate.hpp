#ifndef FCUDATREECOORDINATE_HPP
#define FCUDATREECOORDINATE_HPP

#include "FCudaGlobal.hpp"


class FCudaTreeCoordinate {
public:
    __device__ static int3 ConvertCoordinate(const int coordinate[3]) {
        int3 coord;
        coord.x = coordinate[0];
        coord.y = coordinate[1];
        coord.z = coordinate[2];
        return coord;
    }

    __device__ static int3 GetPositionFromMorton(MortonIndex inIndex, const int inLevel){
        MortonIndex mask = 0x1LL;

        int3 coord;
        coord.x = 0;
        coord.y = 0;
        coord.z = 0;

        for(int indexLevel = 0; indexLevel < inLevel ; ++indexLevel){
            coord.z |= int(inIndex & mask);
            inIndex >>= 1;
            coord.y |= int(inIndex & mask);
            inIndex >>= 1;
            coord.x |= int(inIndex & mask);

            mask <<= 1;
        }

        return coord;
    }

    /**
    * To get the morton index of the current position
    * @complexity inLevel
    * @param inLevel the level of the component
    * @return morton index
    */
    __device__ static MortonIndex GetMortonIndex(const int3 coord, const int inLevel) {
        MortonIndex index = 0x0LL;
        MortonIndex mask = 0x1LL;
        // the ordre is xyz.xyz...
        MortonIndex mx = coord.x << 2;
        MortonIndex my = coord.y << 1;
        MortonIndex mz = coord.z;

        for(int indexLevel = 0; indexLevel < inLevel ; ++indexLevel){
            index |= (mz & mask);
            mask <<= 1;
            index |= (my & mask);
            mask <<= 1;
            index |= (mx & mask);
            mask <<= 1;

            mz <<= 2;
            my <<= 2;
            mx <<= 2;
        }

        return index;
    }


    /* @brief Compute the index of the cells in neighborhood of a given cell
   * @param OtreeHeight Height of the Octree
   * @param indexes target array to store the MortonIndexes computed
   * @param indexInArray store
   */
    __device__ static int GetNeighborsIndexes(const int3 coord, const int OctreeHeight, MortonIndex indexes[26], int indexInArray[26]) {
        int idxNeig = 0;
        int limite = 1 << (OctreeHeight - 1);
        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!Between(coord.x + idxX,0, limite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!Between(coord.y + idxY,0, limite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!Between(coord.z + idxZ,0, limite)) continue;

                    // if we are not on the current cell
                    if( idxX || idxY || idxZ ){
                        int3 other;

                        other.x = coord.x + idxX;
                        other.y = coord.y + idxY;
                        other.z = coord.z + idxZ;

                        indexes[ idxNeig ] = GetMortonIndex(other, OctreeHeight - 1);
                        indexInArray[ idxNeig ] = ((idxX+1)*3 + (idxY+1)) * 3 + (idxZ+1);
                        ++idxNeig;
                    }
                }
            }
        }
        return idxNeig;
    }

    __device__ static int GetInteractionNeighbors(const int3 coord, const int inLevel, MortonIndex inNeighbors[189], int inNeighborsPosition[189]) {
        // Then take each child of the parent's neighbors if not in directNeighbors
        // Father coordinate
        int3 parentCell;
        parentCell.x = coord.x>>1;
        parentCell.y = coord.y>>1;
        parentCell.z = coord.z>>1;

        // Limite at parent level number of box (split by 2 by level)
        const int limite = pow2(inLevel-1);

        int idxNeighbors = 0;
        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!Between(parentCell.x + idxX,0,limite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!Between(parentCell.y + idxY,0,limite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!Between(parentCell.z + idxZ,0,limite)) continue;

                    // if we are not on the current cell
                    if( idxX || idxY || idxZ ){
                        int3 otherParent;

                        otherParent.x = parentCell.x + idxX;
                        otherParent.y = parentCell.y + idxY;
                        otherParent.z = parentCell.z + idxZ;

                        const MortonIndex mortonOther = GetMortonIndex(otherParent, inLevel-1);

                        // For each child
                        for(int idxCousin = 0 ; idxCousin < 8 ; ++idxCousin){
                            const int xdiff  = ((otherParent.x<<1) | ( (idxCousin>>2) & 1)) - coord.x;
                            const int ydiff  = ((otherParent.y<<1) | ( (idxCousin>>1) & 1)) - coord.y;
                            const int zdiff  = ((otherParent.z<<1) | (idxCousin&1)) - coord.z;

                            // Test if it is a direct neighbor
                            if(Abs(xdiff) > 1 || Abs(ydiff) > 1 || Abs(zdiff) > 1){
                                // add to neighbors
                                inNeighborsPosition[idxNeighbors] = ((( (xdiff+3) * 7) + (ydiff+3))) * 7 + zdiff + 3;
                                inNeighbors[idxNeighbors++] = (mortonOther << 3) | idxCousin;
                            }
                        }
                    }
                }
            }
        }

        return idxNeighbors;
    }

    template <class NumType>
    __device__ static bool Between(const NumType inValue, const NumType inMin, const NumType inMax){
        return ( inMin <= inValue && inValue < inMax );
    }

    __device__ static int pow2(const int power){
        return (1 << power);
    }

    template <class NumType>
    __device__ static NumType Abs(const NumType inV){
        return (inV < 0 ? -inV : inV);
    }
};

#endif // FCUDATREECOORDINATE_HPP

