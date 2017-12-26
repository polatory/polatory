// See LICENCE file at project root
#ifndef FTREECOORDINATE_HPP
#define FTREECOORDINATE_HPP

#include <string>

#include "../Utils/FGlobal.hpp"
#include "../Utils/FPoint.hpp"
#include "../Utils/FMath.hpp"

#include "../Components/FAbstractSerializable.hpp"

/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * @class FTreeCoordinate
 * Please read the license
 *
 * This class represents tree coordinate. It is used to save
 * the position in "box unit" (not system/space unit!).
 * It is directly related to morton index, as interleaves
 * bits from this coordinate make the morton index
 */
class FTreeCoordinate : public FAbstractSerializable, public FPoint<int, 3> {
private:
  using point_t = FPoint<int, 3>;
    enum {Dim = point_t::Dim};

public:
    /** Default constructor (position = {0,0,0})*/
    FTreeCoordinate(): point_t() {}

    /** Constructor from Morton index */
    [[deprecated]]
    explicit FTreeCoordinate(const MortonIndex mindex, const int) {
        setPositionFromMorton(mindex);
    }

    explicit FTreeCoordinate(const MortonIndex mindex) {
        setPositionFromMorton(mindex);
    }

    /** Constructor from args
     * @param inX the x
     * @param inY the y
     * @param inZ the z
     */
    explicit FTreeCoordinate(const int inX,const int inY,const int inZ)
        : point_t(inX, inY, inZ)
    {}

    explicit FTreeCoordinate(const int inPosition[3])
        : point_t(inPosition)
    {}

    /**
     * \brief Allow build from an FPoint object
     */
    using point_t::point_t;

    /**
     * Copy constructor
     */
    FTreeCoordinate(const FTreeCoordinate&) = default;

    /** Copy and increment constructor
     * @param other the source class to copy
     */
    FTreeCoordinate(const FTreeCoordinate& other, const int inOffset)
        : point_t(other, inOffset)
    {}

    /**
     * Copy assignment
     * @param other the source class to copy
     * @return this a reference to the current object
     */
    FTreeCoordinate& operator=(const FTreeCoordinate& other) = default;

    [[deprecated]]
    MortonIndex getMortonIndex(const int /*inLevel*/) const {
        return getMortonIndex();
    }


    /**
     * To get the morton index of the current position
     * @complexity inLevel
     * @param inLevel the level of the component
     * @return morton index
     */
    MortonIndex getMortonIndex() const{
        MortonIndex index = 0x0LL;
        MortonIndex mask = 0x1LL;
        // the order is xyz.xyz...
        MortonIndex mx = point_t::data()[0] << 2;
        MortonIndex my = point_t::data()[1] << 1;
        MortonIndex mz = point_t::data()[2];

        while( (mask <= mz)
               || ((mask << 1) <= my)
               || ((mask << 2) <= mx))
        {
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

    [[deprecated]]
    void setPositionFromMorton(MortonIndex inIndex, const int /*inLevel*/){
        setPositionFromMorton(inIndex);
    }

    /** This function set the position of the current object using a morton index
     * @param inIndex the morton index to compute position
     */
    void setPositionFromMorton(MortonIndex inIndex) {
        MortonIndex mask = 0x1LL;

        point_t::data()[0] = 0;
        point_t::data()[1] = 0;
        point_t::data()[2] = 0;

        while(inIndex >= mask) {
            point_t::data()[2] |= int(inIndex & mask);
            inIndex >>= 1;
            point_t::data()[1] |= int(inIndex & mask);
            inIndex >>= 1;
            point_t::data()[0] |= int(inIndex & mask);

            mask <<= 1;
        }
    }


    /** Test equal operator
     * @param other the coordinate to compare
     * @return true if other & current object have same position
     */
    bool equals(const int inX, const int inY, const int inZ) const {
        return point_t::data()[0] == inX
            && point_t::data()[1] == inY
            && point_t::data()[2] == inZ;
    }

    /** Use base class save */
    using point_t::save;
    /** Use base class restore */
    using point_t::restore;


    /** To know the size when we save it */
    FSize getSavedSize() const {
        return FSize(3 * sizeof(point_t::data()[0]));
    }


    static std::string MortonToBinary(MortonIndex index, int level){
        std::string str;
        int bits = 1 << ((level * 3) - 1);
        int dim = 0;
        while(bits){
            if(index & bits) str.append("1");
            else str.append("0");
            bits >>= 1;
            // we put a dot each 3 values
            if(++dim == 3){
                str.append(".");
                dim = 0;
            }
        }
        return str;
    }

    /** @brief Compute the index of the cells in neighborhood of a given cell
     * @param OtreeHeight Height of the Octree
     * @param indexes target array to store the MortonIndexes computed
     * @param indexInArray store (must have the same length as indexes)
     */
    int getNeighborsIndexes(const int OctreeHeight, MortonIndex indexes[26], int* indexInArray = nullptr) const {
        int idxNeig = 0;
        int limite = 1 << (OctreeHeight - 1);
        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(this->getX() + idxX,0, limite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(this->getY() + idxY,0, limite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(this->getZ() + idxZ,0, limite)) continue;

                    // if we are not on the current cell
                    if( idxX || idxY || idxZ ){
                        const FTreeCoordinate other(this->getX() + idxX, this->getY() + idxY, this->getZ() + idxZ);
                        indexes[ idxNeig ] = other.getMortonIndex();
                        if(indexInArray)
                            indexInArray[ idxNeig ] = ((idxX+1)*3 + (idxY+1)) * 3 + (idxZ+1);
                        ++idxNeig;
                    }
                }
            }
        }
        return idxNeig;
    }

    /**
     * @param inNeighborsPosition (must have the same length as inNeighbors)
     */
    int getInteractionNeighbors(const int inLevel, MortonIndex inNeighbors[/*189+26+1*/216], int* inNeighborsPosition,
                                const int neighSeparation = 1) const {
        // Then take each child of the parent's neighbors if not in directNeighbors
        // Father coordinate
        const FTreeCoordinate parentCell(this->getX()>>1,this->getY()>>1,this->getZ()>>1);

        // Limite at parent level number of box (split by 2 by level)
        const int limite = FMath::pow2(inLevel-1);

        int idxNeighbors = 0;
        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(parentCell.getX() + idxX,0,limite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(parentCell.getY() + idxY,0,limite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(parentCell.getZ() + idxZ,0,limite)) continue;

                    // if we are not on the current cell
                    if(neighSeparation<1 || idxX || idxY || idxZ ){
                        const FTreeCoordinate otherParent(parentCell.getX() + idxX,parentCell.getY() + idxY,parentCell.getZ() + idxZ);
                        const MortonIndex mortonOther = otherParent.getMortonIndex();

                        // For each child
                        for(int idxCousin = 0 ; idxCousin < 8 ; ++idxCousin){
                            const int xdiff  = ((otherParent.getX()<<1) | ( (idxCousin>>2) & 1)) - this->getX();
                            const int ydiff  = ((otherParent.getY()<<1) | ( (idxCousin>>1) & 1)) - this->getY();
                            const int zdiff  = ((otherParent.getZ()<<1) | (idxCousin&1)) - this->getZ();

                            // Test if it is a direct neighbor
                            if(FMath::Abs(xdiff) > neighSeparation || FMath::Abs(ydiff) > neighSeparation || FMath::Abs(zdiff) > neighSeparation){
                                // add to neighbors
                                if(inNeighborsPosition)
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

    int getInteractionNeighbors(const int inLevel, MortonIndex inNeighbors[/*189+26+1*/216], const int neighSeparation = 1) const{
        return getInteractionNeighbors(inLevel, inNeighbors, nullptr, neighSeparation);
    }

};



#endif //FTREECOORDINATE_HPP
