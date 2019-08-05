#ifndef FTAYLORCELLPOD_HPP
#define FTAYLORCELLPOD_HPP


#include "../../Utils/FGlobal.hpp"
#include "../Core/FBasicCellPOD.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

typedef FBasicCellPOD FTaylorCellPODCore;

template <class FReal, int P, int order>
struct alignas(FStarPUDefaultAlign::StructAlign) FTaylorCellPODPole {
    //Size of Multipole Vector
    static const int MultipoleSize = ((P+1)*(P+2)*(P+3))*order/6;
    //Multipole vector
    FReal multipole_exp[MultipoleSize];
};

template <class FReal, int P, int order>
struct alignas(FStarPUDefaultAlign::StructAlign) FTaylorCellPODLocal {
    //Size of Local Vector
    static const int LocalSize = ((P+1)*(P+2)*(P+3))*order/6;
    //Local vector
    FReal local_exp[LocalSize];
};


template <class FReal, int P, int order>
class FTaylorCellPOD
{
    FTaylorCellPODCore* symb;
    FTaylorCellPODPole<FReal, P,order>* up;
    FTaylorCellPODLocal<FReal, P,order>* down;

public:
    FTaylorCellPOD(FTaylorCellPODCore* inSymb, FTaylorCellPODPole<FReal, P,order>* inUp,
              FTaylorCellPODLocal<FReal, P,order>* inDown): symb(inSymb), up(inUp), down(inDown){
    }

    FTaylorCellPOD()
        : symb(nullptr), up(nullptr), down(nullptr){
    }

    /** To get the morton index */
    MortonIndex getMortonIndex() const {
        return symb->mortonIndex;
    }

    /** To set the morton index */
    void setMortonIndex(const MortonIndex inMortonIndex) {
        symb->mortonIndex = inMortonIndex;
    }

    /** To get the cell level */
    int getLevel() const {
        return symb->level;
    }

    /** To set the cell level */
    void setLevel(const int level) {
        symb->level = level;
    }

    /** To get the position */
    FTreeCoordinate getCoordinate() const {
        return FTreeCoordinate(symb->coordinates[0],
                symb->coordinates[1], symb->coordinates[2]);
    }

    /** To set the position */
    void setCoordinate(const FTreeCoordinate& inCoordinate) {
        symb->coordinates[0] = inCoordinate.getX();
        symb->coordinates[1] = inCoordinate.getY();
        symb->coordinates[2] = inCoordinate.getZ();
    }

    /** To set the position from 3 FReals */
    void setCoordinate(const int inX, const int inY, const int inZ) {
        symb->coordinates[0] = inX;
        symb->coordinates[1] = inY;
        symb->coordinates[2] = inZ;
    }

    /** Get Multipole */
    const FReal* getMultipole() const
    {	return up->multipole_exp;
    }
    /** Get Local */
    const FReal* getLocal() const{
        return down->local_exp;
    }

    /** Get Multipole */
    FReal* getMultipole(){
        return up->multipole_exp;
    }
    /** Get Local */
    FReal* getLocal(){
        return down->local_exp;
    }

    /** To get the leading dim of a vec */
    int getVectorSize() const{
        return down->LocalSize;
    }

    /** Make it like the begining */
    void resetToInitialState(){
        memset(up->multipole_exp, 0, sizeof(FReal) * up->MultipoleSize);
        memset(down->local_exp,         0, sizeof(FReal) * down->LocalSize);
    }
};

#endif // FTAYLORCELLPOD_HPP
