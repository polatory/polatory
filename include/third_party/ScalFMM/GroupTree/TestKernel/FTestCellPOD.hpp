#ifndef FTESTCELLPOD_HPP
#define FTESTCELLPOD_HPP

#include "../../Utils/FGlobal.hpp"
#include "../Core/FBasicCellPOD.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

typedef FBasicCellPOD FTestCellPODCore;

typedef long long FTestCellPODData;

class FTestCellPOD {
protected:
    FTestCellPODCore* symb;
    FTestCellPODData* up;
    FTestCellPODData* down;

public:
    FTestCellPOD(FTestCellPODCore* inSymb, FTestCellPODData* inUp,
                 FTestCellPODData* inDown)
        : symb(inSymb), up(inUp), down(inDown){
    }
    FTestCellPOD()
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

    /** When doing the upward pass */
    long long int getDataUp() const {
        return (*up);
    }
    /** When doing the upward pass */
    void setDataUp(const long long int inData){
        (*up) = inData;
    }
    /** When doing the downard pass */
    long long int getDataDown() const {
        return (*down);
    }
    /** When doing the downard pass */
    void setDataDown(const long long int inData){
        (*down) = inData;
    }

    /** Make it like the begining */
    void resetToInitialState(){
        (*down) = 0;
        (*up)   = 0;
    }

};


#endif // FTESTCELLPOD_HPP
