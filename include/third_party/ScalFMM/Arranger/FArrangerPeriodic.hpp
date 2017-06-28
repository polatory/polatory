// ===================================================================================
// Copyright ScalFmm 2016 INRIA, Olivier Coulaud, Bérenger Bramas,
// Matthias Messner olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the
// FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
// An extension to the license is given to allow static linking of scalfmm
// inside a proprietary application (no matter its license).
// See the main license file for more details.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================

#ifndef FARRANGERPERIODIC_HPP
#define FARRANGERPERIODIC_HPP

#include "Utils/FGlobalPeriodic.hpp"
#include "Arranger/FOctreeArranger.hpp"

template <class FReal, class OctreeClass, class ContainerClass, class LeafInterface >
class FArrangerPeriodic : public FOctreeArranger<FReal,OctreeClass,ContainerClass,LeafInterface>{

    FReal getPeriodicPos(FReal pos, PeriodicCondition periodicPlus, PeriodicCondition periodicMinus,FReal maxDir,FReal minDir,const int dir){
        FReal res = pos;
        if( TestPeriodicCondition(dir, periodicPlus) ){
            while(res >= maxDir){
                res += (-(this->boxWidth));
            }
        }

        if( TestPeriodicCondition(dir, periodicMinus) ){
            while(res < minDir){
                res += (this->boxWidth);
            }
        }
        return res;
    }

public:

    FArrangerPeriodic(OctreeClass * octree) : FOctreeArranger<FReal,OctreeClass,ContainerClass,LeafInterface>(octree){
    }

    // To put in inhereed class
    void checkPosition(FPoint<FReal>& particlePos) override {
        particlePos.setX( getPeriodicPos( particlePos.getX(), DirMinusX, DirPlusX, (this->MaxBox).getX(),(this->MinBox).getX(),DirX));
        particlePos.setY( getPeriodicPos( particlePos.getY(), DirMinusY, DirPlusY, (this->MaxBox).getY(),(this->MinBox).getY(),DirY));
        particlePos.setZ( getPeriodicPos( particlePos.getZ(), DirMinusZ, DirPlusZ, (this->MaxBox).getZ(),(this->MinBox).getZ(),DirZ));
    }
};


#endif // FCONVERTERPERIODIC_HPP
