// See LICENCE file at project root

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
