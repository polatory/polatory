// See LICENCE file at project root
#ifndef FPERLEAFLOADER_HPP
#define FPERLEAFLOADER_HPP

#include "../Utils/FPoint.hpp"
#include "../Utils/FGlobal.hpp"

template <class FReal>
class FPerLeafLoader {
    const FReal boxWidth;
    const FPoint<FReal> boxCenter;

    int NbSmallBoxesPerSide;
    FReal SmallBoxWidth;
    FReal SmallBoxWidthDiv2;
    int NbPart;

    int iterX;
    int iterY;
    int iterZ;

public:
    FPerLeafLoader(const int inTreeHeight, const FReal inBoxWidth = 1.0, const FPoint<FReal>& inBoxCenter = FPoint<FReal>())
        : boxWidth(inBoxWidth), boxCenter(inBoxCenter){
        NbSmallBoxesPerSide = (1 << (inTreeHeight-1));
        SmallBoxWidth = boxWidth / FReal(NbSmallBoxesPerSide);
        SmallBoxWidthDiv2 = SmallBoxWidth / 2;
        NbPart = NbSmallBoxesPerSide * NbSmallBoxesPerSide * NbSmallBoxesPerSide;

        iterX = 0;
        iterY = 0;
        iterZ = 0;
    }

    /** Default destructor */
    virtual ~FPerLeafLoader(){
    }

    /**
        * Get the number of particles for this simulation
        * @return number of particles that the loader can fill
        */
    FSize getNumberOfParticles() const{
        return NbPart;
    }


    /**
        * Get the center of the simulation box
        * @return box center needed by the octree
        */
    FPoint<FReal> getCenterOfBox() const{
        return boxCenter;
    }

    /**
        * Get the simulation box width
        * @return box width needed by the octree
        */
    FReal getBoxWidth() const{
        return boxWidth;
    }

    /**
        * To know if the loader is valide (file opened, etc.)
        * @return true if file is open
        */
    bool isOpen() const{
        return true;
    }

    void fillParticle(FPoint<FReal>*const inParticlePositions){
        inParticlePositions->setPosition(
               FReal(iterX)*SmallBoxWidth + SmallBoxWidthDiv2 + boxCenter.getX() - boxWidth/2.0,
               FReal(iterY)*SmallBoxWidth + SmallBoxWidthDiv2 + boxCenter.getY() - boxWidth/2.0,
               FReal(iterZ)*SmallBoxWidth + SmallBoxWidthDiv2 + boxCenter.getZ() - boxWidth/2.0);
        iterX += 1;
        if( iterX == NbSmallBoxesPerSide){
            iterX = 0;
            iterY += 1;
            if(iterY == NbSmallBoxesPerSide){
                iterY = 0;
                iterZ += 1;
            }
        }
    }
};

#endif // FPERLEAFLOADER_HPP
