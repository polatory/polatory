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
#ifndef FHLOADER_HPP
#define FHLOADER_HPP


#include <cstdlib>
#include <ctime>


#include "../Utils/FGlobal.hpp"

#include "FAbstractLoader.hpp"
#include "../Utils/FPoint.hpp"
#include "../Components/FParticleType.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FRandomLoader
* Please read the license
*/
template <class FReal>
class FRandomLoader : public FAbstractLoader<FReal> {
protected:
    const FSize nbParticles;            //< the number of particles
    const FReal boxWidth;             //< the box width
    const FPoint<FReal> centerOfBox;    //< The center of box

public:
    /**
    * The constructor need the simulation data
    *  @param   inNbParticles Number of partcles to generate randomly
    *  @param  inBoxWidth     the width of the box
    *  @param  inCenterOfBox the center of the box
    *  @param  inSeed The seed for the random generator (default value time(nullptr))
    *
    */
    FRandomLoader(const FSize inNbParticles, const FReal inBoxWidth = 1.0,
                  const FPoint<FReal>& inCenterOfBox = FPoint<FReal>(0,0,0),
                  const unsigned int inSeed = static_cast<unsigned int>(time(nullptr)))
        : nbParticles(inNbParticles), boxWidth(inBoxWidth), centerOfBox(inCenterOfBox) {
        srand48(inSeed);
    }
    /**
    * Default destructor
    */
    virtual ~FRandomLoader(){
    }

    /**
      * @return true
      */
    bool isOpen() const{
        return true;
    }

    /**
      * To get the number of particles from this loader
      * @param the number of particles the loader can fill
      */
    FSize getNumberOfParticles() const{
        return FSize(this->nbParticles);
    }

    /**
      * The center of the box
      * @return box center
      */
    FPoint<FReal> getCenterOfBox() const{
        return this->centerOfBox;
    }

    /**
      * The box width
      * @return box width
      */
    FReal getBoxWidth() const{
        return this->boxWidth;
    }

    /**
      * Fill a particle
      * @warning to work with the loader, particles has to expose a setPosition method
      * @param the particle to fill
      */
    void fillParticle(FPoint<FReal>*const inParticlePositions){
        inParticlePositions->setPosition(
                    (getRandom() * boxWidth) + centerOfBox.getX() - boxWidth/2,
                    (getRandom() * boxWidth) + centerOfBox.getY() - boxWidth/2,
                    (getRandom() * boxWidth) + centerOfBox.getZ() - boxWidth/2);
    }

    /** Get a random number between 0 & 1 */
    FReal getRandom() const{
        return FReal(drand48());
    }
};


/** This class is a random loader but it also generate
  * randomly the particles type (target or source)
  */
template <class FReal>
class FRandomLoaderTsm : public FRandomLoader<FReal> {
public:
    FRandomLoaderTsm(const FSize inNbParticles, const FReal inBoxWidth = 1.0,
                  const FPoint<FReal>& inCenterOfBox = FPoint<FReal>(0,0,0), const unsigned int inSeed = static_cast<unsigned int>(time(nullptr)))
        : FRandomLoader<FReal>(inNbParticles,inBoxWidth,inCenterOfBox,inSeed) {
    }


    void fillParticle(FPoint<FReal>*const inParticlePositions, FParticleType*const isTarget){
        FRandomLoader<FReal>::fillParticle(inParticlePositions);
        if(FRandomLoader<FReal>::getRandom() > 0.5 ) (*isTarget) = FParticleType::FParticleTypeTarget;
        else (*isTarget) = FParticleType::FParticleTypeSource;
    }
};


#endif //FHLOADER_HPP


