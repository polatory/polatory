// See LICENCE file at project root
#ifndef FFMATSMLOADER_HPP
#define FFMATSMLOADER_HPP


#include <iostream>
#include <fstream>

#include "../Utils/FGlobal.hpp"
#include "FAbstractLoader.hpp"
#include "../Utils/FPoint.hpp"
#include "../Components/FParticleType.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FFmaTsmLoader
* Please read the license
*
* Load a file with a format like :
* NB_particles Box_width Box_X Box_Y Box_Z // init
* X Y Z PhysicalValue  type // one particle by line
* ....
* @code
*    FFmaTsmLoader<FBasicParticle> loader("../Adir/Tests/particles.basic.txt"); <br>
*    if(!loader.isOpen()){ <br>
*        std::cout << "Loader Error\n"; <br>
*        return 1; <br>
*    } <br>
* <br>
*    FOctree<FBasicParticle, TestCell, FSimpleLeaf> tree(loader.getBoxWidth(),loader.getCenterOfBox()); <br>
* <br>
*    for(FSize r.getNumberOfParticles() ; ++idx){ <br>
*        FBasicParticle* const part = new FBasicParticle(); <br>
*        loader.fillParticle(part); <br>
*        tree.insert(part); <br>
*    } <br>
* @endcode
*
* Particle has to extend {FExtendPhysicalValue,FExtendPosition}
*/
template <class FReal>
class FFmaTsmLoader : public FAbstractLoader<FReal> {
protected:
    std::ifstream file;         //< The file to read
    FPoint<FReal> centerOfBox;    //< The center of box read from file
    FReal boxWidth;             //< the box width read from file
    FSize nbParticles;           //< the number of particles read from file

public:
    /**
    * The constructor need the file name
    * @param filename the name of the file to open
    * you can test if file is successfuly open by calling hasNotFinished()
    */
    FFmaTsmLoader(const char* const filename): file(filename,std::ifstream::in){
        // test if open
        if(this->file.is_open()){
            FReal x,y,z;
            this->file >> this->nbParticles >> this->boxWidth >> x >> y >> z;
            this->centerOfBox.setPosition(x,y,z);
            this->boxWidth *= 2;
        }
        else {
             this->boxWidth = 0;
             this->nbParticles = 0;
        }
    }

    /**
    * Default destructor, simply close the file
    */
    virtual ~FFmaTsmLoader(){
        file.close();
    }

    /**
      * To know if file is open and ready to read
      * @return true if loader can work
      */
    bool isOpen() const{
        return this->file.is_open() && !this->file.eof();
    }

    /**
      * To get the number of particles from this loader
      * @param the number of particles the loader can fill
      */
    FSize getNumberOfParticles() const{
        return FSize(this->nbParticles);
    }

    /**
      * The center of the box from the simulation file opened by the loader
      * @return box center
      */
    FPoint<FReal> getCenterOfBox() const{
        return this->centerOfBox;
    }

    /**
      * The box width from the simulation file opened by the loader
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
    void fillParticle(FPoint<FReal>*const inParticlePositions, FReal*const inPhysicalValue, FParticleType*const particleType){
        FReal x,y,z,data;
        int isTarget;
        this->file >> x >> y >> z >> data >> isTarget;

        inParticlePositions->setPosition(x,y,z);
        *inPhysicalValue = data;
        if(isTarget) (*particleType) = FParticleType::FParticleTypeTarget;
        else         (*particleType) = FParticleType::FParticleTypeSource;
    }

};


#endif //FFMATSMLOADER_HPP


