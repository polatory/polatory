// See LICENCE file at project root
#ifndef FFMASCANFLOADER_HPP
#define FFMASCANFLOADER_HPP


#include <iostream>
#include <fstream>

#include "../Utils/FGlobal.hpp"
#include "FAbstractLoader.hpp"
#include "../Utils/FPoint.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FFmaScanfLoader
* Please read the license
*
* Load a file with a format like :
* NB_particles Box_width Box_X Box_Y Box_Z // init
* X Y Z // one particle by line
* ....
* @code
*    FFmaScanfLoader<FBasicParticle> loader("../Adir/Tests/particles.basic.txt"); <br>
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
class FFmaScanfLoader : public FAbstractLoader<FReal> {
protected:
    FILE* file;                 //< The file to read
    FPoint<FReal> centerOfBox;    //< The center of box read from file
    FReal boxWidth;             //< the box width read from file
    FSize nbParticles;            //< the number of particles read from file

public:
    /**
    * The constructor need the file name
    * @param filename the name of the file to open
    * you can test if file is successfuly open by calling hasNotFinished()
    */
    FFmaScanfLoader(const char* const filename): file(nullptr){
        file = fopen(filename,"r");
        // test if open
        if(this->file){
            float x,y,z, fBoxWidth;
            const int nbReadElements = fscanf(file,"%lld %f %f %f %f",&this->nbParticles,&fBoxWidth,&x,&y,&z);
            if(nbReadElements == 5){
                this->boxWidth = fBoxWidth * 2;
                this->centerOfBox.setPosition(x,y,z);
            }
            else{
                fclose(file);
                file = nullptr;
            }
        }
        else {
             this->boxWidth = 0;
             this->nbParticles = 0;
        }
    }

    /**
    * Default destructor, simply close the file
    */
    virtual ~FFmaScanfLoader(){
        if(file) fclose(file);
    }

    /**
      * To know if file is open and ready to read
      * @return true if loader can work
      */
    bool isOpen() const{
        return this->file != nullptr;
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
    void fillParticle(FPoint<FReal>*const inParticlePositions, FReal*const inPhysicalValue){
            float x,y,z,data;
            int unusedReturn = fscanf(this->file,"%f %f %f %f",&x,&y,&z,&data);
            unusedReturn += unusedReturn; // just to remove warning
            inParticlePositions->setPosition(x,y,z);
            (*inPhysicalValue) = data;
    }
};


#endif //FFMASCANFLOADER_HPP


