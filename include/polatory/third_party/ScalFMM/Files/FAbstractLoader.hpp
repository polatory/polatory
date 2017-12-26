// See LICENCE file at project root
#ifndef FABSTRACTLOADER_HPP
#define FABSTRACTLOADER_HPP


#include "../Utils/FGlobal.hpp"
#include "../Utils/FPoint.hpp"


/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FAbstractLoader
* Please read the license
*
* A loader is the component that fills an octree.
*
* If you want to use a specific file format you then need to inherit from this loader
* and implement several methods.
*
* Please look at FBasicLoader or FFmaLoader to see an example.
*
* @warning Inherit from this class when defining a loader class
*/
template <class FReal>
class FAbstractLoader {
public:	
    /** Default destructor */
    virtual ~FAbstractLoader(){
    }

    /**
        * Get the number of particles for this simulation
        * @return number of particles that the loader can fill
        */
    virtual FSize getNumberOfParticles() const = 0;

    /**
        * Get the center of the simulation box
        * @return box center needed by the octree
        */
    virtual FPoint<FReal> getCenterOfBox() const = 0;

    /**
        * Get the simulation box width
        * @return box width needed by the octree
        */
    virtual FReal getBoxWidth() const = 0;

    /**
        * To know if the loader is valide (file opened, etc.)
        * @return true if file is open
        */
    virtual bool isOpen() const = 0;
};


#endif //FABSTRACTLOADER_HPP


