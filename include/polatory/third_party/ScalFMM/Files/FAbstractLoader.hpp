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


