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
#ifndef FFMAPARTICLECONTAINER_HPP
#define FFMAPARTICLECONTAINER_HPP

#include "FBasicParticleContainer.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* Please read the license
*
* This class defines a particle container for FMA loader.
* position + 1 real for the physical value
*/
template <class FReal>
class FFmaParticleContainer : public FBasicParticleContainer<FReal, 1> {
    typedef FBasicParticleContainer<1> Parent;

public:
    /**
     * @brief getPhysicalValues to get the array of physical values
     * @return
     */
    FReal* getPhysicalValues(){
        return Parent::getAttribute<0>();
    }

    /**
     * @brief getPhysicalValues to get the array of physical values
     * @return
     */
    const FReal* getPhysicalValues() const {
        return Parent::getAttribute<0>();
    }
};


#endif //FFMAPARTICLECONTAINER_HPP


