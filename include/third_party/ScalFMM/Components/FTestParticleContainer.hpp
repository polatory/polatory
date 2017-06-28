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
#ifndef FTESTPARTICLECONTAINER_HPP
#define FTESTPARTICLECONTAINER_HPP


#include "FBasicParticleContainer.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FTestParticle
* Please read the license
*
* This class is used in the FTestKernels, please
* look at this class to know whit it is.
* We store the positions + 1 long long int
*/
template <class FReal>
class FTestParticleContainer : public FBasicParticleContainer<FReal, 1, long long int> {
    typedef FBasicParticleContainer<FReal, 1, long long int> Parent;

public:
    /**
     * @brief getDataDown
     * @return
     */
    long long int* getDataDown(){
        return Parent::template getAttribute<0>();
    }

    /**
     * @brief getDataDown
     * @return
     */
    const long long int* getDataDown() const {
        return Parent::template getAttribute<0>();
    }
};


#endif //FTESTPARTICLECONTAINER_HPP


