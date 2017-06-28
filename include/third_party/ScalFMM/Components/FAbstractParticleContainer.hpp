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
#ifndef FABSTRACTPARTICLECONTAINER_HPP
#define FABSTRACTPARTICLECONTAINER_HPP

#include "../Utils/FGlobal.hpp"
#include "../Utils/FLog.hpp"
#include "../Utils/FPoint.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @brief
* Please read the license
*
* This class define the method that every particle container
* has to implement.
*
* @warning Inherit from this class when implement a specific particle type
*/
template <class FReal>
class FAbstractParticleContainer {
public:
    /** Default destructor */
    virtual ~FAbstractParticleContainer(){
    }

    /**
     * This method should be inherited (or your leaf will do nothing)
     * the point is coming from the tree and is followed by what let the leaf
     * pass through its push method.
     */
    template<typename... Args>
    void push(const FPoint<FReal>& /*inParticlePosition*/, Args ... /*args*/){
        FLOG( FLog::Controller.write("Warning, push is not implemented!").write(FLog::Flush) );
    }
};


#endif //FABSTRACTPARTICLECONTAINER_HPP


