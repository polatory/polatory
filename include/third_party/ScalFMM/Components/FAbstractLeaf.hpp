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
#ifndef FABSTRACTLEAF_HPP
#define FABSTRACTLEAF_HPP

#include "../Utils/FPoint.hpp"
#include "../Utils/FLog.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FAbstractLeaf
* @brief This class is used to enable the use of typed particles (source XOR target) or simple system (source AND target).
*
* It has to be implemented has show in FSimpleLeaf.
* Leaf are stored in the octree.
*/
template<class FReal, class ContainerClass >
class FAbstractLeaf {
public:
    /** Default destructor */
    virtual ~FAbstractLeaf(){
    }

    /**
        * To add a new particle in the leaf
        * @param particle the new particle
        * Depending on the system to use the class that inherit
        * this interface can sort the particle as they like.
        */
    template<typename... Args>
    void push(const FPoint<FReal>& /*inParticlePosition*/, Args ... /*args*/){
        FLOG( FLog::Controller.write("Warning, push is not implemented!").write(FLog::Flush) );
    }

    /**
        * To get all the sources in a leaf
        * @return a pointer to the list of particles that are sources
        * Depending on the system to use the class that inherit
        * this interface can sort the particle as they like.
        */
    virtual ContainerClass* getSrc() = 0;

    /**
        * To get all the target in a leaf
        * @return a pointer to the list of particles that are targets
        * Depending on the system to use the class that inherit
        * this interface can sort the particle as they like.
        */
    virtual ContainerClass* getTargets() = 0;

};


#endif //FABSTRACTLEAF_HPP


