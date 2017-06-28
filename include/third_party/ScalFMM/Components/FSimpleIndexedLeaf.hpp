// ===================================================================================
// Copyright ScalFmm 2014 INRIA
// olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.  
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info". 
// "http://www.gnu.org/licenses".
// ===================================================================================
#ifndef FSIMPLELINDEXEAF_HPP
#define FSIMPLELINDEXEAF_HPP


#include "FSimpleLeaf.hpp"

/**
* @author Olivier Coulaud (Olivier.Coulaud@inria.fr)
* @class FSimpleIndexedLeaf
* @brief This class is used as a leaf in simple system (source AND target).
* Here there only one container stores all particles.
*/
template< class FReal, class ContainerClass >
class FSimpleIndexedLeaf : public FSimpleLeaf<FReal, ContainerClass> {

	long int index ; //  Index of the leaf. useful for debug purpose
public:
    /** Default destructor */
    virtual ~FSimpleIndexedLeaf(){
    }

    /**
    * To get all the index of a leaf
    * @return the index of the leaf
    */
    long int getIndex() {
        return this->index;
    }

    /**
    * To set all the index of a leaf
    * @param id the index of the leaf
    */
    void setIndex(const long int &id ) {
        this->index = id;
    }


};


#endif //FSIMPLELEAF_HPP


