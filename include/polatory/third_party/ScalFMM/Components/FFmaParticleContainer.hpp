// See LICENCE file at project root
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


