// See LICENCE file at project root
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


