
// Keep in private GIT
#ifndef FGROUPTESTPARTICLECONTAINER_HPP
#define FGROUPTESTPARTICLECONTAINER_HPP

#include "../Core/FGroupAttachedLeaf.hpp"

template <class FReal>
class FGroupTestParticleContainer : public FGroupAttachedLeaf<FReal, 0, 1, long long int> {
    typedef FGroupAttachedLeaf<FReal, 0, 1, long long int> Parent;

public:
    FGroupTestParticleContainer(){}
    FGroupTestParticleContainer(const FSize inNbParticles, FReal* inPositionBuffer, const size_t inLeadingPosition,
                                long long int* inAttributesBuffer, const size_t inLeadingAttributes)
        : Parent(inNbParticles, inPositionBuffer, inLeadingPosition, inAttributesBuffer, inLeadingAttributes) {

    }

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

#endif // FGROUPTESTPARTICLECONTAINER_HPP
