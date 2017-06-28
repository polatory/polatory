
#ifndef FGROUPATTACHEDLEAFDYN_HPP
#define FGROUPATTACHEDLEAFDYN_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Utils/FAssert.hpp"
#include "../../Utils/FPoint.hpp"

template <class FReal>
struct UnknownDescriptor{
    FPoint<FReal> pos;
    FSize originalIndex;
    MortonIndex mindex;
};


template <class FReal>
class FGroupAttachedLeafDyn {
protected:
    unsigned char* symbPart;
    unsigned char* downPart;

public:
    /** Empty constructor to point to nothing */
    FGroupAttachedLeafDyn() : symbPart(nullptr), downPart(nullptr) {
    }

    FGroupAttachedLeafDyn(unsigned char* inSymbPart, unsigned char* inDownPart)
        : symbPart(inSymbPart), downPart(inDownPart){

    }

    const unsigned char* getSymbPart() const {
        return symbPart;
    }

    unsigned char* getSymbPart() {
        return symbPart;
    }


    const unsigned char* getDownPart() const {
        return downPart;
    }

    unsigned char* getDownPart() {
        return downPart;
    }

    /** Return true if it has been attached to a memoy block */
    bool isAttachedToSomething() const {
        return symbPart != nullptr;
    }

    /** Allocate a new leaf by calling its constructor */
    template<class ParticleClassContainer>
    void copyFromContainer(const MortonIndex /*inMindex*/, const ParticleClassContainer* /*particles*/){
        static_assert(sizeof(ParticleClassContainer) == 0, "copyFromContainer should be implemented by the subclass");
    }

    virtual void init(const MortonIndex inIndex, const UnknownDescriptor<FReal> inParticles[],
              const FSize inNbParticles, const size_t inSymbSize, const size_t inDownSize) = 0;
};


#endif // FGROUPATTACHEDLEAFDYN_HPP

