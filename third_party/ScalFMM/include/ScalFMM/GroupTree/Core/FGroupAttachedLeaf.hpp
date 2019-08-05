
// Keep in private GIT
#ifndef FGROUPATTACHEDLEAF_HPP
#define FGROUPATTACHEDLEAF_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Utils/FAssert.hpp"

/**
 * This class is "attached" to a buffer.
 * It is a wrapper/interface to a memory area that stores all the particles info.
 * The idea is to hidde the group allocation from the group tree but
 * to keep the same interface than the FBasicParticlesContainer.
 */
template <class FReal, unsigned NbSymbAttributes, unsigned NbAttributesPerParticle, class AttributeClass = FReal>
class FGroupAttachedLeaf {
protected:
    //< Nb of particles in the current leaf
    FSize nbParticles;
    //< Pointers to the positions of the particles
    FReal* positionsPointers[3];
    //< Pointers to the attributes of the particles
    AttributeClass* attributes[NbSymbAttributes+NbAttributesPerParticle];

public:
    /** Empty constructor to point to nothing */
    FGroupAttachedLeaf() : nbParticles(-1) {
        memset(positionsPointers, 0, sizeof(FReal*) * 3);
        memset(attributes, 0, sizeof(AttributeClass*) * (NbSymbAttributes+NbAttributesPerParticle));
    }

    /**
     * @brief FGroupAttachedLeaf
     * @param inNbParticles the number of particles in the leaf
     * @param inPositionBuffer the memory address of the X array of particls
     * @param inLeadingPosition each position is access by inPositionBuffer + in bytes inLeadingPosition*idx
     * @param inAttributesBuffer the memory address of the first attribute
     * @param inLeadingAttributes each attribute is access by inAttributesBuffer + in bytes inLeadingAttributes*idx
     */
    FGroupAttachedLeaf(const FSize inNbParticles, FReal* inPositionBuffer, const size_t inLeadingPosition,
                       AttributeClass* inAttributesBuffer, const size_t inLeadingAttributes)
        : nbParticles(inNbParticles){
        // Redirect pointers to position
        positionsPointers[0] = inPositionBuffer;
        positionsPointers[1] = reinterpret_cast<FReal*>(reinterpret_cast<unsigned char*>(inPositionBuffer) + inLeadingPosition);
        positionsPointers[2] = reinterpret_cast<FReal*>(reinterpret_cast<unsigned char*>(inPositionBuffer) + inLeadingPosition*2);

        for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes ; ++idxAttribute){
            attributes[idxAttribute] = reinterpret_cast<AttributeClass*>(reinterpret_cast<unsigned char*>(inPositionBuffer) + inLeadingPosition*(idxAttribute+3));
        }

        // Redirect pointers to data
        if(inAttributesBuffer){
            for(unsigned idxAttribute = 0 ; idxAttribute < NbAttributesPerParticle ; ++idxAttribute){
                attributes[idxAttribute+NbSymbAttributes] = reinterpret_cast<AttributeClass*>(reinterpret_cast<unsigned char*>(inAttributesBuffer) + idxAttribute*inLeadingAttributes);
            }
        }
        else{
            memset(&attributes[NbSymbAttributes], 0, sizeof(AttributeClass*)*NbAttributesPerParticle);
        }
    }    

    /** Copy the attached group to another one (copy the pointer not the content!) */
    FGroupAttachedLeaf(const FGroupAttachedLeaf& other) : nbParticles(other.nbParticles) {
        positionsPointers[0] = other.positionsPointers[0];
        positionsPointers[1] = other.positionsPointers[1];
        positionsPointers[2] = other.positionsPointers[2];

        // Redirect pointers to data
        for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes+NbAttributesPerParticle ; ++idxAttribute){
            attributes[idxAttribute] = other.attributes[idxAttribute];
        }
    }

    /** Copy the attached group to another one (copy the pointer not the content!) */
    FGroupAttachedLeaf& operator=(const FGroupAttachedLeaf& other){
        nbParticles = (other.nbParticles);

        positionsPointers[0] = other.positionsPointers[0];
        positionsPointers[1] = other.positionsPointers[1];
        positionsPointers[2] = other.positionsPointers[2];

        // Redirect pointers to data
        for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes+NbAttributesPerParticle ; ++idxAttribute){
            attributes[idxAttribute] = other.attributes[idxAttribute];
        }

        return (*this);
    }

    /**
     * @brief getNbParticles
     * @return the number of particles in the leaf
     */
    FSize getNbParticles() const{
        return nbParticles;
    }

    /**
     * @brief getPositions
     * @return a FReal*[3] to get access to the positions
     */
    const FReal*const* getPositions() const {
        return positionsPointers;
    }

    /**
     * @brief getWPositions
     * @return get the position in write mode
     */
    FReal* const* getWPositions() {
        return positionsPointers;
    }

    /**
     * @brief getAttribute
     * @param index
     * @return the attribute at index index
     */
    AttributeClass* getAttribute(const int index) {
        return attributes[index];
    }

    /**
     * @brief getAttribute
     * @param index
     * @return
     */
    const AttributeClass* getAttribute(const int index) const {
        return attributes[index];
    }

    /**
     * Get the attribute with a forcing compile optimization
     */
    template <int index>
    AttributeClass* getAttribute() {
        static_assert(index < NbSymbAttributes+NbAttributesPerParticle, "Index to get attributes is out of scope.");
        return attributes[index];
    }

    /**
     * Get the attribute with a forcing compile optimization
     */
    template <int index>
    const AttributeClass* getAttribute() const {
        static_assert(index < NbSymbAttributes+NbAttributesPerParticle, "Index to get attributes is out of scope.");
        return attributes[index];
    }

    /** Return true if it has been attached to a memoy block */
    bool isAttachedToSomething() const {
        return nbParticles != -1;
    }

    /** Allocate a new leaf by calling its constructor */
    template<class ParticleClassContainer>
    void copyFromContainer(const ParticleClassContainer* particles, const FSize offsetInSrcContainer){
        FAssertLF(isAttachedToSomething());
        // Copy position
        memcpy(positionsPointers[0], particles->getPositions()[0] + offsetInSrcContainer, nbParticles*sizeof(FReal));
        memcpy(positionsPointers[1], particles->getPositions()[1] + offsetInSrcContainer, nbParticles*sizeof(FReal));
        memcpy(positionsPointers[2], particles->getPositions()[2] + offsetInSrcContainer, nbParticles*sizeof(FReal));

        // Copy data
        for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes+NbAttributesPerParticle ; ++idxAttribute){
            memcpy(attributes[idxAttribute], particles->getAttribute(idxAttribute) + offsetInSrcContainer, nbParticles*sizeof(AttributeClass));
        }
    }


    /** Copy data for one particle (from the ParticleClassContainer into the attached buffer) */
    template<class ParticleClassContainer>
    void setParticle(const FSize destPartIdx, const FSize srcPartIdx, const ParticleClassContainer* particles){
        // Copy position
        positionsPointers[0][destPartIdx] = particles->getPositions()[0][srcPartIdx];
        positionsPointers[1][destPartIdx] = particles->getPositions()[1][srcPartIdx];
        positionsPointers[2][destPartIdx] = particles->getPositions()[2][srcPartIdx];

        // Copy data
        for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes+NbAttributesPerParticle ; ++idxAttribute){
#ifndef SCALFMM_SIMGRID_NODATA
            attributes[idxAttribute][destPartIdx] = particles->getAttribute(idxAttribute)[srcPartIdx];
#endif
        }
    }

};

#endif // FGROUPATTACHEDLEAF_HPP
