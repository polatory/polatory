#ifndef FCUDAGROUPOFPARTICLES_HPP
#define FCUDAGROUPOFPARTICLES_HPP

#include "FCudaGlobal.hpp"
#include "../../Utils/FGlobal.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

template <class FReal, unsigned NbSymbAttributes, unsigned NbAttributesPerParticle, class AttributeClass = FReal>
class FCudaGroupOfParticles {
    /** One header is allocated at the beginning of each block */
    struct alignas(FStarPUDefaultAlign::StructAlign) BlockHeader{
        MortonIndex startingIndex;
        MortonIndex endingIndex;
        int numberOfLeavesInBlock;

        //< The real number of particles allocated
        FSize nbParticlesAllocatedInGroup;
        //< Starting point of position
        size_t offsetPosition;
        //< Bytes difference/offset between position
        size_t positionsLeadingDim;
        //< Bytes difference/offset between attributes
        size_t attributeLeadingDim;
        //< The total number of particles in the group
        FSize nbParticlesInGroup;
    };

    /** Information about a leaf */
    struct alignas(FStarPUDefaultAlign::StructAlign) LeafHeader {
        MortonIndex mindex;
        FSize nbParticles;
        size_t offSet;
    };


protected:
    static const int MemoryAlignementBytes     = FP2PDefaultAlignement;
    static const int MemoryAlignementParticles = MemoryAlignementBytes/sizeof(FReal);

    /** This function return the correct number of particles that should be used to have a correct pack.
     * If alignement is 32 and use double (so 4 particles in pack), then this function returns:
     * RoundToUpperParticles(1) = 1 + 3 = 4
     * RoundToUpperParticles(63) = 63 + 1 = 64
     */
    template <class NumClass>
    __device__ static NumClass RoundToUpperParticles(const NumClass& nbParticles){
        return nbParticles + (MemoryAlignementParticles - (nbParticles%MemoryAlignementParticles));
    }

    //< This value is for not used leaves
    static const int LeafIsEmptyFlag = -1;

    //< The size of memoryBuffer in byte
    size_t allocatedMemoryInByte;
    //< Pointer to a block memory
    unsigned char* memoryBuffer;

    //< Pointer to the header inside the block memory
    BlockHeader*    blockHeader;
    //< Pointer to leaves information
    LeafHeader*     leafHeader;

    //< Pointers to particle position x, y, z
    FReal* particlePosition[3];

    //< Pointers to the particles data inside the block memory
    AttributeClass* attributesBuffer;
    AttributeClass* particleAttributes[NbSymbAttributes+NbAttributesPerParticle];

public:
    /**
     * Init from a given buffer
     * @param inBuffer
     * @param inAllocatedMemoryInByte
     */
    __device__ FCudaGroupOfParticles(unsigned char* inBuffer, const size_t inAllocatedMemoryInByte,
                                     unsigned char* inAttributes)
        : allocatedMemoryInByte(inAllocatedMemoryInByte), memoryBuffer(inBuffer),
          blockHeader(nullptr), leafHeader(nullptr),
          attributesBuffer(nullptr){
        // Move the pointers to the correct position
        blockHeader         = reinterpret_cast<BlockHeader*>(inBuffer);
        inBuffer += sizeof(BlockHeader);
        leafHeader          = reinterpret_cast<LeafHeader*>(inBuffer);

        // Init particle pointers
        // Assert blockHeader->positionsLeadingDim == (sizeof(FReal) * blockHeader->nbParticlesAllocatedInGroup);
        particlePosition[0] = reinterpret_cast<FReal*>(memoryBuffer + blockHeader->offsetPosition);
        particlePosition[1] = (particlePosition[0] + blockHeader->nbParticlesAllocatedInGroup);
        particlePosition[2] = (particlePosition[1] + blockHeader->nbParticlesAllocatedInGroup);

        // Redirect pointer to data
        // Assert blockHeader->attributeLeadingDim == (sizeof(AttributeClass) * blockHeader->nbParticlesAllocatedInGroup);
        AttributeClass* symAttributes = (AttributeClass*)(&particlePosition[2][blockHeader->nbParticlesAllocatedInGroup]);
        for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes ; ++idxAttribute){
            particleAttributes[idxAttribute] = symAttributes;
            symAttributes += blockHeader->nbParticlesAllocatedInGroup;
        }
        if(inAttributes){
            attributesBuffer = (AttributeClass*)inAttributes;
            for(unsigned idxAttribute = 0 ; idxAttribute < NbAttributesPerParticle ; ++idxAttribute){
                particleAttributes[idxAttribute+NbSymbAttributes] = &attributesBuffer[idxAttribute*blockHeader->nbParticlesAllocatedInGroup];
            }
        }
    }

    /** The index of the fist leaf (set from the constructor) */
    __device__ MortonIndex getStartingIndex() const {
        return blockHeader->startingIndex;
    }

    /** The index of the last leaf + 1 (set from the constructor) */
    __device__ MortonIndex getEndingIndex() const {
        return blockHeader->endingIndex;
    }

    /** The number of leaf (set from the constructor) */
    __device__ int getNumberOfLeavesInBlock() const {
        return blockHeader->numberOfLeavesInBlock;
    }

    /** Get the total number of particles in the group */
    __device__ FSize getNbParticlesInGroup() const {
        return blockHeader->nbParticlesInGroup;
    }

    /** The size of the interval endingIndex-startingIndex (set from the constructor) */
    __device__ MortonIndex getSizeOfInterval() const {
        return (blockHeader->endingIndex-blockHeader->startingIndex);
    }

    /** Return true if inIndex should be located in the current block */
    __device__ bool isInside(const MortonIndex inIndex) const{
        return blockHeader->startingIndex <= inIndex && inIndex < blockHeader->endingIndex;
    }

    /** Return the idx in array of the cell */
    __device__ MortonIndex getLeafMortonIndex(const int id) const{
        return leafHeader[id].mindex;
    }

    /** Check if a cell exist (by binary search) and return it index */
    __device__ int getLeafIndex(const MortonIndex leafIdx) const{
        int idxLeft = 0;
        int idxRight = blockHeader->numberOfLeavesInBlock-1;
        while(idxLeft <= idxRight){
            const int idxMiddle = (idxLeft+idxRight)/2;
            if(leafHeader[idxMiddle].mindex == leafIdx){
                return idxMiddle;
            }
            if(leafIdx < leafHeader[idxMiddle].mindex){
                idxRight = idxMiddle-1;
            }
            else{
                idxLeft = idxMiddle+1;
            }
        }
        return -1;
    }

    /** Return true if inIndex is located in the current block and is not empty */
    __device__ bool exists(const MortonIndex inIndex) const {
        return isInside(inIndex) && (getLeafIndex(inIndex) != -1);
    }

    /** Return the address of the leaf if it exists (or NULL) */
    template<class ParticlesAttachedClass>
    __device__ ParticlesAttachedClass getLeaf(const int id){
        return ParticlesAttachedClass(leafHeader[id].nbParticles,
                                          particlePosition[0] + leafHeader[id].offSet,
                                            blockHeader->positionsLeadingDim,
                                            (attributesBuffer?particleAttributes[NbSymbAttributes] + leafHeader[id].offSet:nullptr),
                                            blockHeader->attributeLeadingDim);
    }
};

#endif // FCUDAGROUPOFPARTICLES_HPP

