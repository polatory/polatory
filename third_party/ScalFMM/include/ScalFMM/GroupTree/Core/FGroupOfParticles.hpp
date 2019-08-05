
// Keep in private GIT
#ifndef FGROUPOFPARTICLES_HPP
#define FGROUPOFPARTICLES_HPP


#include "../../Utils/FGlobal.hpp"
#include "../../Utils/FAssert.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../../Utils/FAlignedMemory.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

#include <list>
#include <functional>


/**
* @brief The FGroupOfParticles class manages the leaves in block allocation.
*/
template <class FReal, unsigned NbSymbAttributes, unsigned NbAttributesPerParticle, class AttributeClass = FReal>
class FGroupOfParticles {
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
    static NumClass RoundToUpperParticles(const NumClass& nbParticles){
        return nbParticles + (MemoryAlignementParticles - (nbParticles%MemoryAlignementParticles));
    }

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
#ifndef SCALFMM_SIMGRID_NODATA
    AttributeClass* particleAttributes[NbSymbAttributes+NbAttributesPerParticle];
#else
    AttributeClass* particleAttributes[NbSymbAttributes];
#endif

    /** To know if we have to delete the buffer */
    bool deleteBuffer;

public:
    typedef AttributeClass ParticleDataType;

    /**
     * Init from a given buffer
     * @param inBuffer
     * @param inAllocatedMemoryInByte
     */
    FGroupOfParticles(unsigned char* inBuffer, const size_t inAllocatedMemoryInByte,
                      unsigned char* inAttributes)
        : allocatedMemoryInByte(inAllocatedMemoryInByte), memoryBuffer(inBuffer),
          blockHeader(nullptr), leafHeader(nullptr),
          attributesBuffer(nullptr), deleteBuffer(false){
        // Move the pointers to the correct position
        blockHeader         = reinterpret_cast<BlockHeader*>(inBuffer);
        inBuffer += sizeof(BlockHeader);
        leafHeader          = reinterpret_cast<LeafHeader*>(inBuffer);

        // Init particle pointers
        FAssertLF( blockHeader->positionsLeadingDim == (sizeof(FReal) * blockHeader->nbParticlesAllocatedInGroup) );
        particlePosition[0] = reinterpret_cast<FReal*>(memoryBuffer + blockHeader->offsetPosition);
        particlePosition[1] = (particlePosition[0] + blockHeader->nbParticlesAllocatedInGroup);
        particlePosition[2] = (particlePosition[1] + blockHeader->nbParticlesAllocatedInGroup);

        // Redirect pointer to data
        FAssertLF(blockHeader->attributeLeadingDim == (sizeof(AttributeClass) * blockHeader->nbParticlesAllocatedInGroup));
        AttributeClass* symAttributes = (AttributeClass*)(&particlePosition[2][blockHeader->nbParticlesAllocatedInGroup]);
        for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes ; ++idxAttribute){
            particleAttributes[idxAttribute] = symAttributes;
            symAttributes += blockHeader->nbParticlesAllocatedInGroup;
        }
#ifndef SCALFMM_SIMGRID_NODATA
        if(inAttributes){
            attributesBuffer = (AttributeClass*)inAttributes;
            for(unsigned idxAttribute = 0 ; idxAttribute < NbAttributesPerParticle ; ++idxAttribute){
                particleAttributes[idxAttribute+NbSymbAttributes] = &attributesBuffer[idxAttribute*blockHeader->nbParticlesAllocatedInGroup];
            }
        }
#endif
    }

    /**
 * @brief FGroupOfParticles
 * @param inStartingIndex first leaf morton index
 * @param inEndingIndex last leaf morton index + 1
 * @param inNumberOfLeaves total number of leaves in the interval (should be <= inEndingIndex-inEndingIndex)
 */
    FGroupOfParticles(const MortonIndex inStartingIndex, const MortonIndex inEndingIndex, const int inNumberOfLeaves, const FSize inNbParticles)
        : allocatedMemoryInByte(0), memoryBuffer(nullptr), blockHeader(nullptr), leafHeader(nullptr),
          deleteBuffer(true){
        memset(particlePosition, 0, sizeof(particlePosition));
        memset(particleAttributes, 0, sizeof(particleAttributes));

        const FSize nbParticlesAllocatedInGroup = RoundToUpperParticles(inNbParticles+(MemoryAlignementParticles-1)*inNumberOfLeaves);

        // Find the number of leaf to allocate in the blocks
        FAssertLF((inEndingIndex-inStartingIndex) >= MortonIndex(inNumberOfLeaves));
        // Total number of bytes in the block
        const size_t sizeOfOneParticle = (3*sizeof(FReal) + NbSymbAttributes*sizeof(AttributeClass));
        const size_t memoryToAlloc = sizeof(BlockHeader)
                                    + (inNumberOfLeaves*sizeof(MortonIndex))
                                    + (inNumberOfLeaves*sizeof(LeafHeader))
                                    + nbParticlesAllocatedInGroup*sizeOfOneParticle;

        // Allocate
        FAssertLF(0 <= int(memoryToAlloc) && int(memoryToAlloc) < std::numeric_limits<int>::max());
        allocatedMemoryInByte = memoryToAlloc;
        memoryBuffer = (unsigned char*)FAlignedMemory::AllocateBytes<MemoryAlignementBytes>(memoryToAlloc);
        FAssertLF(memoryBuffer);
        memset(memoryBuffer, 0, memoryToAlloc);

        // Move the pointers to the correct position
        unsigned char* bufferPtr = memoryBuffer;
        blockHeader         = reinterpret_cast<BlockHeader*>(bufferPtr);
        bufferPtr += sizeof(BlockHeader);
        leafHeader          = reinterpret_cast<LeafHeader*>(bufferPtr);

        // Init header
        blockHeader->startingIndex = inStartingIndex;
        blockHeader->endingIndex   = inEndingIndex;
        blockHeader->numberOfLeavesInBlock  = inNumberOfLeaves;
        blockHeader->nbParticlesAllocatedInGroup = nbParticlesAllocatedInGroup;
        blockHeader->nbParticlesInGroup = inNbParticles;

        // Init particle pointers
        blockHeader->positionsLeadingDim = (sizeof(FReal) * nbParticlesAllocatedInGroup);
        particlePosition[0] = reinterpret_cast<FReal*>((reinterpret_cast<size_t>(leafHeader + inNumberOfLeaves)
                                                       +MemoryAlignementBytes-1) & ~(MemoryAlignementBytes-1));
        particlePosition[1] = (particlePosition[0] + nbParticlesAllocatedInGroup);
        particlePosition[2] = (particlePosition[1] + nbParticlesAllocatedInGroup);

        blockHeader->offsetPosition = size_t(particlePosition[0]) - size_t(memoryBuffer);

        // Redirect pointer to data
        blockHeader->attributeLeadingDim = (sizeof(AttributeClass) * nbParticlesAllocatedInGroup);

        AttributeClass* symAttributes = (AttributeClass*)(&particlePosition[2][blockHeader->nbParticlesAllocatedInGroup]);
        for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes ; ++idxAttribute){
            particleAttributes[idxAttribute] = symAttributes;
            symAttributes += blockHeader->nbParticlesAllocatedInGroup;
        }
#ifndef SCALFMM_SIMGRID_NODATA
        attributesBuffer = (AttributeClass*)FAlignedMemory::AllocateBytes<MemoryAlignementBytes>(blockHeader->attributeLeadingDim*NbAttributesPerParticle);
        memset(attributesBuffer, 0, blockHeader->attributeLeadingDim*NbAttributesPerParticle);
        for(unsigned idxAttribute = 0 ; idxAttribute < NbAttributesPerParticle ; ++idxAttribute){
            particleAttributes[idxAttribute+NbSymbAttributes] = &attributesBuffer[idxAttribute*nbParticlesAllocatedInGroup];
        }
#else
        attributesBuffer = nullptr;
#endif

        // Set all index to not used
        for(int idxLeafPtr = 0 ; idxLeafPtr < inNumberOfLeaves ; ++idxLeafPtr){
            leafHeader[idxLeafPtr].mindex = -1;
        }
    }

    /** Call the destructor of leaves and dealloc block memory */
    ~FGroupOfParticles(){
        if(deleteBuffer){
            FAlignedMemory::DeallocBytes(memoryBuffer);
            FAlignedMemory::DeallocBytes(attributesBuffer);
        }
    }

    /** Give access to the buffer to send the data */
    const unsigned char* getRawBuffer() const{
        return memoryBuffer;
    }

    /** The the size of the allocated buffer */
    size_t getBufferSizeInByte() const {
        return allocatedMemoryInByte;
    }

    /** Give access to the buffer to send the data */
    const unsigned char* getRawAttributesBuffer() const{
        return reinterpret_cast<const unsigned char*>(attributesBuffer);
    }

    /** Give access to the buffer to send the data */
    unsigned char* getRawAttributesBuffer(){
        return reinterpret_cast<unsigned char*>(attributesBuffer);
    }

    /** The the size of the allocated buffer */
    size_t getAttributesBufferSizeInByte() const {
        return blockHeader->attributeLeadingDim*NbAttributesPerParticle;
    }

    /** To know if the object will delete the memory block */
    bool getDeleteMemory() const{
        return deleteBuffer;
    }

    /** The index of the fist leaf (set from the constructor) */
    MortonIndex getStartingIndex() const {
        return blockHeader->startingIndex;
    }

    /** The index of the last leaf + 1 (set from the constructor) */
    MortonIndex getEndingIndex() const {
        return blockHeader->endingIndex;
    }

    /** The number of leaf (set from the constructor) */
    int getNumberOfLeavesInBlock() const {
        return blockHeader->numberOfLeavesInBlock;
    }

    /** Get the total number of particles in the group */
    FSize getNbParticlesInGroup() const {
        return blockHeader->nbParticlesInGroup;
    }

    /** The size of the interval endingIndex-startingIndex (set from the constructor) */
    MortonIndex getSizeOfInterval() const {
        return (blockHeader->endingIndex-blockHeader->startingIndex);
    }

    /** Return true if inIndex should be located in the current block */
    bool isInside(const MortonIndex inIndex) const{
        return blockHeader->startingIndex <= inIndex && inIndex < blockHeader->endingIndex;
    }

    /** Return the idx in array of the cell */
    MortonIndex getLeafMortonIndex(const int id) const{
        FAssertLF(id < blockHeader->numberOfLeavesInBlock);
        return leafHeader[id].mindex;
    }

    /** Check if a cell exist (by binary search) and return it index */
    int getLeafIndex(const MortonIndex leafIdx) const{
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
    bool exists(const MortonIndex inIndex) const {
        return isInside(inIndex) && (getLeafIndex(inIndex) != -1);
    }

    /** Allocate a new leaf by calling its constructor */
    size_t newLeaf(const MortonIndex inIndex, const int id, const FSize nbParticles, const size_t offsetInGroup){
        FAssertLF(isInside(inIndex));
        FAssertLF(!exists(inIndex));
        FAssertLF(id < blockHeader->numberOfLeavesInBlock);
        FAssertLF(offsetInGroup < size_t(blockHeader->nbParticlesAllocatedInGroup));
        leafHeader[id].nbParticles = nbParticles;
        leafHeader[id].offSet = offsetInGroup;
        leafHeader[id].mindex = inIndex;

        const size_t nextLeafOffsetInGroup = RoundToUpperParticles(offsetInGroup+nbParticles);
        FAssertLF(nextLeafOffsetInGroup <= size_t(blockHeader->nbParticlesAllocatedInGroup + MemoryAlignementParticles));
        return nextLeafOffsetInGroup;
    }

    /** Iterate on each allocated leaves */
    template<class ParticlesAttachedClass>
    void forEachLeaf(std::function<void(ParticlesAttachedClass*)> function){
        for(int idxLeafPtr = 0 ; idxLeafPtr < blockHeader->numberOfLeavesInBlock ; ++idxLeafPtr){
            ParticlesAttachedClass leaf(leafHeader[idxLeafPtr].nbParticles,
                                            particlePosition[0] + leafHeader[idxLeafPtr].offSet,
                                            blockHeader->positionsLeadingDim,
#ifndef SCALFMM_SIMGRID_NODATA
                                            (attributesBuffer?particleAttributes[NbSymbAttributes] + leafHeader[idxLeafPtr].offSet:nullptr),
#else
                                            nullptr,
#endif
                                            blockHeader->attributeLeadingDim);
            function(&leaf);
        }
    }


    /** Return the address of the leaf if it exists (or NULL) */
    template<class ParticlesAttachedClass>
    ParticlesAttachedClass getLeaf(const int id){
        FAssertLF(id < blockHeader->numberOfLeavesInBlock);
        return ParticlesAttachedClass(leafHeader[id].nbParticles,
                                          particlePosition[0] + leafHeader[id].offSet,
                                            blockHeader->positionsLeadingDim,
#ifndef SCALFMM_SIMGRID_NODATA
                                            (attributesBuffer?particleAttributes[NbSymbAttributes] + leafHeader[id].offSet:nullptr),
#else
                                            nullptr,
#endif
                                            blockHeader->attributeLeadingDim);
    }


    /** Extract methods */

    size_t getExtractBufferSize(const std::vector<int>& inLeavesIdxToExtract) const {
        size_t totalSize = 0;
        for(size_t idxEx = 0 ; idxEx < inLeavesIdxToExtract.size() ; ++idxEx){
            const int idLeaf = inLeavesIdxToExtract[idxEx];
            totalSize += leafHeader[idLeaf].nbParticles*sizeof(FReal)*NbSymbAttributes;
        }
        return totalSize;
    }

    void extractData(const std::vector<int>& inLeavesIdxToExtract,
                     unsigned char* outputBuffer, const size_t outputBufferSize) const {
        size_t idxValue = 0;
        for(size_t idxEx = 0 ; idxEx < inLeavesIdxToExtract.size() ; ++idxEx){
            const int idLeaf = inLeavesIdxToExtract[idxEx];
            for(unsigned idxAttr = 0 ; idxAttr < NbSymbAttributes ; ++idxAttr){
                const FReal* ptrData = particlePosition[0] + leafHeader[idLeaf].offSet
                        + blockHeader->positionsLeadingDim*idxAttr;
                for(int idxPart = 0 ; idxPart < leafHeader[idLeaf].nbParticles ; ++idxPart){
                    ((FReal*)outputBuffer)[idxValue++] = ptrData[idxPart];
                }
            }
        }
        FAssertLF(idxValue*sizeof(FReal) == outputBufferSize);
    }

    void restoreData(const std::vector<int>& inLeavesIdxToExtract,
                     const unsigned char* intputBuffer, const size_t inputBufferSize){
        size_t idxValue = 0;
        for(size_t idxEx = 0 ; idxEx < inLeavesIdxToExtract.size() ; ++idxEx){
            const int idLeaf = inLeavesIdxToExtract[idxEx];
            for(unsigned idxAttr = 0 ; idxAttr < NbSymbAttributes ; ++idxAttr){
                FReal* ptrData = particlePosition[0] + leafHeader[idLeaf].offSet
                        + blockHeader->positionsLeadingDim*idxAttr;
                for(int idxPart = 0 ; idxPart < leafHeader[idLeaf].nbParticles ; ++idxPart){
                    ptrData[idxPart] = ((const FReal*)intputBuffer)[idxValue++];
                }
            }
        }
        FAssertLF(idxValue*sizeof(FReal) == inputBufferSize);
    }
};



#endif // FGROUPOFPARTICLES_HPP
