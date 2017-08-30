
#ifndef FGROUPOFCELLSDYN_HPP
#define FGROUPOFCELLSDYN_HPP


#include "../../Utils/FAssert.hpp"
#include "../../Utils/FAlignedMemory.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

#include <list>
#include <functional>

/**
* @brief The FGroupOfCellsDyn class manages the cells in block allocation.
*/
template <class CompositeCellClass>
class FGroupOfCellsDyn {
    /** One header is allocated at the beginning of each block */
    struct alignas(FStarPUDefaultAlign::StructAlign) BlockHeader{
        MortonIndex startingIndex;
        MortonIndex endingIndex;
        int numberOfCellsInBlock;
    };

    struct alignas(FStarPUDefaultAlign::StructAlign) CellClassSizes{
        size_t symbCellClassSize;
        size_t poleCellClassSize;
        size_t localCellClassSize;
    };

protected:
    //< The size of the memoryBuffer
    size_t allocatedMemoryInByte;
    //< Pointer to a block memory
    unsigned char* memoryBuffer;

    //< Pointer to the header inside the block memory
    BlockHeader*    blockHeader;
    CellClassSizes* cellSizes;
    //< Pointer to the indexes table inside the block memory
    MortonIndex*    cellIndexes;
    //< Pointer to the cells inside the block memory
    unsigned char*  blockCells;

    //< The multipole data
    unsigned char* cellMultipoles;
    //< The local data size
    unsigned char* cellLocals;

    //< To kown if the object has to delete the memory
    bool deleteBuffer;

public:
    typedef CompositeCellClass CompleteCellClass;

    FGroupOfCellsDyn()
        : allocatedMemoryInByte(0), memoryBuffer(nullptr),
          blockHeader(nullptr), cellSizes(nullptr), cellIndexes(nullptr), blockCells(nullptr),
          cellMultipoles(nullptr), cellLocals(nullptr), deleteBuffer(false){
    }

    void reset(unsigned char* inBuffer, const size_t inAllocatedMemoryInByte,
               unsigned char* inCellMultipoles, unsigned char* inCellLocals){
        if(deleteBuffer){
            for(int idxCellPtr = 0 ; idxCellPtr < blockHeader->numberOfCellsInBlock ; ++idxCellPtr){
                CompositeCellClass cell(&blockCells[idxCellPtr*cellSizes->symbCellClassSize],
                                    &cellMultipoles[idxCellPtr*cellSizes->poleCellClassSize],
                                    &cellLocals[idxCellPtr*cellSizes->localCellClassSize]);
                cell.release();
            }
            FAlignedMemory::DeallocBytes(memoryBuffer);
            FAlignedMemory::DeallocBytes(cellMultipoles);
            FAlignedMemory::DeallocBytes(cellLocals);
        }
        // Move the pointers to the correct position
        allocatedMemoryInByte = (inAllocatedMemoryInByte);
        memoryBuffer = (inBuffer);
        blockHeader         = reinterpret_cast<BlockHeader*>(inBuffer);
        inBuffer += sizeof(BlockHeader);
        cellSizes           = reinterpret_cast<CellClassSizes*>(inBuffer);
        inBuffer += sizeof(CellClassSizes);
        cellIndexes   = reinterpret_cast<MortonIndex*>(inBuffer);
        inBuffer += (blockHeader->numberOfCellsInBlock*sizeof(MortonIndex));
        blockCells          = reinterpret_cast<unsigned char*>(inBuffer);
        inBuffer += (cellSizes->symbCellClassSize*blockHeader->numberOfCellsInBlock);
        FAssertLF(size_t(inBuffer -memoryBuffer) == allocatedMemoryInByte);

        cellMultipoles = inCellMultipoles;
        cellLocals     = inCellLocals;
        deleteBuffer = (false);
    }

    /**
     * Init from a given buffer
     * @param inBuffer
     * @param inAllocatedMemoryInByte
     */
    FGroupOfCellsDyn(unsigned char* inBuffer, const size_t inAllocatedMemoryInByte,
                  unsigned char* inCellMultipoles, unsigned char* inCellLocals)
        : allocatedMemoryInByte(inAllocatedMemoryInByte), memoryBuffer(inBuffer),
          blockHeader(nullptr), cellSizes(nullptr), cellIndexes(nullptr), blockCells(nullptr),
          cellMultipoles(nullptr), cellLocals(nullptr), deleteBuffer(false){
        // Move the pointers to the correct position
        allocatedMemoryInByte = (inAllocatedMemoryInByte);
        memoryBuffer = (inBuffer);
        blockHeader         = reinterpret_cast<BlockHeader*>(inBuffer);
        inBuffer += sizeof(BlockHeader);
        cellSizes           = reinterpret_cast<CellClassSizes*>(inBuffer);
        inBuffer += sizeof(CellClassSizes);
        cellIndexes   = reinterpret_cast<MortonIndex*>(inBuffer);
        inBuffer += (blockHeader->numberOfCellsInBlock*sizeof(MortonIndex));
        blockCells          = reinterpret_cast<unsigned char*>(inBuffer);
        inBuffer += (cellSizes->symbCellClassSize*blockHeader->numberOfCellsInBlock);
        FAssertLF(size_t(inBuffer -memoryBuffer) == allocatedMemoryInByte);

        cellMultipoles = inCellMultipoles;
        cellLocals     = inCellLocals;
    }

    /**
 * @brief FGroupOfCellsDyn
 * @param inStartingIndex first cell morton index
 * @param inEndingIndex last cell morton index + 1
 * @param inNumberOfCells total number of cells in the interval (should be <= inEndingIndex-inEndingIndex)
 */
    FGroupOfCellsDyn(const MortonIndex inStartingIndex, const MortonIndex inEndingIndex, const int inNumberOfCells,
                     const size_t inSymbCellClassSize, const size_t inPoleCellClassSize, const size_t inLocalCellClassSize)
        : allocatedMemoryInByte(0), memoryBuffer(nullptr), blockHeader(nullptr), cellSizes(nullptr),
          cellIndexes(nullptr), blockCells(nullptr),
          cellMultipoles(nullptr), cellLocals(nullptr), deleteBuffer(true){
        FAssertLF((inEndingIndex-inStartingIndex) >= MortonIndex(inNumberOfCells));
        // Total number of bytes in the block
        const size_t memoryToAlloc = sizeof(BlockHeader) + sizeof(CellClassSizes)
                + (inNumberOfCells*sizeof(MortonIndex))
                + (inNumberOfCells*inSymbCellClassSize);

        // Allocate
        FAssertLF(0 <= int(memoryToAlloc) && int(memoryToAlloc) < std::numeric_limits<int>::max());
        allocatedMemoryInByte = memoryToAlloc;
        memoryBuffer = (unsigned char*)FAlignedMemory::AllocateBytes<32>(memoryToAlloc);
        FAssertLF(memoryBuffer);
        memset(memoryBuffer, 0, memoryToAlloc);

        // Move the pointers to the correct position
        unsigned char* bufferPtr = memoryBuffer;
        blockHeader         = reinterpret_cast<BlockHeader*>(bufferPtr);
        bufferPtr += sizeof(BlockHeader);
        cellSizes           = reinterpret_cast<CellClassSizes*>(bufferPtr);
        bufferPtr += sizeof(CellClassSizes);
        cellIndexes   = reinterpret_cast<MortonIndex*>(bufferPtr);
        bufferPtr += (inNumberOfCells*sizeof(MortonIndex));
        blockCells          = reinterpret_cast<unsigned char*>(bufferPtr);
        bufferPtr += (inNumberOfCells*inSymbCellClassSize);
        FAssertLF(size_t(bufferPtr - memoryBuffer) == allocatedMemoryInByte);

        // Init header
        blockHeader->startingIndex = inStartingIndex;
        blockHeader->endingIndex   = inEndingIndex;
        blockHeader->numberOfCellsInBlock  = inNumberOfCells;

        cellSizes->symbCellClassSize = inSymbCellClassSize;
        cellSizes->poleCellClassSize = inPoleCellClassSize;
        cellSizes->localCellClassSize = inLocalCellClassSize;

        cellMultipoles    = (unsigned char*)FAlignedMemory::AllocateBytes<32>(inNumberOfCells*cellSizes->poleCellClassSize);
        memset(cellMultipoles, 0, inNumberOfCells*cellSizes->poleCellClassSize);

        cellLocals     = (unsigned char*)FAlignedMemory::AllocateBytes<32>(inNumberOfCells*cellSizes->localCellClassSize);
        memset(cellLocals, 0, inNumberOfCells*cellSizes->localCellClassSize);

        // Set all index to not used
        for(int idxCellPtr = 0 ; idxCellPtr < inNumberOfCells ; ++idxCellPtr){
            cellIndexes[idxCellPtr] = -1;
        }
    }

    /** Call the destructor of cells and dealloc block memory */
    ~FGroupOfCellsDyn(){
        if(deleteBuffer){
            for(int idxCellPtr = 0 ; idxCellPtr < blockHeader->numberOfCellsInBlock ; ++idxCellPtr){
                CompositeCellClass cell(&blockCells[idxCellPtr*cellSizes->symbCellClassSize],
                                        &cellMultipoles[idxCellPtr*cellSizes->poleCellClassSize],
                                        &cellLocals[idxCellPtr*cellSizes->localCellClassSize]);
                cell.release();
            }
            FAlignedMemory::DeallocBytes(memoryBuffer);
            FAlignedMemory::DeallocBytes(cellMultipoles);
            FAlignedMemory::DeallocBytes(cellLocals);
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
    const unsigned char* getRawMultipoleBuffer() const{
        return cellMultipoles;
    }

    /** Give access to the buffer to send the data */
    unsigned char* getRawMultipoleBuffer() {
        return cellMultipoles;
    }

    /** The the size of the allocated buffer */
    size_t getMultipoleBufferSizeInByte() const {
        return cellSizes->poleCellClassSize*blockHeader->numberOfCellsInBlock;
    }

    /** Give access to the buffer to send the data */
    unsigned char* getRawLocalBuffer(){
        return cellLocals;
    }

    /** Give access to the buffer to send the data */
    const unsigned char* getRawLocalBuffer() const{
        return cellLocals;
    }

    /** The the size of the allocated buffer */
    size_t getLocalBufferSizeInByte() const {
        return cellSizes->localCellClassSize*blockHeader->numberOfCellsInBlock;
    }

    /** To know if the object will delete the memory block */
    bool getDeleteMemory() const{
        return deleteBuffer;
    }

    /** The index of the fist cell (set from the constructor) */
    MortonIndex getStartingIndex() const {
        return blockHeader->startingIndex;
    }

    /** The index of the last cell + 1 (set from the constructor) */
    MortonIndex getEndingIndex() const {
        return blockHeader->endingIndex;
    }

    /** The number of cell (set from the constructor) */
    int getNumberOfCellsInBlock() const {
        return blockHeader->numberOfCellsInBlock;
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
    MortonIndex getCellMortonIndex(const int cellPos) const{
        FAssertLF(cellPos < blockHeader->numberOfCellsInBlock);
        return cellIndexes[cellPos];
    }

    /** Check if a cell exist (by binary search) and return it index */
    int getCellIndex(const MortonIndex cellIdx) const{
        int idxLeft = 0;
        int idxRight = blockHeader->numberOfCellsInBlock-1;
        while(idxLeft <= idxRight){
            const int idxMiddle = (idxLeft+idxRight)/2;
            if(cellIndexes[idxMiddle] == cellIdx){
                return idxMiddle;
            }
            if(cellIdx < cellIndexes[idxMiddle]){
                idxRight = idxMiddle-1;
            }
            else{
                idxLeft = idxMiddle+1;
            }
        }
        return -1;
    }

    /** Check if a cell exist (by binary search) and return it index */
    int getFistChildIdx(const MortonIndex parentIdx) const{
        int idxLeft = 0;
        int idxRight = blockHeader->numberOfCellsInBlock-1;
        while(idxLeft <= idxRight){
            int idxMiddle = (idxLeft+idxRight)/2;
            if((cellIndexes[idxMiddle]>>3) == parentIdx){
                while(0 < idxMiddle && (cellIndexes[idxMiddle-1]>>3) == parentIdx){
                    idxMiddle -= 1;
                }
                return idxMiddle;
            }
            if(parentIdx < (cellIndexes[idxMiddle]>>3)){
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
        return isInside(inIndex) && (getCellIndex(inIndex) != -1);
    }

    /** Return the address of the cell if it exists (or NULL) */
    CompositeCellClass getCompleteCell(const MortonIndex cellPos){
        FAssertLF(cellMultipoles && cellLocals);
        FAssertLF(cellPos < blockHeader->numberOfCellsInBlock);
        return CompositeCellClass(&blockCells[cellPos*cellSizes->symbCellClassSize],
                    &cellMultipoles[cellPos*cellSizes->poleCellClassSize],
                    &cellLocals[cellPos*cellSizes->localCellClassSize]);
    }

    /** Return the address of the cell if it exists (or NULL) */
    CompositeCellClass getUpCell(const MortonIndex cellPos){
        FAssertLF(cellMultipoles);
        FAssertLF(cellPos < blockHeader->numberOfCellsInBlock);
        return CompositeCellClass(&blockCells[cellPos*cellSizes->symbCellClassSize],
                    &cellMultipoles[cellPos*cellSizes->poleCellClassSize], nullptr);
    }

    /** Return the address of the cell if it exists (or NULL) */
    CompositeCellClass getDownCell(const MortonIndex cellPos){
        FAssertLF(cellLocals);
        FAssertLF(cellPos < blockHeader->numberOfCellsInBlock);
        return CompositeCellClass(&blockCells[cellPos*cellSizes->symbCellClassSize],
                    nullptr, &cellLocals[cellPos*cellSizes->localCellClassSize]);
    }

    /** Allocate a new cell by calling its constructor */
    void newCell(const MortonIndex inIndex, const int id,
                 std::function<void(const MortonIndex mindex,
                                    unsigned char* symbBuff, const size_t symbSize,
                                    unsigned char* upBuff, const size_t upSize,
                                    unsigned char* downBuff, const size_t downSize,
                                    const int level)> BuildCellFunc,
                                    const int inLevel){
        FAssertLF(isInside(inIndex));
        FAssertLF(!exists(inIndex));
        FAssertLF(id < blockHeader->numberOfCellsInBlock);
        BuildCellFunc(inIndex,
                      &blockCells[id*cellSizes->symbCellClassSize],cellSizes->symbCellClassSize,
                      &cellMultipoles[id*cellSizes->poleCellClassSize],cellSizes->poleCellClassSize,
                      &cellLocals[id*cellSizes->localCellClassSize],cellSizes->localCellClassSize,
                      inLevel);
        cellIndexes[id] = inIndex;
    }

    /** Iterate on each allocated cells */
    template<typename... FunctionParams>
    void forEachCell(std::function<void(CompositeCellClass, FunctionParams...)> function, FunctionParams... args){
        for(int idxCellPtr = 0 ; idxCellPtr < blockHeader->numberOfCellsInBlock ; ++idxCellPtr){
            function(CompositeCellClass(&blockCells[idxCellPtr*cellSizes->symbCellClassSize],
                         &cellMultipoles[idxCellPtr*cellSizes->poleCellClassSize],
                        &cellLocals[idxCellPtr*cellSizes->localCellClassSize]), args...);
        }
    }

    void forEachCell(std::function<void(CompositeCellClass)> function){
        for(int idxCellPtr = 0 ; idxCellPtr < blockHeader->numberOfCellsInBlock ; ++idxCellPtr){
            function(CompositeCellClass(&blockCells[idxCellPtr*cellSizes->symbCellClassSize],
                         &cellMultipoles[idxCellPtr*cellSizes->poleCellClassSize],
                        &cellLocals[idxCellPtr*cellSizes->localCellClassSize]));
        }
    }
};


#endif // FGROUPOFCELLSDYN_HPP

