#ifndef FCUDAGROUPOFCELLS_HPP
#define FCUDAGROUPOFCELLS_HPP

#include "FCudaGlobal.hpp"
#include "FCudaCompositeCell.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

/**
* @brief The FCudaGroupOfCells class manages the cells in block allocation.
*/
template <class SymboleCellClass, class PoleCellClass, class LocalCellClass>
class FCudaGroupOfCells {
    /** One header is allocated at the beginning of each block */
    struct alignas(FStarPUDefaultAlign::StructAlign) BlockHeader{
        MortonIndex startingIndex;
        MortonIndex endingIndex;
        int numberOfCellsInBlock;
    };

protected:
    //< The size of the memoryBuffer
    size_t allocatedMemoryInByte;
    //< Pointer to a block memory
    unsigned char* memoryBuffer;

    //< Pointer to the header inside the block memory
    BlockHeader*    blockHeader;
    //< Pointer to the indexes table inside the block memory
    MortonIndex*    cellIndexes;
    //< Pointer to the cells inside the block memory
    SymboleCellClass*      blockCells;

    //< The multipole data
    PoleCellClass* cellMultipoles;
    //< The local data
    LocalCellClass* cellLocals;

public:
    typedef FCudaCompositeCell<SymboleCellClass, PoleCellClass, LocalCellClass> CompleteCellClass;

    __device__ FCudaGroupOfCells()
        : allocatedMemoryInByte(0), memoryBuffer(nullptr),
          blockHeader(nullptr), cellIndexes(nullptr), blockCells(nullptr),
          cellMultipoles(nullptr), cellLocals(nullptr){
    }

    __device__ void reset(unsigned char* inBuffer, const size_t inAllocatedMemoryInByte,
                          unsigned char* inCellMultipoles, unsigned char* inCellLocals){
        // Move the pointers to the correct position
        allocatedMemoryInByte = (inAllocatedMemoryInByte);
        memoryBuffer = (inBuffer);
        blockHeader         = reinterpret_cast<BlockHeader*>(inBuffer);
        inBuffer += sizeof(BlockHeader);
        cellIndexes   = reinterpret_cast<MortonIndex*>(inBuffer);
        inBuffer += (blockHeader->numberOfCellsInBlock*sizeof(MortonIndex));
        blockCells          = reinterpret_cast<SymboleCellClass*>(inBuffer);
        inBuffer += (sizeof(SymboleCellClass)*blockHeader->numberOfCellsInBlock);
        //FAssertLF(size_t(inBuffer-memoryBuffer) == allocatedMemoryInByte);

        cellMultipoles = (PoleCellClass*)inCellMultipoles;
        cellLocals     = (LocalCellClass*)inCellLocals;
    }

    /**
     * Init from a given buffer
     * @param inBuffer
     * @param inAllocatedMemoryInByte
     */
    __device__ FCudaGroupOfCells(unsigned char* inBuffer, const size_t inAllocatedMemoryInByte,
                                 unsigned char* inCellMultipoles, unsigned char* inCellLocals)
        : allocatedMemoryInByte(inAllocatedMemoryInByte), memoryBuffer(inBuffer),
          blockHeader(nullptr), cellIndexes(nullptr), blockCells(nullptr),
          cellMultipoles(nullptr), cellLocals(nullptr){
        // Move the pointers to the correct position
        blockHeader         = reinterpret_cast<BlockHeader*>(inBuffer);
        inBuffer += sizeof(BlockHeader);
        cellIndexes   = reinterpret_cast<MortonIndex*>(inBuffer);
        inBuffer += (blockHeader->numberOfCellsInBlock*sizeof(MortonIndex));
        blockCells          = reinterpret_cast<SymboleCellClass*>(inBuffer);
        inBuffer += (sizeof(SymboleCellClass)*blockHeader->numberOfCellsInBlock);
        //FAssertLF(size_t(inBuffer-memoryBuffer) == allocatedMemoryInByte);

        cellMultipoles = (PoleCellClass*)inCellMultipoles;
        cellLocals     = (LocalCellClass*)inCellLocals;
    }

    /** The index of the fist cell (set from the constructor) */
    __device__ MortonIndex getStartingIndex() const {
        return blockHeader->startingIndex;
    }

    /** The index of the last cell + 1 (set from the constructor) */
    __device__ MortonIndex getEndingIndex() const {
        return blockHeader->endingIndex;
    }

    /** The number of cell (set from the constructor) */
    __device__ int getNumberOfCellsInBlock() const {
        return blockHeader->numberOfCellsInBlock;
    }

    /** The size of the interval endingIndex-startingIndex (set from the constructor) */
    MortonIndex getSizeOfInterval() const {
        return (blockHeader->endingIndex-blockHeader->startingIndex);
    }

    /** Return true if inIndex should be located in the current block */
    __device__ bool isInside(const MortonIndex inIndex) const{
        return blockHeader->startingIndex <= inIndex && inIndex < blockHeader->endingIndex;
    }

    /** Return the idx in array of the cell */
    __device__ MortonIndex getCellMortonIndex(const int cellPos) const{
        return cellIndexes[cellPos];
    }

    /** Check if a cell exist (by binary search) and return it index */
    __device__ int getFistChildIdx(const MortonIndex parentIdx) const{
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

    /** Check if a cell exist (by binary search) and return it index */
    __device__ int getCellIndex(const MortonIndex cellIdx) const{
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

    /** Return true if inIndex is located in the current block and is not empty */
    __device__ bool exists(const MortonIndex inIndex) const {
        return isInside(inIndex) && (getCellIndex(inIndex) != -1);
    }

    /** Return the address of the cell if it exists (or NULL) */
    __device__ CompleteCellClass getCompleteCell(const int cellPos){
        //FAssertLF(cellMultipoles && cellLocals);
        CompleteCellClass cell;
        cell.symb = &blockCells[cellPos];
        cell.up   = &cellMultipoles[cellPos];
        cell.down = &cellLocals[cellPos];
        return cell;
    }

    /** Return the address of the cell if it exists (or NULL) */
    __device__ CompleteCellClass getUpCell(const int cellPos){
        //FAssertLF(cellMultipoles);
        CompleteCellClass cell;
        cell.symb = &blockCells[cellPos];
        cell.up   = &cellMultipoles[cellPos];
        cell.down = nullptr;
        return cell;
    }

    /** Return the address of the cell if it exists (or NULL) */
    __device__ CompleteCellClass getDownCell(const int cellPos){
        //FAssertLF(cellLocals);
        CompleteCellClass cell;
        cell.symb = &blockCells[cellPos];
        cell.up   = nullptr;
        cell.down = &cellLocals[cellPos];
        return cell;
    }

};

#endif // FCUDAGROUPOFCELLS_HPP

