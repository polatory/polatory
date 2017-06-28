
#ifndef FGROUPTREEDYN_HPP
#define FGROUPTREEDYN_HPP

#include <vector>
#include <functional>
#include <memory>
#include <functional>

#include "../../Utils/FAssert.hpp"
#include "../../Utils/FPoint.hpp"
#include "../../Utils/FQuickSort.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../../Containers/FCoordinateComputer.hpp"
#include "FGroupOfCellsDyn.hpp"
#include "FGroupOfParticlesDyn.hpp"
#include "FGroupAttachedLeafDyn.hpp"



template <class FReal, class CompositeCellClass, class GroupAttachedLeafClass>
class FGroupTreeDyn {
public:
    typedef GroupAttachedLeafClass BasicAttachedClass;
    typedef FGroupOfParticlesDyn ParticleGroupClass;
    typedef FGroupOfCellsDyn<CompositeCellClass> CellGroupClass;

protected:
    //< height of the tree (1 => only the root)
    const int treeHeight;
    //< max number of cells in a block
    const int nbElementsPerBlock;
    //< all the blocks of the tree
    std::vector<CellGroupClass*>* cellBlocksPerLevel;
    //< all the blocks of leaves
    std::vector<ParticleGroupClass*> particleBlocks;

    //< the space system center
    const FPoint<FReal> boxCenter;
    //< the space system corner (used to compute morton index)
    const FPoint<FReal> boxCorner;
    //< the space system width
    const FReal boxWidth;
    //< the width of a box at width level
    const FReal boxWidthAtLeafLevel;

public:
    typedef typename std::vector<CellGroupClass*>::iterator CellGroupIterator;
    typedef typename std::vector<CellGroupClass*>::const_iterator CellGroupConstIterator;
    typedef typename std::vector<ParticleGroupClass*>::iterator ParticleGroupIterator;
    typedef typename std::vector<ParticleGroupClass*>::const_iterator ParticleGroupConstIterator;

    /** This constructor create a blocked octree from a usual octree
   * The cell are allocated as in the usual octree (no copy constructor are called!)
   * Once allocated each cell receive its morton index and tree coordinate.
   * No blocks are allocated at level 0.
   */
    template<class OctreeClass>
    FGroupTreeDyn(const int inTreeHeight, const int inNbElementsPerBlock, OctreeClass*const inOctreeSrc,
                  const size_t inSymbSizePerLevel[], const size_t inPoleSizePerLevel[], const size_t inLocalSizePerLevel[],
                  std::function<void(const MortonIndex, const void*, size_t*, size_t*)> GetSizeFunc,
                  std::function<void(const MortonIndex mindex,
                                     unsigned char* symbBuff, const size_t symbSize,
                                     unsigned char* upBuff, const size_t upSize,
                                     unsigned char* downBuff, const size_t downSize,
                                     const int level)> BuildCellFunc)
        : treeHeight(inTreeHeight), nbElementsPerBlock(inNbElementsPerBlock), cellBlocksPerLevel(nullptr),
          boxCenter(inOctreeSrc->getBoxCenter()), boxCorner(inOctreeSrc->getBoxCenter(),-(inOctreeSrc->getBoxWidth()/2)),
          boxWidth(inOctreeSrc->getBoxWidth()), boxWidthAtLeafLevel(inOctreeSrc->getBoxWidth()/FReal(1<<(inTreeHeight-1))){
        cellBlocksPerLevel = new std::vector<CellGroupClass*>[treeHeight];

        // Iterate on the tree and build
        typename OctreeClass::Iterator octreeIterator(inOctreeSrc);
        octreeIterator.gotoBottomLeft();

        { // First leaf level, we create leaves and cells groups
            std::unique_ptr<size_t[]> symbSizePerLeaf(new size_t[nbElementsPerBlock]);
            std::unique_ptr<size_t[]> downSizePerDown(new size_t[nbElementsPerBlock]);

            const int idxLevel = treeHeight-1;
            typename OctreeClass::Iterator avoidGotoLeft = octreeIterator;
            // For each cell at this level
            do {
                typename OctreeClass::Iterator blockIteratorInOctree = octreeIterator;
                // Move the iterator per nbElementsPerBlock (or until it cannot move right)
                int sizeOfBlock = 1;
                FSize nbParticlesInGroup = octreeIterator.getCurrentLeaf()->getSrc()->getNbParticles();
                while(sizeOfBlock < nbElementsPerBlock && octreeIterator.moveRight()){
                    sizeOfBlock += 1;
                    nbParticlesInGroup += octreeIterator.getCurrentLeaf()->getSrc()->getNbParticles();
                }

                // Create a block with the apropriate parameters
                CellGroupClass*const newBlock = new CellGroupClass(blockIteratorInOctree.getCurrentGlobalIndex(),
                                                                   octreeIterator.getCurrentGlobalIndex()+1,
                                                                   sizeOfBlock, inSymbSizePerLevel[idxLevel],
                                                                   inPoleSizePerLevel[idxLevel], inLocalSizePerLevel[idxLevel]);
                {
                    typename OctreeClass::Iterator blockIteratorCellInOctree = blockIteratorInOctree;
                    // Initialize each cell of the block
                    int cellIdInBlock = 0;
                    while(cellIdInBlock != sizeOfBlock){
                        const MortonIndex newNodeIndex = blockIteratorCellInOctree.getCurrentCell()->getMortonIndex();
                        const FTreeCoordinate newNodeCoordinate = blockIteratorCellInOctree.getCurrentCell()->getCoordinate();
                        // Add cell
                        newBlock->newCell(newNodeIndex, cellIdInBlock, BuildCellFunc, idxLevel);

                        CompositeCellClass newNode = newBlock->getCompleteCell(cellIdInBlock);
                        newNode.setMortonIndex(newNodeIndex);
                        newNode.setCoordinate(newNodeCoordinate);

                        cellIdInBlock += 1;
                        blockIteratorCellInOctree.moveRight();
                    }

                    // Keep the block
                    cellBlocksPerLevel[idxLevel].push_back(newBlock);
                }

                {
                    typename OctreeClass::Iterator blockIteratorLeafInOctree = blockIteratorInOctree;
                    int cellIdInBlock = 0;
                    while(cellIdInBlock != sizeOfBlock){
                        GetSizeFunc(blockIteratorLeafInOctree.getCurrentCell()->getMortonIndex(),
                                    blockIteratorLeafInOctree.getCurrentLeaf()->getSrc(),
                                    &symbSizePerLeaf[cellIdInBlock],&downSizePerDown[cellIdInBlock]);

                        cellIdInBlock += 1;
                        blockIteratorLeafInOctree.moveRight();
                    }
                }

                ParticleGroupClass*const newParticleBlock = new ParticleGroupClass(blockIteratorInOctree.getCurrentGlobalIndex(),
                                                                                   octreeIterator.getCurrentGlobalIndex()+1,
                                                                                   sizeOfBlock, symbSizePerLeaf.get(), downSizePerDown.get());
                {
                    typename OctreeClass::Iterator blockIteratorLeafInOctree = blockIteratorInOctree;
                    // Initialize each cell of the block
                    int cellIdInBlock = 0;
                    while(cellIdInBlock != sizeOfBlock){
                        const MortonIndex newNodeIndex = blockIteratorLeafInOctree.getCurrentCell()->getMortonIndex();
                        // Add leaf
                        newParticleBlock->newLeaf(newNodeIndex, cellIdInBlock);

                        BasicAttachedClass attachedLeaf = newParticleBlock->template getLeaf<BasicAttachedClass>(cellIdInBlock);
                        attachedLeaf.copyFromContainer(newNodeIndex,
                                                       blockIteratorLeafInOctree.getCurrentLeaf()->getSrc());

                        cellIdInBlock += 1;
                        blockIteratorLeafInOctree.moveRight();
                    }
                    // Keep the block
                    particleBlocks.push_back(newParticleBlock);
                }

                // If we can move right then add another block
            } while(octreeIterator.moveRight());

            avoidGotoLeft.moveUp();
            octreeIterator = avoidGotoLeft;
        }

        // For each level from heigth - 2 to 1
        for(int idxLevel = treeHeight-2; idxLevel > 0 ; --idxLevel){
            typename OctreeClass::Iterator avoidGotoLeft = octreeIterator;
            // For each cell at this level
            do {
                typename OctreeClass::Iterator blockIteratorInOctree = octreeIterator;
                // Move the iterator per nbElementsPerBlock (or until it cannot move right)
                int sizeOfBlock = 1;
                while(sizeOfBlock < nbElementsPerBlock && octreeIterator.moveRight()){
                    sizeOfBlock += 1;
                }

                // Create a block with the apropriate parameters
                CellGroupClass*const newBlock = new CellGroupClass(blockIteratorInOctree.getCurrentGlobalIndex(),
                                                                   octreeIterator.getCurrentGlobalIndex()+1,
                                                                   sizeOfBlock, inSymbSizePerLevel[idxLevel],
                                                                   inPoleSizePerLevel[idxLevel], inLocalSizePerLevel[idxLevel]);

                // Initialize each cell of the block
                int cellIdInBlock = 0;
                while(cellIdInBlock != sizeOfBlock){
                    const MortonIndex newNodeIndex = blockIteratorInOctree.getCurrentCell()->getMortonIndex();
                    const FTreeCoordinate newNodeCoordinate = blockIteratorInOctree.getCurrentCell()->getCoordinate();
                    newBlock->newCell(newNodeIndex, cellIdInBlock, BuildCellFunc, idxLevel);

                    CompositeCellClass newNode = newBlock->getCompleteCell(cellIdInBlock);
                    newNode.setMortonIndex(newNodeIndex);
                    newNode.setCoordinate(newNodeCoordinate);

                    cellIdInBlock += 1;
                    blockIteratorInOctree.moveRight();
                }

                // Keep the block
                cellBlocksPerLevel[idxLevel].push_back(newBlock);

                // If we can move right then add another block
            } while(octreeIterator.moveRight());

            avoidGotoLeft.moveUp();
            octreeIterator = avoidGotoLeft;
        }
    }

    /**
     * This constructor create a group tree from a particle container index.
     * The morton index are computed and the particles are sorted in a first stage.
     * Then the leaf level is done.
     * Finally the other leve are proceed one after the other.
     * It should be easy to make it parallel using for and tasks.
     * If no limite give inLeftLimite = -1
     */
    FGroupTreeDyn(const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter,
                  const int inNbElementsPerBlock, const size_t inSymbSizePerLevel[],
                  const size_t inPoleSizePerLevel[], const size_t inLocalSizePerLevel[],
                  UnknownDescriptor<FReal> inParticlesContainer[], const FSize nbParticles,
                  std::function<void(const MortonIndex, const UnknownDescriptor<FReal>[],
                                     const FSize, size_t*, size_t*)> GetSizeFunc,
                  std::function<void(const MortonIndex, const UnknownDescriptor<FReal> [],
                                     const FSize ,
                                     unsigned char* , const size_t,
                                     unsigned char* , const size_t)> InitLeafFunc,
                  std::function<void(const MortonIndex mindex,
                                     unsigned char* symbBuff, const size_t symbSize,
                                     unsigned char* upBuff, const size_t upSize,
                                     unsigned char* downBuff, const size_t downSize,
                                     const int level)> BuildCellFunc,
                  const bool particlesAreSorted = false, MortonIndex inLeftLimite = -1):
        treeHeight(inTreeHeight),nbElementsPerBlock(inNbElementsPerBlock),cellBlocksPerLevel(nullptr),
        boxCenter(inBoxCenter), boxCorner(inBoxCenter,-(inBoxWidth/2)), boxWidth(inBoxWidth),
        boxWidthAtLeafLevel(inBoxWidth/FReal(1<<(inTreeHeight-1))){

        cellBlocksPerLevel = new std::vector<CellGroupClass*>[treeHeight];

        std::unique_ptr<MortonIndex[]> currentBlockIndexes(new MortonIndex[nbElementsPerBlock]);
        // First we work at leaf level
        {
            // Sort if needed
            if(particlesAreSorted == false){
                for(FSize idxPart = 0 ; idxPart < nbParticles ; ++idxPart){
                    const FTreeCoordinate host = FCoordinateComputer::GetCoordinateFromPositionAndCorner<FReal>(this->boxCorner, this->boxWidth,
                                                                                                                treeHeight,
                                                                                                                inParticlesContainer[idxPart].pos);
                    const MortonIndex particleIndex = host.getMortonIndex();
                    inParticlesContainer[idxPart].mindex = particleIndex;
                    inParticlesContainer[idxPart].originalIndex = idxPart;
                }

                FQuickSort<UnknownDescriptor<FReal>, FSize>::QsOmp(inParticlesContainer, nbParticles, [](const UnknownDescriptor<FReal>& v1, const UnknownDescriptor<FReal>& v2){
                    return v1.mindex <= v2.mindex;
                });
            }

            FAssertLF(nbParticles == 0 || inLeftLimite < inParticlesContainer[0].mindex);

            // Convert to block
            const int idxLevel = (treeHeight - 1);

            std::unique_ptr<size_t[]> symbSizePerLeaf(new size_t[nbElementsPerBlock]);
            std::unique_ptr<size_t[]> downSizePerDown(new size_t[nbElementsPerBlock]);
            std::unique_ptr<size_t[]> nbParticlesPerLeaf(new size_t [nbElementsPerBlock]);
            FSize firstParticle = 0;
            // We need to proceed each group in sub level
            while(firstParticle != nbParticles){
                int sizeOfBlock = 0;
                FSize lastParticle = firstParticle;
                // Count until end of sub group is reached or we have enough cells
                while(sizeOfBlock < nbElementsPerBlock && lastParticle < nbParticles){
                    if(sizeOfBlock == 0 || currentBlockIndexes[sizeOfBlock-1] != inParticlesContainer[lastParticle].mindex){
                        currentBlockIndexes[sizeOfBlock] = inParticlesContainer[lastParticle].mindex;
                        nbParticlesPerLeaf[sizeOfBlock]  = 1;
                        sizeOfBlock += 1;
                    }
                    else{
                        nbParticlesPerLeaf[sizeOfBlock-1] += 1;
                    }
                    lastParticle += 1;
                }
                while(lastParticle < nbParticles && currentBlockIndexes[sizeOfBlock-1] == inParticlesContainer[lastParticle].mindex){
                    nbParticlesPerLeaf[sizeOfBlock-1] += 1;
                    lastParticle += 1;
                }


                // Create a group
                CellGroupClass*const newBlock = new CellGroupClass(currentBlockIndexes[0],
                        currentBlockIndexes[sizeOfBlock-1]+1,
                        sizeOfBlock, inSymbSizePerLevel[idxLevel],
                        inPoleSizePerLevel[idxLevel], inLocalSizePerLevel[idxLevel]);
                {
                    for(int cellIdInBlock = 0; cellIdInBlock != sizeOfBlock ; ++cellIdInBlock){
                        newBlock->newCell(currentBlockIndexes[cellIdInBlock], cellIdInBlock, BuildCellFunc, idxLevel);

                        CompositeCellClass newNode = newBlock->getCompleteCell(cellIdInBlock);
                        newNode.setMortonIndex(currentBlockIndexes[cellIdInBlock]);
                        FTreeCoordinate coord;
                        coord.setPositionFromMorton(currentBlockIndexes[cellIdInBlock]);
                        newNode.setCoordinate(coord);
                    }
                }

                {
                    FSize offsetParts = firstParticle;
                    for(int cellIdInBlock = 0; cellIdInBlock != sizeOfBlock ; ++cellIdInBlock){
                        GetSizeFunc(currentBlockIndexes[cellIdInBlock], &inParticlesContainer[offsetParts],
                                    nbParticlesPerLeaf[cellIdInBlock],
                                    &symbSizePerLeaf[cellIdInBlock], &downSizePerDown[cellIdInBlock]);
                        offsetParts += nbParticlesPerLeaf[cellIdInBlock];
                    }
                }

                ParticleGroupClass*const newParticleBlock = new ParticleGroupClass(currentBlockIndexes[0],
                        currentBlockIndexes[sizeOfBlock-1]+1,
                        sizeOfBlock, symbSizePerLeaf.get(), downSizePerDown.get());

                // Init cells
                FSize offsetParts = firstParticle;
                for(int cellIdInBlock = 0; cellIdInBlock != sizeOfBlock ; ++cellIdInBlock){
                    // Add leaf
                    newParticleBlock->newLeaf(currentBlockIndexes[cellIdInBlock], cellIdInBlock);

                    InitLeafFunc(currentBlockIndexes[cellIdInBlock], &inParticlesContainer[offsetParts],
                                 nbParticlesPerLeaf[cellIdInBlock],
                                 newParticleBlock->getLeafSymbBuffer(cellIdInBlock), symbSizePerLeaf[cellIdInBlock],
                                 newParticleBlock->getLeafDownBuffer(cellIdInBlock), downSizePerDown[cellIdInBlock]);

                    offsetParts += nbParticlesPerLeaf[cellIdInBlock];
                }

                // Keep the block
                cellBlocksPerLevel[idxLevel].push_back(newBlock);
                particleBlocks.push_back(newParticleBlock);

                sizeOfBlock = 0;
                firstParticle = lastParticle;
            }
        }


        // For each level from heigth - 2 to 1
        for(int idxLevel = treeHeight-2; idxLevel > 0 ; --idxLevel){
            inLeftLimite = (inLeftLimite == -1 ? inLeftLimite : (inLeftLimite>>3));

            CellGroupConstIterator iterChildCells = cellBlocksPerLevel[idxLevel+1].begin();
            const CellGroupConstIterator iterChildEndCells = cellBlocksPerLevel[idxLevel+1].end();

            // Skip blocks that do not respect limit
            while(iterChildCells != iterChildEndCells
                  && ((*iterChildCells)->getEndingIndex()>>3) <= inLeftLimite){
                ++iterChildCells;
            }
            // If lower level is empty or all blocks skiped stop here
            if(iterChildCells == iterChildEndCells){
                break;
            }

            MortonIndex currentCellIndex = (*iterChildCells)->getStartingIndex();
            if((currentCellIndex>>3) <= inLeftLimite) currentCellIndex = ((inLeftLimite+1)<<3);
            int sizeOfBlock = 0;

            // We need to proceed each group in sub level
            while(iterChildCells != iterChildEndCells){
                // Count until end of sub group is reached or we have enough cells
                while(sizeOfBlock < nbElementsPerBlock && iterChildCells != iterChildEndCells ){
                    if((sizeOfBlock == 0 || currentBlockIndexes[sizeOfBlock-1] != (currentCellIndex>>3))
                            && (*iterChildCells)->exists(currentCellIndex)){
                        currentBlockIndexes[sizeOfBlock] = (currentCellIndex>>3);
                        sizeOfBlock += 1;
                        currentCellIndex = (((currentCellIndex>>3)+1)<<3);
                    }
                    else{
                        currentCellIndex += 1;
                    }
                    // If we are at the end of the sub group, move to next
                    while(iterChildCells != iterChildEndCells && (*iterChildCells)->getEndingIndex() <= currentCellIndex){
                        ++iterChildCells;
                        // Update morton index
                        if(iterChildCells != iterChildEndCells && currentCellIndex < (*iterChildCells)->getStartingIndex()){
                            currentCellIndex = (*iterChildCells)->getStartingIndex();
                        }
                    }
                }

                // If group is full
                if(sizeOfBlock == nbElementsPerBlock || (sizeOfBlock && iterChildCells == iterChildEndCells)){
                    // Create a group
                    CellGroupClass*const newBlock = new CellGroupClass(currentBlockIndexes[0],
                            currentBlockIndexes[sizeOfBlock-1]+1,
                            sizeOfBlock,inSymbSizePerLevel[idxLevel],
                            inPoleSizePerLevel[idxLevel], inLocalSizePerLevel[idxLevel]);
                    // Init cells
                    for(int cellIdInBlock = 0; cellIdInBlock != sizeOfBlock ; ++cellIdInBlock){
                        newBlock->newCell(currentBlockIndexes[cellIdInBlock], cellIdInBlock, BuildCellFunc, idxLevel);

                        CompositeCellClass newNode = newBlock->getCompleteCell(cellIdInBlock);
                        newNode.setMortonIndex(currentBlockIndexes[cellIdInBlock]);
                        FTreeCoordinate coord;
                        coord.setPositionFromMorton(currentBlockIndexes[cellIdInBlock]);
                        newNode.setCoordinate(coord);
                    }

                    // Keep the block
                    cellBlocksPerLevel[idxLevel].push_back(newBlock);

                    sizeOfBlock = 0;
                }
            }
        }
    }

    /**
     * This constructor create a group tree from a particle container index.
     * The morton index are computed and the particles are sorted in a first stage.
     * Then the leaf level is done.
     * Finally the other leve are proceed one after the other.
     * It should be easy to make it parallel using for and tasks.
     * If no limite give inLeftLimite = -1
     * The cover ration is the minimum pourcentage of cell that should
     * exist in a group (0 means no limite, 1 means the block must be dense)
     * oneParent should be turned on if it is better to have one block parent
     * per sublock (in case of have the cost of FMM that increase with the level
     * this could be an asset).
     */
    FGroupTreeDyn(const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter,
                  const int inNbElementsPerBlock, const size_t inSymbSizePerLevel[],
                  const size_t inPoleSizePerLevel[], const size_t inLocalSizePerLevel[],
                  UnknownDescriptor<FReal> inParticlesContainer[], const FSize nbParticles,
                  std::function<void(const MortonIndex, const UnknownDescriptor<FReal>[],
                                     const FSize, size_t*, size_t*)> GetSizeFunc,
                  std::function<void(const MortonIndex, const UnknownDescriptor<FReal> [],
                                     const FSize ,
                                     unsigned char* , const size_t,
                                     unsigned char* , const size_t)> InitLeafFunc,
                  std::function<void(const MortonIndex mindex,
                                     unsigned char* symbBuff, const size_t symbSize,
                                     unsigned char* upBuff, const size_t upSize,
                                     unsigned char* downBuff, const size_t downSize,
                                     const int level)> BuildCellFunc,
                  const bool particlesAreSorted, const bool oneParent,
                  const FReal inCoverRatio = 0.0, MortonIndex inLeftLimite = -1):
        treeHeight(inTreeHeight),nbElementsPerBlock(inNbElementsPerBlock),cellBlocksPerLevel(nullptr),
        boxCenter(inBoxCenter), boxCorner(inBoxCenter,-(inBoxWidth/2)), boxWidth(inBoxWidth),
        boxWidthAtLeafLevel(inBoxWidth/FReal(1<<(inTreeHeight-1))){

        FAssertLF(inCoverRatio == 0.0 || oneParent == true, "If a ratio is choosen oneParent should be turned on");
        const bool userCoverRatio = (inCoverRatio != 0.0);

        cellBlocksPerLevel = new std::vector<CellGroupClass*>[treeHeight];

        std::unique_ptr<MortonIndex[]> currentBlockIndexes(new MortonIndex[nbElementsPerBlock]);
        // First we work at leaf level
        {
            // Sort if needed
            if(particlesAreSorted == false){
                for(FSize idxPart = 0 ; idxPart < nbParticles ; ++idxPart){
                    const FTreeCoordinate host = FCoordinateComputer::GetCoordinateFromPositionAndCorner<FReal>(this->boxCorner, this->boxWidth,
                                                                                                                treeHeight,
                                                                                                                inParticlesContainer[idxPart].pos);
                    const MortonIndex particleIndex = host.getMortonIndex();
                    inParticlesContainer[idxPart].mindex = particleIndex;
                    inParticlesContainer[idxPart].originalIndex = idxPart;
                }

                FQuickSort<UnknownDescriptor<FReal>, FSize>::QsOmp(inParticlesContainer, nbParticles, [](const UnknownDescriptor<FReal>& v1, const UnknownDescriptor<FReal>& v2){
                    return v1.mindex <= v2.mindex;
                });
            }

            FAssertLF(nbParticles == 0 || inLeftLimite < inParticlesContainer[0].mindex);

            // Convert to block
            const int idxLevel = (treeHeight - 1);
            std::unique_ptr<size_t[]> symbSizePerLeaf(new size_t[nbElementsPerBlock]);
            std::unique_ptr<size_t[]> downSizePerDown(new size_t[nbElementsPerBlock]);
            std::unique_ptr<size_t[]> nbParticlesPerLeaf(new size_t [nbElementsPerBlock]);
            int firstParticle = 0;
            // We need to proceed each group in sub level
            while(firstParticle != nbParticles){
                int sizeOfBlock = 0;
                int lastParticle = firstParticle;
                // Count until end of sub group is reached or we have enough cells
                while(sizeOfBlock < nbElementsPerBlock && lastParticle < nbParticles
                      && (userCoverRatio == false
                          || sizeOfBlock == 0
                          || currentBlockIndexes[sizeOfBlock-1] == inParticlesContainer[lastParticle].mindex
                          || (FReal(sizeOfBlock+1)/FReal(inParticlesContainer[lastParticle].mindex-inParticlesContainer[firstParticle].mindex)) >= inCoverRatio)){
                    if(sizeOfBlock == 0 || currentBlockIndexes[sizeOfBlock-1] != inParticlesContainer[lastParticle].mindex){
                        currentBlockIndexes[sizeOfBlock] = inParticlesContainer[lastParticle].mindex;
                        nbParticlesPerLeaf[sizeOfBlock]  = 1;
                        sizeOfBlock += 1;
                    }
                    else{
                        nbParticlesPerLeaf[sizeOfBlock-1] += 1;
                    }
                    lastParticle += 1;
                }
                while(lastParticle < nbParticles && currentBlockIndexes[sizeOfBlock-1] == inParticlesContainer[lastParticle].mindex){
                    nbParticlesPerLeaf[sizeOfBlock-1] += 1;
                    lastParticle += 1;
                }

                // Create a group
                CellGroupClass*const newBlock = new CellGroupClass(currentBlockIndexes[0],
                        currentBlockIndexes[sizeOfBlock-1]+1,
                        sizeOfBlock, inSymbSizePerLevel[idxLevel],
                        inPoleSizePerLevel[idxLevel], inLocalSizePerLevel[idxLevel]);
                {
                    for(int cellIdInBlock = 0; cellIdInBlock != sizeOfBlock ; ++cellIdInBlock){
                        newBlock->newCell(currentBlockIndexes[cellIdInBlock], cellIdInBlock, BuildCellFunc, idxLevel);

                        CompositeCellClass newNode = newBlock->getCompleteCell(cellIdInBlock);
                        newNode.setMortonIndex(currentBlockIndexes[cellIdInBlock]);
                        FTreeCoordinate coord;
                        coord.setPositionFromMorton(currentBlockIndexes[cellIdInBlock]);
                        newNode.setCoordinate(coord);
                    }
                }

                {
                    FSize offsetParts = firstParticle;
                    for(int cellIdInBlock = 0; cellIdInBlock != sizeOfBlock ; ++cellIdInBlock){
                        GetSizeFunc(currentBlockIndexes[cellIdInBlock], &inParticlesContainer[offsetParts],
                                    nbParticlesPerLeaf[cellIdInBlock],
                                    &symbSizePerLeaf[cellIdInBlock], &downSizePerDown[cellIdInBlock]);
                        offsetParts += nbParticlesPerLeaf[cellIdInBlock];
                    }
                }

                ParticleGroupClass*const newParticleBlock = new ParticleGroupClass(currentBlockIndexes[0],
                        currentBlockIndexes[sizeOfBlock-1]+1,
                        sizeOfBlock, symbSizePerLeaf.get(), downSizePerDown.get());

                // Init cells
                FSize offsetParts = firstParticle;
                for(int cellIdInBlock = 0; cellIdInBlock != sizeOfBlock ; ++cellIdInBlock){
                    // Add leaf
                    newParticleBlock->newLeaf(currentBlockIndexes[cellIdInBlock], cellIdInBlock);

                    InitLeafFunc(currentBlockIndexes[cellIdInBlock], &inParticlesContainer[offsetParts],
                                 nbParticlesPerLeaf[cellIdInBlock],
                                 newParticleBlock->getLeafSymbBuffer(cellIdInBlock), symbSizePerLeaf[cellIdInBlock],
                                 newParticleBlock->getLeafDownBuffer(cellIdInBlock), downSizePerDown[cellIdInBlock]);

                    offsetParts += nbParticlesPerLeaf[cellIdInBlock];
                }

                // Keep the block
                cellBlocksPerLevel[idxLevel].push_back(newBlock);
                particleBlocks.push_back(newParticleBlock);

                sizeOfBlock = 0;
                firstParticle = lastParticle;
            }
        }


        // For each level from heigth - 2 to 1
        for(int idxLevel = treeHeight-2; idxLevel > 0 ; --idxLevel){
            inLeftLimite = (inLeftLimite == -1 ? inLeftLimite : (inLeftLimite>>3));

            CellGroupConstIterator iterChildCells = cellBlocksPerLevel[idxLevel+1].begin();
            const CellGroupConstIterator iterChildEndCells = cellBlocksPerLevel[idxLevel+1].end();

            // Skip blocks that do not respect limit
            while(iterChildCells != iterChildEndCells
                  && ((*iterChildCells)->getEndingIndex()>>3) <= inLeftLimite){
                ++iterChildCells;
            }
            // If lower level is empty or all blocks skiped stop here
            if(iterChildCells == iterChildEndCells){
                break;
            }

            MortonIndex currentCellIndex = (*iterChildCells)->getStartingIndex();
            if((currentCellIndex>>3) <= inLeftLimite) currentCellIndex = ((inLeftLimite+1)<<3);
            int sizeOfBlock = 0;

            if(oneParent == false){
                // We need to proceed each group in sub level
                while(iterChildCells != iterChildEndCells){
                    // Count until end of sub group is reached or we have enough cells
                    while(sizeOfBlock < nbElementsPerBlock && iterChildCells != iterChildEndCells ){
                        if((sizeOfBlock == 0 || currentBlockIndexes[sizeOfBlock-1] != (currentCellIndex>>3))
                                && (*iterChildCells)->exists(currentCellIndex)){
                            currentBlockIndexes[sizeOfBlock] = (currentCellIndex>>3);
                            sizeOfBlock += 1;
                            currentCellIndex = (((currentCellIndex>>3)+1)<<3);
                        }
                        else{
                            currentCellIndex += 1;
                        }
                        // If we are at the end of the sub group, move to next
                        while(iterChildCells != iterChildEndCells && (*iterChildCells)->getEndingIndex() <= currentCellIndex){
                            ++iterChildCells;
                            // Update morton index
                            if(iterChildCells != iterChildEndCells && currentCellIndex < (*iterChildCells)->getStartingIndex()){
                                currentCellIndex = (*iterChildCells)->getStartingIndex();
                            }
                        }
                    }

                    // If group is full
                    if(sizeOfBlock == nbElementsPerBlock || (sizeOfBlock && iterChildCells == iterChildEndCells)){
                        // Create a group
                        CellGroupClass*const newBlock = new CellGroupClass(currentBlockIndexes[0],
                                currentBlockIndexes[sizeOfBlock-1]+1,
                                sizeOfBlock,inSymbSizePerLevel[idxLevel],
                                inPoleSizePerLevel[idxLevel], inLocalSizePerLevel[idxLevel]);
                        // Init cells
                        for(int cellIdInBlock = 0; cellIdInBlock != sizeOfBlock ; ++cellIdInBlock){
                            newBlock->newCell(currentBlockIndexes[cellIdInBlock], cellIdInBlock, BuildCellFunc, idxLevel);

                            CompositeCellClass newNode = newBlock->getCompleteCell(cellIdInBlock);
                            newNode.setMortonIndex(currentBlockIndexes[cellIdInBlock]);
                            FTreeCoordinate coord;
                            coord.setPositionFromMorton(currentBlockIndexes[cellIdInBlock]);
                            newNode.setCoordinate(coord);
                        }

                        // Keep the block
                        cellBlocksPerLevel[idxLevel].push_back(newBlock);

                        sizeOfBlock = 0;
                    }
                }
            }
            else{
                // We need to proceed each group in sub level
                while(iterChildCells != iterChildEndCells){
                    // We want one parent group per child group so we will stop the parent group
                    // when we arrive to the same parent as lastChildIndex (which is lastChildIndex>>3)
                    const MortonIndex lastChildIndex = ((*iterChildCells)->getEndingIndex()-1);
                    // Count until end of sub group is reached or we passe the requested parent
                    while( iterChildCells != iterChildEndCells
                           && (currentCellIndex>>3) <= (lastChildIndex>>3) ){
                        // Proceed until the requested parent
                        while(currentCellIndex != (*iterChildCells)->getEndingIndex()
                              && (currentCellIndex>>3) <= (lastChildIndex>>3) ){
                            if((*iterChildCells)->exists(currentCellIndex)){
                                currentBlockIndexes[sizeOfBlock] = (currentCellIndex>>3);
                                sizeOfBlock += 1;
                                currentCellIndex = (((currentCellIndex>>3)+1)<<3);
                            }
                            else{
                                currentCellIndex += 1;
                            }
                        }
                        // If we are at the end of the sub group, move to next (otherwise we have consume a part of it)
                        while(iterChildCells != iterChildEndCells && (*iterChildCells)->getEndingIndex() <= currentCellIndex){
                            ++iterChildCells;
                            // Update morton index
                            if(iterChildCells != iterChildEndCells && currentCellIndex < (*iterChildCells)->getStartingIndex()){
                                currentCellIndex = (*iterChildCells)->getStartingIndex();
                            }
                        }
                    }

                    // If group is full
                    if(sizeOfBlock){
                        // Create a group
                        CellGroupClass*const newBlock = new CellGroupClass(currentBlockIndexes[0],
                                currentBlockIndexes[sizeOfBlock-1]+1,
                                sizeOfBlock,inSymbSizePerLevel[idxLevel],
                                inPoleSizePerLevel[idxLevel], inLocalSizePerLevel[idxLevel]);
                        // Init cells
                        for(int cellIdInBlock = 0; cellIdInBlock != sizeOfBlock ; ++cellIdInBlock){
                            newBlock->newCell(currentBlockIndexes[cellIdInBlock], cellIdInBlock, BuildCellFunc, idxLevel);

                            CompositeCellClass newNode = newBlock->getCompleteCell(cellIdInBlock);
                            newNode.setMortonIndex(currentBlockIndexes[cellIdInBlock]);
                            FTreeCoordinate coord;
                            coord.setPositionFromMorton(currentBlockIndexes[cellIdInBlock]);
                            newNode.setCoordinate(coord);
                        }

                        // Keep the block
                        cellBlocksPerLevel[idxLevel].push_back(newBlock);

                        sizeOfBlock = 0;
                    }
                }
            }
        }
    }


    /** This function dealloc the tree by deleting each block */
    ~FGroupTreeDyn(){
        for(int idxLevel = 0 ; idxLevel < treeHeight ; ++idxLevel){
            std::vector<CellGroupClass*>& levelBlocks = cellBlocksPerLevel[idxLevel];
            for (CellGroupClass* block: levelBlocks){
                delete block;
            }
        }
        delete[] cellBlocksPerLevel;

        for (ParticleGroupClass* block: particleBlocks){
            delete block;
        }
    }


    /////////////////////////////////////////////////////////
    // Lambda function to apply to all member
    /////////////////////////////////////////////////////////

    /**
   * @brief forEachLeaf iterate on the leaf and apply the function
   * @param function
   */
    template<class ParticlesAttachedClass>
    void forEachLeaf(std::function<void(ParticlesAttachedClass*)> function){
        for (ParticleGroupClass* block: particleBlocks){
            block->forEachLeaf(function);
        }
    }

    /**
   * @brief forEachLeaf iterate on the cell and apply the function
   * @param function
   */
    void forEachCell(std::function<void(CompositeCellClass)> function){
        for(int idxLevel = 0 ; idxLevel < treeHeight ; ++idxLevel){
            std::vector<CellGroupClass*>& levelBlocks = cellBlocksPerLevel[idxLevel];
            for (CellGroupClass* block: levelBlocks){
                block->forEachCell(function);
            }
        }
    }

    /**
   * @brief forEachLeaf iterate on the cell and apply the function
   * @param function
   */
    void forEachCellWithLevel(std::function<void(CompositeCellClass,const int)> function){
        for(int idxLevel = 0 ; idxLevel < treeHeight ; ++idxLevel){
            std::vector<CellGroupClass*>& levelBlocks = cellBlocksPerLevel[idxLevel];
            for (CellGroupClass* block: levelBlocks){
                block->forEachCell(function, idxLevel);
            }
        }
    }

    /**
   * @brief forEachLeaf iterate on the cell and apply the function
   * @param function
   */
    template<class ParticlesAttachedClass>
    void forEachCellLeaf(std::function<void(CompositeCellClass,ParticlesAttachedClass*)> function){
        CellGroupIterator iterCells = cellBlocksPerLevel[treeHeight-1].begin();
        const CellGroupIterator iterEndCells = cellBlocksPerLevel[treeHeight-1].end();

        ParticleGroupIterator iterLeaves = particleBlocks.begin();
        const ParticleGroupIterator iterEndLeaves = particleBlocks.end();

        while(iterCells != iterEndCells && iterLeaves != iterEndLeaves){
            (*iterCells)->forEachCell([&](CompositeCellClass aCell){
                const int leafPos = (*iterLeaves)->getLeafIndex(aCell.getMortonIndex());
                FAssertLF(leafPos != -1);
                ParticlesAttachedClass aLeaf = (*iterLeaves)->template getLeaf <ParticlesAttachedClass>(leafPos);
                FAssertLF(aLeaf.isAttachedToSomething());
                function(aCell, &aLeaf);
            });

            ++iterCells;
            ++iterLeaves;
        }

        FAssertLF(iterCells == iterEndCells && iterLeaves == iterEndLeaves);
    }



    /** @brief, for statistic purpose, display each block with number of
   * cell, size of header, starting index, and ending index
   */
    void printInfoBlocks(){
        std::cout << "Group Tree information:\n";
        std::cout << "\t Group Size = " << nbElementsPerBlock << "\n";
        std::cout << "\t Tree height = " << treeHeight << "\n";
        for(int idxLevel = 1 ; idxLevel < treeHeight ; ++idxLevel){
            std::vector<CellGroupClass*>& levelBlocks = cellBlocksPerLevel[idxLevel];
            std::cout << "Level " << idxLevel << ", there are " << levelBlocks.size() << " groups.\n";
            int idxGroup = 0;
            for (const CellGroupClass* block: levelBlocks){
                std::cout << "\t Group " << (idxGroup++);
                std::cout << "\t Size = " << block->getNumberOfCellsInBlock();
                std::cout << "\t Starting Index = " << block->getStartingIndex();
                std::cout << "\t Ending Index = " << block->getEndingIndex();
                std::cout << "\t Ratio of usage = " << float(block->getNumberOfCellsInBlock())/float(block->getEndingIndex()-block->getStartingIndex()) << "\n";
            }
        }

        std::cout << "There are " << particleBlocks.size() << " leaf-groups.\n";
        int idxGroup = 0;
        for (const ParticleGroupClass* block: particleBlocks){
            std::cout << "\t Group " << (idxGroup++);
            std::cout << "\t Size = " << block->getNumberOfLeavesInBlock();
            std::cout << "\t Starting Index = " << block->getStartingIndex();
            std::cout << "\t Ending Index = " << block->getEndingIndex();
            std::cout << "\t Ratio of usage = " << float(block->getNumberOfLeavesInBlock())/float(block->getEndingIndex()-block->getStartingIndex()) << "\n";
        }
    }

    /////////////////////////////////////////////////////////
    // Algorithm function
    /////////////////////////////////////////////////////////

    int getHeight() const {
        return treeHeight;
    }

    CellGroupIterator cellsBegin(const int inLevel){
        FAssertLF(inLevel < treeHeight);
        return cellBlocksPerLevel[inLevel].begin();
    }

    CellGroupConstIterator cellsBegin(const int inLevel) const {
        FAssertLF(inLevel < treeHeight);
        return cellBlocksPerLevel[inLevel].begin();
    }

    CellGroupIterator cellsEnd(const int inLevel){
        FAssertLF(inLevel < treeHeight);
        return cellBlocksPerLevel[inLevel].end();
    }

    CellGroupConstIterator cellsEnd(const int inLevel) const {
        FAssertLF(inLevel < treeHeight);
        return cellBlocksPerLevel[inLevel].end();
    }

    int getNbCellGroupAtLevel(const int inLevel) const {
        FAssertLF(inLevel < treeHeight);
        return int(cellBlocksPerLevel[inLevel].size());
    }

    CellGroupClass* getCellGroup(const int inLevel, const int inIdx){
        FAssertLF(inLevel < treeHeight);
        FAssertLF(inIdx < int(cellBlocksPerLevel[inLevel].size()));
        return cellBlocksPerLevel[inLevel][inIdx];
    }

    const CellGroupClass* getCellGroup(const int inLevel, const int inIdx) const {
        FAssertLF(inLevel < treeHeight);
        FAssertLF(inIdx < int(cellBlocksPerLevel[inLevel].size()));
        return cellBlocksPerLevel[inLevel][inIdx];
    }

    ParticleGroupIterator leavesBegin(){
        return particleBlocks.begin();
    }

    ParticleGroupConstIterator leavesBegin() const {
        return particleBlocks.begin();
    }

    ParticleGroupIterator leavesEnd(){
        return particleBlocks.end();
    }

    ParticleGroupConstIterator leavesEnd() const {
        return particleBlocks.end();
    }

    int getNbParticleGroup() const {
        return int(particleBlocks.size());
    }

    ParticleGroupClass* getParticleGroup(const int inIdx){
        FAssertLF(inIdx < int(particleBlocks.size()));
        return particleBlocks[inIdx];
    }

    const ParticleGroupClass* getParticleGroup(const int inIdx) const {
        FAssertLF(inIdx < int(particleBlocks.size()));
        return particleBlocks[inIdx];
    }
};

#endif // FGROUPTREEDYN_HPP
