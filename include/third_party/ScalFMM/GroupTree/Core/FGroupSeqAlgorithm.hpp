
// Keep in private GIT
#ifndef FGROUPSEQALGORITHM_HPP
#define FGROUPSEQALGORITHM_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Core/FCoreCommon.hpp"
#include "../../Utils/FQuickSort.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../../Utils/FLog.hpp"
#include "../../Utils/FTic.hpp"

#include "FOutOfBlockInteraction.hpp"

#include <vector>
#include <vector>

template <class OctreeClass, class CellContainerClass, class CellClass, class KernelClass, class ParticleGroupClass, class ParticleContainerClass>
class FGroupSeqAlgorithm : public FAbstractAlgorithm {
protected:
    const int MaxThreads;         //< The number of threads
    OctreeClass*const tree;       //< The Tree
    KernelClass*const kernels;    //< The kernels

public:
    FGroupSeqAlgorithm(OctreeClass*const inTree, KernelClass* inKernels) : MaxThreads(1), tree(inTree), kernels(inKernels){
        FAssertLF(tree, "tree cannot be null");
        FAssertLF(kernels, "kernels cannot be null");

        FAbstractAlgorithm::setNbLevelsInTree(tree->getHeight());

        FLOG(FLog::Controller << "FGroupSeqAlgorithm (Max Thread " << MaxThreads << ")\n");
    }

    ~FGroupSeqAlgorithm(){
    }

protected:
    /**
      * Runs the complete algorithm.
      */
    void executeCore(const unsigned operationsToProceed) override {
        FLOG( FLog::Controller << "\tStart FGroupSeqAlgorithm\n" );

        if(operationsToProceed & FFmmP2M) bottomPass();

        if(operationsToProceed & FFmmM2M) upwardPass();

        if(operationsToProceed & FFmmM2L) transferPass();

        if(operationsToProceed & FFmmL2L) downardPass();

        if( (operationsToProceed & FFmmP2P) || (operationsToProceed & FFmmL2P) ){
            directPass((operationsToProceed & FFmmP2P), (operationsToProceed & FFmmL2P));
        }
    }

    void bottomPass(){
        FLOG( FTic timer; );
        typename OctreeClass::ParticleGroupIterator iterParticles = tree->leavesBegin();
        const typename OctreeClass::ParticleGroupIterator endParticles = tree->leavesEnd();

        typename OctreeClass::CellGroupIterator iterCells = tree->cellsBegin(tree->getHeight()-1);
        const typename OctreeClass::CellGroupIterator endCells = tree->cellsEnd(tree->getHeight()-1);

        while(iterParticles != endParticles && iterCells != endCells){
            { // Can be a task(in:iterParticles, out:iterCells)
                FAssertLF((*iterCells)->getNumberOfCellsInBlock() == (*iterParticles)->getNumberOfLeavesInBlock());

                for(int leafIdx = 0 ; leafIdx < (*iterCells)->getNumberOfCellsInBlock() ; ++leafIdx){
                    CellClass cell = (*iterCells)->getUpCell(leafIdx);
                    ParticleContainerClass particles = (*iterParticles)->template getLeaf<ParticleContainerClass>(leafIdx);
                    FAssertLF((*iterCells)->getCellMortonIndex(leafIdx) == (*iterParticles)->getLeafMortonIndex(leafIdx));
                    kernels->P2M(&cell, &particles);
                }
            }

            ++iterParticles;
            ++iterCells;
        }

        FAssertLF(iterParticles == endParticles && iterCells == endCells);
        FLOG( FLog::Controller << "\t\t bottomPass in " << timer.tacAndElapsed() << "s\n" );
    }

    void upwardPass(){
        FLOG( FTic timer; );
        for(int idxLevel = FMath::Min(tree->getHeight() - 2, FAbstractAlgorithm::lowerWorkingLevel - 1) ; idxLevel >= FAbstractAlgorithm::upperWorkingLevel ; --idxLevel){
            typename OctreeClass::CellGroupIterator iterCells = tree->cellsBegin(idxLevel);
            const typename OctreeClass::CellGroupIterator endCells = tree->cellsEnd(idxLevel);

            typename OctreeClass::CellGroupIterator iterChildCells = tree->cellsBegin(idxLevel+1);
            const typename OctreeClass::CellGroupIterator endChildCells = tree->cellsEnd(idxLevel+1);

            int idxChildCell = 0;

            while(iterCells != endCells && iterChildCells != endChildCells){
                { // Can be a task(in:iterParticles, out:iterChildCells ...)
                    CellClass childData[8];

                    for(int cellIdx = 0 ; cellIdx < (*iterCells)->getNumberOfCellsInBlock() ; ++cellIdx){
                        CellClass cell = (*iterCells)->getUpCell(cellIdx);
                        CellClass* child[8] = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};

                        FAssertLF(iterChildCells != endChildCells);

                        while(iterChildCells != endChildCells
                              && ((*iterChildCells)->getCellMortonIndex(idxChildCell)>>3) == cell.getMortonIndex()){
                            const int idxChild = (((*iterChildCells)->getCellMortonIndex(idxChildCell)) & 7);
                            FAssertLF(child[idxChild] == nullptr);
                            childData[idxChild] = (*iterChildCells)->getUpCell(idxChildCell);
                            FAssertLF((*iterChildCells)->getCellMortonIndex(idxChildCell) == childData[idxChild].getMortonIndex());
                            child[idxChild] = &childData[idxChild];
                            idxChildCell += 1;
                            if(idxChildCell == (*iterChildCells)->getNumberOfCellsInBlock()){
                                idxChildCell = 0;
                                ++iterChildCells;
                            }
                        }

                        kernels->M2M(&cell, child, idxLevel);
                    }
                }

                ++iterCells;
            }

            FAssertLF(iterCells == endCells);
            FAssertLF(iterChildCells == endChildCells);
            FAssertLF(iterCells == endCells && (iterChildCells == endChildCells || (++iterChildCells) == endChildCells));
        }
        FLOG( FLog::Controller << "\t\t upwardPass in " << timer.tacAndElapsed() << "s\n" );
    }

    void transferPass(){
        FLOG( FTic timer; );
        for(int idxLevel = FAbstractAlgorithm::lowerWorkingLevel-1 ; idxLevel >= FAbstractAlgorithm::upperWorkingLevel ; --idxLevel){
            typename OctreeClass::CellGroupIterator iterCells = tree->cellsBegin(idxLevel);
            const typename OctreeClass::CellGroupIterator endCells = tree->cellsEnd(idxLevel);

            while(iterCells != endCells){
                std::vector<OutOfBlockInteraction> outsideInteractions;

                { // Can be a task(inout:iterCells, out:outsideInteractions)
                    CellClass interactionsData[189];
                    const CellClass* interactions[189];
                    const MortonIndex blockStartIdx = (*iterCells)->getStartingIndex();
                    const MortonIndex blockEndIdx = (*iterCells)->getEndingIndex();

                    for(int cellIdx  = 0 ; cellIdx < (*iterCells)->getNumberOfCellsInBlock() ; ++cellIdx){
                        CellClass cell = (*iterCells)->getDownCell(cellIdx);
                        const MortonIndex mindex = (*iterCells)->getCellMortonIndex(cellIdx);
                        FAssertLF(cell.getMortonIndex() == mindex);
                        MortonIndex interactionsIndexes[189];
                        int interactionsPosition[189];
                        const FTreeCoordinate coord(cell.getCoordinate());
                        int counter = coord.getInteractionNeighbors(idxLevel,interactionsIndexes,interactionsPosition);

                        int counterExistingCell = 0;

                        for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
                            if( blockStartIdx <= interactionsIndexes[idxInter] && interactionsIndexes[idxInter] < blockEndIdx ){
                                const int cellPos = (*iterCells)->getCellIndex(interactionsIndexes[idxInter]);
                                if(cellPos != -1){
                                    CellClass interCell = (*iterCells)->getUpCell(cellPos);
                                    FAssertLF(interCell.getMortonIndex() == interactionsIndexes[idxInter]);
                                    interactionsPosition[counterExistingCell] = interactionsPosition[idxInter];
                                    interactionsData[counterExistingCell] = interCell;
                                    interactions[counterExistingCell] = &interactionsData[counterExistingCell];
                                    counterExistingCell += 1;
                                }
                            }
                            else if(interactionsIndexes[idxInter] < mindex){
                                OutOfBlockInteraction property;
                                property.insideIndex = mindex;
                                property.outIndex    = interactionsIndexes[idxInter];
                                property.relativeOutPosition = interactionsPosition[idxInter];
                                property.insideIdxInBlock = cellIdx;
                                outsideInteractions.push_back(property);
                            }
                        }

                        kernels->M2L( &cell , interactions, interactionsPosition, counterExistingCell, idxLevel);
                    }
                }


                // Manage outofblock interaction
                FQuickSort<OutOfBlockInteraction, int>::QsSequential(outsideInteractions.data(),int(outsideInteractions.size()));

                typename OctreeClass::CellGroupIterator iterLeftCells = tree->cellsBegin(idxLevel);
                int currentOutInteraction = 0;
                while(iterLeftCells != iterCells && currentOutInteraction < int(outsideInteractions.size())){
                    const MortonIndex outBlockStartIdx = (*iterLeftCells)->getStartingIndex();
                    const MortonIndex outBlockEndIdx = (*iterLeftCells)->getEndingIndex();

                    while(currentOutInteraction < int(outsideInteractions.size()) && outsideInteractions[currentOutInteraction].outIndex < outBlockStartIdx){
                        currentOutInteraction += 1;
                    }

                    int lastOutInteraction = currentOutInteraction;
                    while(lastOutInteraction < int(outsideInteractions.size()) && outsideInteractions[lastOutInteraction].outIndex < outBlockEndIdx){
                        lastOutInteraction += 1;
                    }

                    { // Can be a task(in:currentOutInteraction, in:outsideInteractions, in:lastOutInteraction, inout:iterLeftCells, inout:iterCells)

                        for(int outInterIdx = currentOutInteraction ; outInterIdx < lastOutInteraction ; ++outInterIdx){
                            const int cellPos = (*iterLeftCells)->getCellIndex(outsideInteractions[outInterIdx].outIndex);
                            if(cellPos != -1){
                                CellClass interCell = (*iterLeftCells)->getCompleteCell(cellPos);
                                FAssertLF(interCell.getMortonIndex() == outsideInteractions[outInterIdx].outIndex);
                                CellClass cell = (*iterCells)->getCompleteCell(outsideInteractions[outInterIdx].insideIdxInBlock);
                                FAssertLF(cell.getMortonIndex() == outsideInteractions[outInterIdx].insideIndex);

                                const CellClass* ptCell = &interCell;
                                kernels->M2L( &cell , &ptCell, &outsideInteractions[outInterIdx].relativeOutPosition, 1, idxLevel);
                                const int otherPos = getOppositeInterIndex(outsideInteractions[outInterIdx].relativeOutPosition);
                                ptCell = &cell;
                                kernels->M2L( &interCell , &ptCell, &otherPos, 1, idxLevel);
                            }
                        }
                    }

                    currentOutInteraction = lastOutInteraction;
                    ++iterLeftCells;
                }

                ++iterCells;
            }

        }
        FLOG( FLog::Controller << "\t\t transferPass in " << timer.tacAndElapsed() << "s\n" );
    }

    void downardPass(){
        FLOG( FTic timer; );
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel - 1 ; ++idxLevel){
            typename OctreeClass::CellGroupIterator iterCells = tree->cellsBegin(idxLevel);
            const typename OctreeClass::CellGroupIterator endCells = tree->cellsEnd(idxLevel);

            typename OctreeClass::CellGroupIterator iterChildCells = tree->cellsBegin(idxLevel+1);
            const typename OctreeClass::CellGroupIterator endChildCells = tree->cellsEnd(idxLevel+1);

            int idxChildCell = 0;

            while(iterCells != endCells && iterChildCells != endChildCells){
                { // Can be a task(in:iterParticles, inout:iterChildCells ...)
                    CellClass childData[8];

                    for(int cellIdx = 0 ; cellIdx < (*iterCells)->getNumberOfCellsInBlock() ; ++cellIdx){
                        CellClass cell = (*iterCells)->getDownCell(cellIdx);
                        FAssertLF(cell.getMortonIndex() == (*iterCells)->getCellMortonIndex(cellIdx));
                        CellClass* child[8] = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};

                        FAssertLF(iterChildCells != endChildCells);

                        while(iterChildCells != endChildCells
                              && ((*iterChildCells)->getCellMortonIndex(idxChildCell)>>3) == cell.getMortonIndex()){
                            const int idxChild = (((*iterChildCells)->getCellMortonIndex(idxChildCell)) & 7);
                            FAssertLF(child[idxChild] == nullptr);
                            childData[idxChild] = (*iterChildCells)->getDownCell(idxChildCell);
                            FAssertLF((*iterChildCells)->getCellMortonIndex(idxChildCell) == childData[idxChild].getMortonIndex());
                            child[idxChild] = &childData[idxChild];
                            idxChildCell += 1;
                            if(idxChildCell == (*iterChildCells)->getNumberOfCellsInBlock()){
                                idxChildCell = 0;
                                ++iterChildCells;
                            }
                        }

                        kernels->L2L(&cell, child, idxLevel);
                    }
                }

                ++iterCells;
            }

            FAssertLF(iterCells == endCells && iterChildCells == endChildCells);
        }
        FLOG( FLog::Controller << "\t\t downardPass in " << timer.tacAndElapsed() << "s\n" );
    }

    void directPass(const bool p2pEnabled, const bool l2pEnabled){
        FLOG( FTic timer; );
        if(l2pEnabled){
            typename OctreeClass::ParticleGroupIterator iterParticles = tree->leavesBegin();
            const typename OctreeClass::ParticleGroupIterator endParticles = tree->leavesEnd();

            typename OctreeClass::CellGroupIterator iterCells = tree->cellsBegin(tree->getHeight()-1);
            const typename OctreeClass::CellGroupIterator endCells = tree->cellsEnd(tree->getHeight()-1);

            while(iterParticles != endParticles && iterCells != endCells){
                { // Can be a task(in:iterCells, inout:iterParticles)                    
                    for(int leafIdx = 0 ; leafIdx < (*iterCells)->getNumberOfCellsInBlock() ; ++leafIdx){
                        CellClass cell = (*iterCells)->getDownCell(leafIdx);
                        ParticleContainerClass particles = (*iterParticles)->template getLeaf<ParticleContainerClass>(leafIdx);
                        FAssertLF((*iterCells)->getCellMortonIndex(leafIdx) == (*iterParticles)->getLeafMortonIndex(leafIdx));
                        kernels->L2P(&cell, &particles);
                    }
                }

                ++iterParticles;
                ++iterCells;
            }

            FAssertLF(iterParticles == endParticles && iterCells == endCells);
        }
        if(p2pEnabled){
            typename OctreeClass::ParticleGroupIterator iterParticles = tree->leavesBegin();
            const typename OctreeClass::ParticleGroupIterator endParticles = tree->leavesEnd();

            while(iterParticles != endParticles){
                typename std::vector<OutOfBlockInteraction> outsideInteractions;

                { // Can be a task(inout:iterCells, out:outsideInteractions)
                    const MortonIndex blockStartIdx = (*iterParticles)->getStartingIndex();
                    const MortonIndex blockEndIdx = (*iterParticles)->getEndingIndex();

                    for(int leafIdx = 0 ; leafIdx < (*iterParticles)->getNumberOfLeavesInBlock() ; ++leafIdx){
                        ParticleContainerClass particles = (*iterParticles)->template getLeaf<ParticleContainerClass>(leafIdx);

                        const MortonIndex mindex = (*iterParticles)->getLeafMortonIndex(leafIdx);
                        MortonIndex interactionsIndexes[26];
                        int interactionsPosition[26];
                        FTreeCoordinate coord(mindex, tree->getHeight()-1);
                        int counter = coord.getNeighborsIndexes(tree->getHeight(),interactionsIndexes,interactionsPosition);

                        ParticleContainerClass interactionsObjects[26];
                        ParticleContainerClass* interactions[26];
                        int counterExistingCell = 0;

                        for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
                            if( blockStartIdx <= interactionsIndexes[idxInter] && interactionsIndexes[idxInter] < blockEndIdx ){
                                const int leafPos = (*iterParticles)->getLeafIndex(interactionsIndexes[idxInter]);
                                if(leafPos != -1){
                                    interactionsObjects[counterExistingCell] = (*iterParticles)->template getLeaf<ParticleContainerClass>(leafPos);
                                    interactionsPosition[counterExistingCell] = interactionsPosition[idxInter];
                                    interactions[counterExistingCell] = &interactionsObjects[counterExistingCell];
                                    counterExistingCell += 1;
                                }
                            }
                            else if(interactionsIndexes[idxInter] < mindex){
                                OutOfBlockInteraction property;
                                property.insideIndex = mindex;
                                property.outIndex    = interactionsIndexes[idxInter];
                                property.relativeOutPosition = interactionsPosition[idxInter];
                                property.insideIdxInBlock = leafIdx;
                                outsideInteractions.push_back(property);
                            }
                        }

                        kernels->P2P( coord, &particles, &particles , interactions, interactionsPosition, counterExistingCell);
                    }
                }


                // Manage outofblock interaction
                FQuickSort<OutOfBlockInteraction, int>::QsSequential(outsideInteractions.data(),int(outsideInteractions.size()));

                typename OctreeClass::ParticleGroupIterator iterLeftParticles = tree->leavesBegin();
                int currentOutInteraction = 0;
                while(iterLeftParticles != iterParticles && currentOutInteraction < int(outsideInteractions.size())){
                    const MortonIndex blockStartIdx = (*iterLeftParticles)->getStartingIndex();
                    const MortonIndex blockEndIdx = (*iterLeftParticles)->getEndingIndex();

                    while(currentOutInteraction < int(outsideInteractions.size()) && outsideInteractions[currentOutInteraction].outIndex < blockStartIdx){
                        currentOutInteraction += 1;
                    }

                    int lastOutInteraction = currentOutInteraction;
                    while(lastOutInteraction < int(outsideInteractions.size()) && outsideInteractions[lastOutInteraction].outIndex < blockEndIdx){
                        lastOutInteraction += 1;
                    }

                    { // Can be a task(in:currentOutInteraction, in:outsideInteractions, in:lastOutInteraction, inout:iterLeftParticles, inout:iterParticles)
                        for(int outInterIdx = currentOutInteraction ; outInterIdx < lastOutInteraction ; ++outInterIdx){
                            const int leafPos = (*iterLeftParticles)->getLeafIndex(outsideInteractions[outInterIdx].outIndex);
                            if(leafPos != -1){
                                ParticleContainerClass interParticles = (*iterLeftParticles)->template getLeaf<ParticleContainerClass>(leafPos);
                                ParticleContainerClass particles = (*iterParticles)->template getLeaf<ParticleContainerClass>(outsideInteractions[outInterIdx].insideIdxInBlock);

                                FAssertLF((*iterLeftParticles)->getLeafMortonIndex(leafPos) == outsideInteractions[outInterIdx].outIndex);
                                FAssertLF((*iterParticles)->getLeafMortonIndex(outsideInteractions[outInterIdx].insideIdxInBlock) == outsideInteractions[outInterIdx].insideIndex);

                                ParticleContainerClass* ptrLeaf = &interParticles;
                                kernels->P2POuter( FTreeCoordinate(outsideInteractions[outInterIdx].insideIndex, tree->getHeight()-1),
                                                    &particles , &ptrLeaf, &outsideInteractions[outInterIdx].relativeOutPosition, 1);
                                const int otherPosition = getOppositeNeighIndex(outsideInteractions[outInterIdx].relativeOutPosition);
                                ptrLeaf = &particles;
                                kernels->P2POuter( FTreeCoordinate(outsideInteractions[outInterIdx].outIndex, tree->getHeight()-1),
                                                    &interParticles , &ptrLeaf, &otherPosition, 1);
                            }
                        }
                    }

                    currentOutInteraction = lastOutInteraction;
                    ++iterLeftParticles;
                }

                ++iterParticles;
            }
        }
        FLOG( FLog::Controller << "\t\t directPass in " << timer.tacAndElapsed() << "s\n" );
    }

    int getOppositeNeighIndex(const int index) const {
        // ((idxX+1)*3 + (idxY+1)) * 3 + (idxZ+1)
        return 27-index-1;
    }

    int getOppositeInterIndex(const int index) const {
        // ((( (xdiff+3) * 7) + (ydiff+3))) * 7 + zdiff + 3
        return 343-index-1;
    }
};


#endif // FGROUPSEQALGORITHM_HPP
