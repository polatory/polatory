#include "FCudaDeviceWrapper.hpp"
#include "FCudaTreeCoordinate.hpp"
#include "FCudaStructParams.hpp"


#define FMGetOppositeNeighIndex(index) (27-(index)-1)
#define FMGetOppositeInterIndex(index) (343-(index)-1)

#define FCudaMax(x,y) ((x)<(y) ? (y) : (x))
#define FCudaMin(x,y) ((x)>(y) ? (y) : (x))


template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__global__ void FCuda__bottomPassPerform(unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
                                         unsigned char* containersPtr, std::size_t containersSize,
                                         CudaKernelClass* kernel){
    CellContainerClass leafCells(leafCellsPtr, leafCellsSize, leafCellsUpPtr, nullptr);
    ParticleContainerGroupClass containers(containersPtr, containersSize, nullptr);

    for(int leafIdx = blockIdx.x ; leafIdx < leafCells.getNumberOfCellsInBlock() ; leafIdx += gridDim.x){
        typename CellContainerClass::CompleteCellClass cell = leafCells.getUpCell(leafIdx);
        ParticleGroupClass particles = containers.template getLeaf<ParticleGroupClass>(leafIdx);
        FCudaAssertLF(leafCells.getCellMortonIndex(leafIdx) == containers.getLeafMortonIndex(leafIdx));
        kernel->P2M(cell, &particles);
    }
}

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__host__ void FCuda__bottomPassCallback(unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
                                        unsigned char* containersPtr, std::size_t containersSize,
                                        CudaKernelClass* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize){
    FCuda__bottomPassPerform
            <SymboleCellClass, PoleCellClass, LocalCellClass,
            CellContainerClass, ParticleContainerGroupClass, ParticleGroupClass, CudaKernelClass>
            <<<inGridSize, inBlocksSize, 0, currentStream>>>
                            (leafCellsPtr, leafCellsSize,leafCellsUpPtr,
                             containersPtr, containersSize,
                             kernel);
    FCudaCheckAfterCall();
    FCudaCheck(cudaStreamSynchronize(currentStream));
}


/////////////////////////////////////////////////////////////////////////////////////
/// Upward Pass
/////////////////////////////////////////////////////////////////////////////////////

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__global__ void FCuda__upwardPassPerform(unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
                                         unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
                                         int idxLevel, CudaKernelClass* kernel){
    CellContainerClass currentCells(currentCellsPtr, currentCellsSize,currentCellsUpPtr,nullptr);
    CellContainerClass subCellGroup(childCellsPtr, childCellsSize,childCellsUpPtr,nullptr);

    const MortonIndex firstParent = FCudaMax(currentCells.getStartingIndex(), subCellGroup.getStartingIndex()>>3);
    const MortonIndex lastParent = FCudaMin(currentCells.getEndingIndex()-1, (subCellGroup.getEndingIndex()-1)>>3);

    int idxParentCell = currentCells.getCellIndex(firstParent);
    int idxChildCell = subCellGroup.getFistChildIdx(firstParent);

    while(true){
        typename CellContainerClass::CompleteCellClass cell = currentCells.getUpCell(idxParentCell);
        typename CellContainerClass::CompleteCellClass child[8];


        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            child[idxChild].symb = nullptr;
        }

        do{
            const int idxChild = ((subCellGroup.getCellMortonIndex(idxChildCell)) & 7);
            child[idxChild] = subCellGroup.getUpCell(idxChildCell);

            idxChildCell += 1;
        }while(idxChildCell != subCellGroup.getNumberOfCellsInBlock() && cell.symb->mortonIndex == (subCellGroup.getCellMortonIndex(idxChildCell)>>3));

        kernel->M2M(cell, child, idxLevel);

        if(currentCells.getCellMortonIndex(idxParentCell) == lastParent){
            break;
        }

        idxParentCell += 1;
    }
}

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__host__ void FCuda__upwardPassCallback(unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
                                        unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
                                        int idxLevel, CudaKernelClass* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize){

    FCuda__upwardPassPerform
            <SymboleCellClass, PoleCellClass, LocalCellClass,
            CellContainerClass, ParticleContainerGroupClass, ParticleGroupClass, CudaKernelClass>
            <<<inGridSize, inBlocksSize, 0, currentStream>>>
                            (currentCellsPtr, currentCellsSize,currentCellsUpPtr,
                             childCellsPtr, childCellsSize,childCellsUpPtr,
                             idxLevel, kernel);
    FCudaCheckAfterCall();
    FCudaCheck(cudaStreamSynchronize(currentStream));
}



/////////////////////////////////////////////////////////////////////////////////////
/// Transfer Pass Mpi
/////////////////////////////////////////////////////////////////////////////////////
#ifdef SCALFMM_USE_MPI
template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__global__  void FCuda__transferInoutPassPerformMpi(unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
                                                  unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
                                                  const int* safeInteractions, int nbSafeInteractions, int idxLevel, const OutOfBlockInteraction* outsideInteractions,
                                                  int nbOutsideInteractions, CudaKernelClass* kernel){

    CellContainerClass currentCells(currentCellsPtr, currentCellsSize, nullptr, currentCellsDownPtr);
    CellContainerClass cellsOther(externalCellsPtr, externalCellsSize, externalCellsUpPtr, nullptr);

    for(int cellIdx = blockIdx.x ; cellIdx < nbSafeInteractions ; cellIdx += gridDim.x){
        for(int outInterIdx = safeInteractions[cellIdx] ; outInterIdx < safeInteractions[cellIdx+1] ; ++outInterIdx){
            const int cellPos = cellsOther.getCellIndex(outsideInteractions[outInterIdx].outIndex);
            if(cellPos != -1){
                typename CellContainerClass::CompleteCellClass interCell = cellsOther.getUpCell(cellPos);
                FCudaAssertLF(interCell.symb->mortonIndex == outsideInteractions[outInterIdx].outIndex);
                typename CellContainerClass::CompleteCellClass cell = currentCells.getDownCell(outsideInteractions[outInterIdx].insideIdxInBlock);
                FCudaAssertLF(cell.symb->mortonIndex == outsideInteractions[outInterIdx].insideIndex);

                kernel->M2L( cell , &interCell, &outsideInteractions[outInterIdx].relativeOutPosition, 1, idxLevel);
            }
        }
    }
}

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__host__ void FCuda__transferInoutPassCallbackMpi(unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
                                                  unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
                                                  int idxLevel, const OutOfBlockInteraction* outsideInteractions,
                                                  int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, CudaKernelClass* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize){
    OutOfBlockInteraction* cuOutsideInteractions;
    FCudaCheck( cudaMalloc(&cuOutsideInteractions,nbOutsideInteractions*sizeof(OutOfBlockInteraction)) );
    FCudaCheck( cudaMemcpy( cuOutsideInteractions, outsideInteractions, nbOutsideInteractions*sizeof(OutOfBlockInteraction),
                cudaMemcpyHostToDevice ) );

    int* cuSafeInteractions;
    FCudaCheck( cudaMalloc(&cuSafeInteractions,(nbSafeInteractions+1)*sizeof(int)) );
    FCudaCheck( cudaMemcpy( cuSafeInteractions, safeInteractions, (nbSafeInteractions+1)*sizeof(int),
                cudaMemcpyHostToDevice ) );

    FCuda__transferInoutPassPerformMpi
            <SymboleCellClass, PoleCellClass, LocalCellClass,
            CellContainerClass, ParticleContainerGroupClass, ParticleGroupClass, CudaKernelClass>
            <<<inGridSize, inBlocksSize, 0, currentStream>>>(currentCellsPtr, currentCellsSize, currentCellsDownPtr,
                                       externalCellsPtr, externalCellsSize, externalCellsUpPtr,
                                       cuSafeInteractions, nbSafeInteractions, idxLevel, cuOutsideInteractions, nbOutsideInteractions, kernel);
    FCudaCheckAfterCall();
    FCudaCheck(cudaStreamSynchronize(currentStream));

    FCudaCheck(cudaFree(cuSafeInteractions));
    FCudaCheck(cudaFree(cuOutsideInteractions));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////
/// Transfer Pass
/////////////////////////////////////////////////////////////////////////////////////


template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__global__  void FCuda__transferInPassPerform(unsigned char* currentCellsPtr, std::size_t currentCellsSize,
                                              unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
                                              int idxLevel, CudaKernelClass* kernel){

    CellContainerClass currentCells(currentCellsPtr, currentCellsSize, currentCellsUpPtr, currentCellsDownPtr);

    const MortonIndex blockStartIdx = currentCells.getStartingIndex();
    const MortonIndex blockEndIdx = currentCells.getEndingIndex();

    for(int cellIdx = blockIdx.x ; cellIdx < currentCells.getNumberOfCellsInBlock() ; cellIdx += gridDim.x){
        typename CellContainerClass::CompleteCellClass cell = currentCells.getDownCell(cellIdx);

        MortonIndex interactionsIndexes[189];
        int interactionsPosition[189];
        const int3 coord = (FCudaTreeCoordinate::ConvertCoordinate(cell.symb->coordinates));
        int counter = FCudaTreeCoordinate::GetInteractionNeighbors(coord, idxLevel,interactionsIndexes,interactionsPosition);

        typename CellContainerClass::CompleteCellClass interactions[189];
        int counterExistingCell = 0;

        for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
            if( blockStartIdx <= interactionsIndexes[idxInter] && interactionsIndexes[idxInter] < blockEndIdx ){
                const int cellPos = currentCells.getCellIndex(interactionsIndexes[idxInter]);
                if(cellPos != -1){
                    typename CellContainerClass::CompleteCellClass interCell = currentCells.getUpCell(cellPos);
                    interactions[counterExistingCell] = interCell;
                    interactionsPosition[counterExistingCell] = interactionsPosition[idxInter];
                    counterExistingCell += 1;
                }
            }
        }

        kernel->M2L( cell , interactions, interactionsPosition, counterExistingCell, idxLevel);
    }
}

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__host__ void FCuda__transferInPassCallback(unsigned char* currentCellsPtr, std::size_t currentCellsSize,
                                            unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
                                            int idxLevel, CudaKernelClass* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize){

    FCuda__transferInPassPerform
            <SymboleCellClass, PoleCellClass, LocalCellClass,
            CellContainerClass, ParticleContainerGroupClass, ParticleGroupClass, CudaKernelClass>
            <<<inGridSize, inBlocksSize, 0, currentStream>>>(currentCellsPtr, currentCellsSize,
                                                                currentCellsUpPtr, currentCellsDownPtr,
                                                                idxLevel, kernel);
    FCudaCheckAfterCall();
    FCudaCheck(cudaStreamSynchronize(currentStream));
}


template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__global__ void FCuda__transferInoutPassPerform(unsigned char* currentCellsPtr, std::size_t currentCellsSize,
                                                unsigned char* currentCellsDownPtr,
                                                unsigned char* externalCellsPtr, std::size_t externalCellsSize,
                                                unsigned char* externalCellsUpPtr,
                                                int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
                                                int nbOutsideInteractions,
                                                const int* safeInteractions, int nbSafeInteractions, CudaKernelClass* kernel){

    CellContainerClass currentCells(currentCellsPtr, currentCellsSize, nullptr, currentCellsDownPtr);
    CellContainerClass cellsOther(externalCellsPtr, externalCellsSize, externalCellsUpPtr, nullptr);

    if(mode == 1){
        for(int cellIdx = blockIdx.x ; cellIdx < nbSafeInteractions ; cellIdx += gridDim.x){
            for(int outInterIdx = safeInteractions[cellIdx] ; outInterIdx < safeInteractions[cellIdx+1] ; ++outInterIdx){
                typename CellContainerClass::CompleteCellClass interCell = cellsOther.getUpCell(outsideInteractions[outInterIdx].outsideIdxInBlock);
                FCudaAssertLF(interCell.symb->mortonIndex == outsideInteractions[outInterIdx].outIndex);
                typename CellContainerClass::CompleteCellClass cell = currentCells.getDownCell(outsideInteractions[outInterIdx].insideIdxInBlock);
                FCudaAssertLF(cell.symb->mortonIndex == outsideInteractions[outInterIdx].insideIndex);

                kernel->M2L( cell , &interCell, &outsideInteractions[outInterIdx].relativeOutPosition, 1, idxLevel);
            }
        }
    }
    else{
        for(int cellIdx = blockIdx.x ; cellIdx < nbSafeInteractions ; cellIdx += gridDim.x){
            for(int outInterIdx = safeInteractions[cellIdx] ; outInterIdx < safeInteractions[cellIdx+1] ; ++outInterIdx){
                typename CellContainerClass::CompleteCellClass cell = cellsOther.getUpCell(outsideInteractions[outInterIdx].insideIdxInBlock);
                FCudaAssertLF(cell.symb->mortonIndex == outsideInteractions[outInterIdx].insideIndex);
                typename CellContainerClass::CompleteCellClass interCell = currentCells.getDownCell(outsideInteractions[outInterIdx].outsideIdxInBlock);
                FCudaAssertLF(interCell.symb->mortonIndex == outsideInteractions[outInterIdx].outIndex);

                const int otherPosition = FMGetOppositeInterIndex(outsideInteractions[outInterIdx].relativeOutPosition);
                kernel->M2L( interCell , &cell, &otherPosition, 1, idxLevel);
            }
        }
    }
}


template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__host__ void FCuda__transferInoutPassCallback(unsigned char* currentCellsPtr, std::size_t currentCellsSize,
                                               unsigned char* currentCellsDownPtr,
                                               unsigned char* externalCellsPtr, std::size_t externalCellsSize,
                                               unsigned char* externalCellsUpPtr,
                                               int idxLevel, int mode,
                                               const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
                                               const int* safeInteractions, int nbSafeInteractions,
                                               CudaKernelClass* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize){
    OutOfBlockInteraction* cuOutsideInteractions;
    FCudaCheck( cudaMalloc(&cuOutsideInteractions,nbOutsideInteractions*sizeof(OutOfBlockInteraction)) );
    FCudaCheck( cudaMemcpy( cuOutsideInteractions, outsideInteractions, nbOutsideInteractions*sizeof(OutOfBlockInteraction),
                cudaMemcpyHostToDevice ) );

    int* cuSafeInteractions;
    FCudaCheck( cudaMalloc(&cuSafeInteractions,(nbSafeInteractions+1)*sizeof(int)) );
    FCudaCheck( cudaMemcpy( cuSafeInteractions, safeInteractions, (nbSafeInteractions+1)*sizeof(int),
                cudaMemcpyHostToDevice ) );

    FCuda__transferInoutPassPerform
            <SymboleCellClass, PoleCellClass, LocalCellClass,
            CellContainerClass, ParticleContainerGroupClass, ParticleGroupClass, CudaKernelClass>
            <<<inGridSize, inBlocksSize, 0, currentStream>>>(currentCellsPtr, currentCellsSize,
                                                                currentCellsDownPtr,
                                                                externalCellsPtr, externalCellsSize,
                                                                externalCellsUpPtr,
                                                                idxLevel, mode,
                                                                cuOutsideInteractions, nbOutsideInteractions,
                                                                cuSafeInteractions, nbSafeInteractions,
                                                                kernel);
    FCudaCheckAfterCall();
    FCudaCheck(cudaStreamSynchronize(currentStream));

    FCudaCheck(cudaFree(cuOutsideInteractions));
    FCudaCheck(cudaFree(cuSafeInteractions));
}


/////////////////////////////////////////////////////////////////////////////////////
/// Downard Pass
/////////////////////////////////////////////////////////////////////////////////////

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__global__ void FCuda__downardPassPerform(unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
                                          unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
                                          int idxLevel, CudaKernelClass* kernel){
    CellContainerClass currentCells(currentCellsPtr, currentCellsSize,nullptr,currentCellsDownPtr);
    CellContainerClass subCellGroup(childCellsPtr, childCellsSize,nullptr,childCellsDownPtr);

    const MortonIndex firstParent = FCudaMax(currentCells.getStartingIndex(), subCellGroup.getStartingIndex()>>3);
    const MortonIndex lastParent = FCudaMin(currentCells.getEndingIndex()-1, (subCellGroup.getEndingIndex()-1)>>3);

    int idxParentCell = currentCells.getCellIndex(firstParent);
    int idxChildCell = subCellGroup.getFistChildIdx(firstParent);

    while(true){
        typename CellContainerClass::CompleteCellClass cell = currentCells.getDownCell(idxParentCell);
        typename CellContainerClass::CompleteCellClass child[8];


        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            child[idxChild].symb = nullptr;
        }

        do{
            const int idxChild = ((subCellGroup.getCellMortonIndex(idxChildCell)) & 7);
            child[idxChild] = subCellGroup.getDownCell(idxChildCell);

            idxChildCell += 1;
        }while(idxChildCell != subCellGroup.getNumberOfCellsInBlock() && cell.symb->mortonIndex == (subCellGroup.getCellMortonIndex(idxChildCell)>>3));

        kernel->L2L(cell, child, idxLevel);

        if(currentCells.getCellMortonIndex(idxParentCell) == lastParent){
            break;
        }

        idxParentCell += 1;
    }
}

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__host__ void FCuda__downardPassCallback(unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
                                        unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
                                         int idxLevel, CudaKernelClass* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize){

    FCuda__downardPassPerform
            <SymboleCellClass, PoleCellClass, LocalCellClass,
            CellContainerClass, ParticleContainerGroupClass, ParticleGroupClass, CudaKernelClass>
            <<<inGridSize, inBlocksSize, 0, currentStream>>>
            (currentCellsPtr, currentCellsSize, currentCellsDownPtr, childCellsPtr, childCellsSize, childCellsDownPtr,
             idxLevel, kernel);
    FCudaCheckAfterCall();
    FCudaCheck(cudaStreamSynchronize(currentStream));
}

/////////////////////////////////////////////////////////////////////////////////////
/// Direct Pass MPI
/////////////////////////////////////////////////////////////////////////////////////
#ifdef SCALFMM_USE_MPI
template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__global__ void FCuda__directInoutPassPerformMpi(unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
                                                 unsigned char* externalContainersPtr, std::size_t externalContainersSize,
                                                 const OutOfBlockInteraction* outsideInteractions,
                                                 int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
                                                 const int treeHeight, CudaKernelClass* kernel){

    ParticleContainerGroupClass containers(containersPtr, containersSize, containersDownPtr);
    ParticleContainerGroupClass containersOther(externalContainersPtr, externalContainersSize, nullptr);

    for(int leafIdx = blockIdx.x ; leafIdx < counterOuterCell ; leafIdx += gridDim.x){
        for(int outInterIdx = safeOuterInteractions[leafIdx] ; outInterIdx < safeOuterInteractions[leafIdx+1] ; ++outInterIdx){
            const int leafPos = containersOther.getLeafIndex(outsideInteractions[outInterIdx].outIndex);
            if(leafPos != -1){
                ParticleGroupClass interParticles = containersOther.template getLeaf<ParticleGroupClass>(leafPos);
                ParticleGroupClass particles = containers.template getLeaf<ParticleGroupClass>(outsideInteractions[outInterIdx].insideIdxInBlock);

                kernel->P2PRemote( FCudaTreeCoordinate::GetPositionFromMorton(outsideInteractions[outInterIdx].insideIndex, treeHeight-1),
                                   &particles, &particles , &interParticles, &outsideInteractions[outInterIdx].relativeOutPosition, 1);
            }
        }
    }
}

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__host__ void FCuda__directInoutPassCallbackMpi(unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
                                                unsigned char* externalContainersPtr, std::size_t externalContainersSize,
                                                const OutOfBlockInteraction* outsideInteractions,
                                                int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
                                                const int treeHeight, CudaKernelClass* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize){
    OutOfBlockInteraction* cuOutsideInteractions;
    FCudaCheck( cudaMalloc(&cuOutsideInteractions,nbOutsideInteractions*sizeof(OutOfBlockInteraction)) );
    FCudaCheck( cudaMemcpy( cuOutsideInteractions, outsideInteractions, nbOutsideInteractions*sizeof(OutOfBlockInteraction),
                cudaMemcpyHostToDevice ) );

    int* cuSafeOuterInteractions;
    FCudaCheck( cudaMalloc(&cuSafeOuterInteractions,(counterOuterCell+1)*sizeof(int)) );
    FCudaCheck( cudaMemcpy( cuSafeOuterInteractions, safeOuterInteractions, (counterOuterCell+1)*sizeof(int),
                cudaMemcpyHostToDevice ) );

    FCuda__directInoutPassPerformMpi
            <SymboleCellClass, PoleCellClass, LocalCellClass,
            CellContainerClass, ParticleContainerGroupClass, ParticleGroupClass, CudaKernelClass>
            <<<inGridSize, inBlocksSize, 0, currentStream>>>(containersPtr, containersSize, containersDownPtr,
                                  externalContainersPtr, externalContainersSize,
                                  cuOutsideInteractions, nbOutsideInteractions, cuSafeOuterInteractions, counterOuterCell,
                                                             treeHeight, kernel);

    FCudaCheckAfterCall();
    FCudaCheck(cudaStreamSynchronize(currentStream));

    FCudaCheck(cudaFree(cuOutsideInteractions));
    FCudaCheck(cudaFree(cuSafeOuterInteractions));
}
#endif
/////////////////////////////////////////////////////////////////////////////////////
/// Direct Pass
/////////////////////////////////////////////////////////////////////////////////////


template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__global__ void FCuda__directInPassPerform(unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
                                           const int treeHeight, CudaKernelClass* kernel){

    ParticleContainerGroupClass containers(containersPtr, containersSize, containersDownPtr);

    const MortonIndex blockStartIdx = containers.getStartingIndex();
    const MortonIndex blockEndIdx = containers.getEndingIndex();

    for(int leafIdx = blockIdx.x ; leafIdx < containers.getNumberOfLeavesInBlock() ; leafIdx += gridDim.x){
        ParticleGroupClass particles = containers.template getLeaf<ParticleGroupClass>(leafIdx);
        const MortonIndex mindex = containers.getLeafMortonIndex(leafIdx);
        MortonIndex interactionsIndexes[26];
        int interactionsPosition[26];
        const int3 coord = FCudaTreeCoordinate::GetPositionFromMorton(mindex, treeHeight-1);
        int counter = FCudaTreeCoordinate::GetNeighborsIndexes(coord, treeHeight,interactionsIndexes,interactionsPosition);

        ParticleGroupClass interactionsObjects[26];
        int counterExistingCell = 0;

        for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
            if( blockStartIdx <= interactionsIndexes[idxInter] && interactionsIndexes[idxInter] < blockEndIdx ){
                const int leafPos = containers.getLeafIndex(interactionsIndexes[idxInter]);
                if(leafPos != -1){
                    interactionsObjects[counterExistingCell] = containers.template getLeaf<ParticleGroupClass>(leafPos);
                    interactionsPosition[counterExistingCell] = interactionsPosition[idxInter];
                    counterExistingCell += 1;
                }
            }
        }

        kernel->P2P( coord, &particles, &particles , interactionsObjects, interactionsPosition, counterExistingCell);
    }
}

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__host__ void FCuda__directInPassCallback(unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
                                          const int treeHeight, CudaKernelClass* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize){
    FCuda__directInPassPerform
            <SymboleCellClass, PoleCellClass, LocalCellClass,
            CellContainerClass, ParticleContainerGroupClass, ParticleGroupClass, CudaKernelClass>
            <<<inGridSize, inBlocksSize, 0, currentStream>>>(containersPtr, containersSize, containersDownPtr,
                               treeHeight, kernel);
    FCudaCheckAfterCall();
    FCudaCheck(cudaStreamSynchronize(currentStream));
}

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__global__ void FCuda__directInoutPassPerform(unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
                                              unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
                                              const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
                                              const int     safeOuterInteractions[], const int counterOuterCell,
                                              const OutOfBlockInteraction* insideInteractions,
                                              const int     safeInnterInteractions[], const int counterInnerCell,
                                              const int treeHeight, CudaKernelClass* kernel){

    ParticleContainerGroupClass containers(containersPtr, containersSize, containersDownPtr);
    ParticleContainerGroupClass containersOther(externalContainersPtr, externalContainersSize, externalContainersDownPtr);

    for(int leafIdx = blockIdx.x ; leafIdx < counterOuterCell ; leafIdx += gridDim.x){
        for(int outInterIdx = safeOuterInteractions[leafIdx] ; outInterIdx < safeOuterInteractions[leafIdx+1] ; ++outInterIdx){
            ParticleGroupClass interParticles = containersOther.template getLeaf<ParticleGroupClass>(outsideInteractions[outInterIdx].outsideIdxInBlock);
            ParticleGroupClass particles = containers.template getLeaf<ParticleGroupClass>(outsideInteractions[outInterIdx].insideIdxInBlock);

            FCudaAssertLF(containersOther.getLeafMortonIndex(outsideInteractions[outInterIdx].outsideIdxInBlock) == outsideInteractions[outInterIdx].outIndex);
            FCudaAssertLF(containers.getLeafMortonIndex(outsideInteractions[outInterIdx].insideIdxInBlock) == outsideInteractions[outInterIdx].insideIndex);

            kernel->P2POuter( FCudaTreeCoordinate::GetPositionFromMorton(outsideInteractions[outInterIdx].insideIndex, treeHeight-1),
                               &particles , &interParticles, &outsideInteractions[outInterIdx].relativeOutPosition, 1);
        }
    }

    for(int leafIdx = blockIdx.x ; leafIdx < counterInnerCell ; leafIdx += gridDim.x){
        for(int outInterIdx = safeInnterInteractions[leafIdx] ; outInterIdx < safeInnterInteractions[leafIdx+1] ; ++outInterIdx){

            ParticleGroupClass interParticles = containersOther.template getLeaf<ParticleGroupClass>(insideInteractions[outInterIdx].outsideIdxInBlock);
            ParticleGroupClass particles = containers.template getLeaf<ParticleGroupClass>(insideInteractions[outInterIdx].insideIdxInBlock);

            FCudaAssertLF(containersOther.getLeafMortonIndex(insideInteractions[outInterIdx].outsideIdxInBlock) == insideInteractions[outInterIdx].outIndex);
            FCudaAssertLF(containers.getLeafMortonIndex(insideInteractions[outInterIdx].insideIdxInBlock) == insideInteractions[outInterIdx].insideIndex);

            const int otherPosition = FMGetOppositeNeighIndex(insideInteractions[outInterIdx].relativeOutPosition);
            kernel->P2POuter( FCudaTreeCoordinate::GetPositionFromMorton(insideInteractions[outInterIdx].outIndex, treeHeight-1),
                               &interParticles , &particles, &otherPosition, 1);
        }
    }
}

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__host__ void FCuda__directInoutPassCallback(unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
                                             unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
                                             const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
                                             const int     safeOuterInteractions[], const int counterOuterCell,
                                                 const OutOfBlockInteraction* insideInteractions,
                                                 const int     safeInnterInteractions[], const int counterInnerCell,
                                             const int treeHeight, CudaKernelClass* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize){
    OutOfBlockInteraction* cuOutsideInteractions;
    FCudaCheck( cudaMalloc(&cuOutsideInteractions,nbOutsideInteractions*sizeof(OutOfBlockInteraction)) );
    FCudaCheck( cudaMemcpy( cuOutsideInteractions, outsideInteractions, nbOutsideInteractions*sizeof(OutOfBlockInteraction),
                cudaMemcpyHostToDevice ) );

    OutOfBlockInteraction* cuInsideInteractions;
    FCudaCheck( cudaMalloc(&cuInsideInteractions,nbOutsideInteractions*sizeof(OutOfBlockInteraction)) );
    FCudaCheck( cudaMemcpy( cuInsideInteractions, insideInteractions, nbOutsideInteractions*sizeof(OutOfBlockInteraction),
                cudaMemcpyHostToDevice ) );

    int* cuSafeOuterInteractions;
    FCudaCheck( cudaMalloc(&cuSafeOuterInteractions,(counterOuterCell+1)*sizeof(int)) );
    FCudaCheck( cudaMemcpy( cuSafeOuterInteractions, safeOuterInteractions, (counterOuterCell+1)*sizeof(int),
                cudaMemcpyHostToDevice ) );

    int* cuSafeInnterInteractions;
    FCudaCheck( cudaMalloc(&cuSafeInnterInteractions,(counterInnerCell+1)*sizeof(int)) );
    FCudaCheck( cudaMemcpy( cuSafeInnterInteractions, safeInnterInteractions, (counterInnerCell+1)*sizeof(int),
                cudaMemcpyHostToDevice ) );

    FCuda__directInoutPassPerform
            <SymboleCellClass, PoleCellClass, LocalCellClass,
            CellContainerClass, ParticleContainerGroupClass, ParticleGroupClass, CudaKernelClass>
            <<<inGridSize, inBlocksSize, 0, currentStream>>>(containersPtr, containersSize,containersDownPtr,
                                  externalContainersPtr, externalContainersSize,externalContainersDownPtr,
                                  cuOutsideInteractions, nbOutsideInteractions,
                                 cuSafeOuterInteractions,counterOuterCell,
                                  cuInsideInteractions,
                                  cuSafeInnterInteractions , counterInnerCell,
                                  treeHeight, kernel);

    FCudaCheckAfterCall();
    FCudaCheck(cudaStreamSynchronize(currentStream));

    FCudaCheck(cudaFree(cuOutsideInteractions));
    FCudaCheck(cudaFree(cuInsideInteractions));
    FCudaCheck(cudaFree(cuSafeOuterInteractions));
    FCudaCheck(cudaFree(cuSafeInnterInteractions));
}


/////////////////////////////////////////////////////////////////////////////////////
/// Merge Pass
/////////////////////////////////////////////////////////////////////////////////////


template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__global__ void FCuda__mergePassPerform(unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
                                        unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
                                        CudaKernelClass* kernel){    
    CellContainerClass leafCells(leafCellsPtr,leafCellsSize, nullptr, leafCellsDownPtr);
    ParticleContainerGroupClass containers(containersPtr,containersSize, containersDownPtr);

    for(int cellIdx = blockIdx.x ; cellIdx < leafCells.getNumberOfCellsInBlock() ; cellIdx += gridDim.x){
        typename CellContainerClass::CompleteCellClass cell = leafCells.getDownCell(cellIdx);
        FCudaAssertLF(cell.symb->mortonIndex == leafCells.getCellMortonIndex(cellIdx));
        ParticleGroupClass particles = containers.template getLeaf<ParticleGroupClass>(cellIdx);
        FCudaAssertLF(leafCells.getCellMortonIndex(cellIdx) == containers.getLeafMortonIndex(cellIdx));
        kernel->L2P(cell, &particles);
    }
}

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
__host__ void FCuda__mergePassCallback(unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
                                       unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
                                       CudaKernelClass* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize){
    FCuda__mergePassPerform
            <SymboleCellClass, PoleCellClass, LocalCellClass,
            CellContainerClass, ParticleContainerGroupClass, ParticleGroupClass, CudaKernelClass>
            <<<inGridSize, inBlocksSize, 0, currentStream>>>(leafCellsPtr, leafCellsSize,leafCellsDownPtr,
                            containersPtr, containersSize,containersDownPtr,
                            kernel);

    FCudaCheckAfterCall();
    FCudaCheck(cudaStreamSynchronize(currentStream));
}


template <class CudaKernelClass>
CudaKernelClass* FCuda__BuildCudaKernel(void* kernel){
    return CudaKernelClass::InitKernelKernel(kernel);
}

template <class CudaKernelClass>
void FCuda__ReleaseCudaKernel(CudaKernelClass* cukernel){
    CudaKernelClass::ReleaseKernel(cukernel);
}

template <class CudaKernelClass>
dim3 FCuda__GetGridSize(CudaKernelClass* /*kernel*/, int intervalSize){
    return CudaKernelClass::GetGridSize(intervalSize);
}

template <class CudaKernelClass>
dim3 FCuda__GetBlockSize(CudaKernelClass* /*kernel*/){
    return CudaKernelClass::GetBlocksSize();
}


/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

#include "FCudaGroupOfCells.hpp"
#include "FCudaGroupAttachedLeaf.hpp"
#include "FCudaGroupOfParticles.hpp"

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

#include "FCudaEmptyKernel.hpp"
#include "FCudaEmptyCellSymb.hpp"

template void FCuda__bottomPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<int,0,0,int>, FCudaGroupAttachedLeaf<int,0,0,int>, FCudaEmptyKernel<int> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
    unsigned char* containersPtr, std::size_t containersSize,
    FCudaEmptyKernel<int>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__upwardPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<int,0,0,int>, FCudaGroupAttachedLeaf<int,0,0,int>, FCudaEmptyKernel<int> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
int idxLevel, FCudaEmptyKernel<int>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__transferInoutPassCallbackMpi<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<int,0,0,int>, FCudaGroupAttachedLeaf<int,0,0,int>, FCudaEmptyKernel<int> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
    int idxLevel, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, FCudaEmptyKernel<int>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__transferInPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<int,0,0,int>, FCudaGroupAttachedLeaf<int,0,0,int>, FCudaEmptyKernel<int> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
    int idxLevel, FCudaEmptyKernel<int>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__transferInoutPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<int,0,0,int>, FCudaGroupAttachedLeaf<int,0,0,int>, FCudaEmptyKernel<int> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
unsigned char* currentCellsDownPtr,
unsigned char* externalCellsPtr, std::size_t externalCellsSize,
unsigned char* externalCellsUpPtr,
int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
int nbOutsideInteractions,
const int* safeInteractions, int nbSafeInteractions, FCudaEmptyKernel<int>* kernel, cudaStream_t currentStream,
                                    const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__downardPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<int,0,0,int>, FCudaGroupAttachedLeaf<int,0,0,int>, FCudaEmptyKernel<int> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
int idxLevel, FCudaEmptyKernel<int>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__directInoutPassCallbackMpi<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<int,0,0,int>, FCudaGroupAttachedLeaf<int,0,0,int>, FCudaEmptyKernel<int> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize,
    const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
const int treeHeight, FCudaEmptyKernel<int>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__directInPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<int,0,0,int>, FCudaGroupAttachedLeaf<int,0,0,int>, FCudaEmptyKernel<int> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    const int treeHeight, FCudaEmptyKernel<int>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__directInoutPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<int,0,0,int>, FCudaGroupAttachedLeaf<int,0,0,int>, FCudaEmptyKernel<int> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
const int     safeOuterInteractions[], const int counterOuterCell,
    const OutOfBlockInteraction* insideInteractions,
    const int     safeInnterInteractions[], const int counterInnerCell,
const int treeHeight, FCudaEmptyKernel<int>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__mergePassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                       FCudaGroupOfParticles<int,0,0,int>, FCudaGroupAttachedLeaf<int,0,0,int>, FCudaEmptyKernel<int> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    FCudaEmptyKernel<int>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template FCudaEmptyKernel<int>* FCuda__BuildCudaKernel< FCudaEmptyKernel<int> >(void* kernel);
template void FCuda__ReleaseCudaKernel< FCudaEmptyKernel<int> >(FCudaEmptyKernel<int>* cukernel);
template dim3 FCuda__GetGridSize< FCudaEmptyKernel<int> >(FCudaEmptyKernel<int>* kernel, int intervalSize);
template dim3 FCuda__GetBlockSize< FCudaEmptyKernel<int> >(FCudaEmptyKernel<int>* cukernel);

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

#include "../TestKernel/FCudaTestKernels.hpp"
#include "../TestKernel/FTestCellPOD.hpp"

template void FCuda__bottomPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<float,0, 1, long long int>, FCudaGroupAttachedLeaf<float,0, 1, long long int>, FTestCudaKernels<float> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
unsigned char* containersPtr, std::size_t containersSize,
    FTestCudaKernels<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__upwardPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<float,0, 1, long long int>, FCudaGroupAttachedLeaf<float,0, 1, long long int>, FTestCudaKernels<float> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
int idxLevel, FTestCudaKernels<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__transferInoutPassCallbackMpi<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<float,0, 1, long long int>, FCudaGroupAttachedLeaf<float,0, 1, long long int>, FTestCudaKernels<float> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
    int idxLevel, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, FTestCudaKernels<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__transferInPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<float,0, 1, long long int>, FCudaGroupAttachedLeaf<float,0, 1, long long int>, FTestCudaKernels<float> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
    int idxLevel, FTestCudaKernels<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__transferInoutPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<float,0, 1, long long int>, FCudaGroupAttachedLeaf<float,0, 1, long long int>, FTestCudaKernels<float> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
unsigned char* currentCellsDownPtr,
unsigned char* externalCellsPtr, std::size_t externalCellsSize,
unsigned char* externalCellsUpPtr,
int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
int nbOutsideInteractions,
const int* safeInteractions, int nbSafeInteractions, FTestCudaKernels<float>* kernel, cudaStream_t currentStream,
                                    const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__downardPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<float,0, 1, long long int>, FCudaGroupAttachedLeaf<float,0, 1, long long int>, FTestCudaKernels<float> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
int idxLevel, FTestCudaKernels<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__directInoutPassCallbackMpi<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<float,0, 1, long long int>, FCudaGroupAttachedLeaf<float,0, 1, long long int>, FTestCudaKernels<float> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize,
    const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
const int treeHeight, FTestCudaKernels<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__directInPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<float,0, 1, long long int>, FCudaGroupAttachedLeaf<float,0, 1, long long int>, FTestCudaKernels<float> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    const int treeHeight, FTestCudaKernels<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__directInoutPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<float,0, 1, long long int>, FCudaGroupAttachedLeaf<float,0, 1, long long int>, FTestCudaKernels<float> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
const int     safeOuterInteractions[], const int counterOuterCell,
    const OutOfBlockInteraction* insideInteractions,
    const int     safeInnterInteractions[], const int counterInnerCell, const int treeHeight, FTestCudaKernels<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__mergePassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<float,0, 1, long long int>, FCudaGroupAttachedLeaf<float,0, 1, long long int>, FTestCudaKernels<float> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    FTestCudaKernels<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template FTestCudaKernels<float>* FCuda__BuildCudaKernel<FTestCudaKernels<float>>(void* kernel);
template void FCuda__ReleaseCudaKernel<FTestCudaKernels<float>>(FTestCudaKernels<float>* cukernel);

template dim3 FCuda__GetGridSize< FTestCudaKernels<float> >(FTestCudaKernels<float>* kernel, int intervalSize);
template dim3 FCuda__GetBlockSize< FTestCudaKernels<float> >(FTestCudaKernels<float>* cukernel);




template void FCuda__bottomPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<double,0, 1, long long int>, FCudaGroupAttachedLeaf<double,0, 1, long long int>, FTestCudaKernels<double> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
unsigned char* containersPtr, std::size_t containersSize,
    FTestCudaKernels<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__upwardPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<double,0, 1, long long int>, FCudaGroupAttachedLeaf<double,0, 1, long long int>, FTestCudaKernels<double> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
int idxLevel, FTestCudaKernels<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__transferInoutPassCallbackMpi<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<double,0, 1, long long int>, FCudaGroupAttachedLeaf<double,0, 1, long long int>, FTestCudaKernels<double> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
    int idxLevel, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, FTestCudaKernels<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__transferInPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<double,0, 1, long long int>, FCudaGroupAttachedLeaf<double,0, 1, long long int>, FTestCudaKernels<double> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
    int idxLevel, FTestCudaKernels<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__transferInoutPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<double,0, 1, long long int>, FCudaGroupAttachedLeaf<double,0, 1, long long int>, FTestCudaKernels<double> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
unsigned char* currentCellsDownPtr,
unsigned char* externalCellsPtr, std::size_t externalCellsSize,
unsigned char* externalCellsUpPtr,
int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
int nbOutsideInteractions,
const int* safeInteractions, int nbSafeInteractions, FTestCudaKernels<double>* kernel, cudaStream_t currentStream,
                                    const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__downardPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<double,0, 1, long long int>, FCudaGroupAttachedLeaf<double,0, 1, long long int>, FTestCudaKernels<double> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
    int idxLevel, FTestCudaKernels<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__directInoutPassCallbackMpi<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<double,0, 1, long long int>, FCudaGroupAttachedLeaf<double,0, 1, long long int>, FTestCudaKernels<double> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize,
    const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
const int treeHeight, FTestCudaKernels<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__directInPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<double,0, 1, long long int>, FCudaGroupAttachedLeaf<double,0, 1, long long int>, FTestCudaKernels<double> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    const int treeHeight, FTestCudaKernels<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__directInoutPassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<double,0, 1, long long int>, FCudaGroupAttachedLeaf<double,0, 1, long long int>, FTestCudaKernels<double> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
const int     safeOuterInteractions[], const int counterOuterCell,
    const OutOfBlockInteraction* insideInteractions,
    const int     safeInnterInteractions[], const int counterInnerCell, const int treeHeight, FTestCudaKernels<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__mergePassCallback<FTestCellPODCore, FTestCellPODData, FTestCellPODData, FCudaGroupOfCells<FTestCellPODCore, FTestCellPODData, FTestCellPODData>,
                                        FCudaGroupOfParticles<double,0, 1, long long int>, FCudaGroupAttachedLeaf<double,0, 1, long long int>, FTestCudaKernels<double> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    FTestCudaKernels<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template FTestCudaKernels<double>* FCuda__BuildCudaKernel<FTestCudaKernels<double>>(void* kernel);
template void FCuda__ReleaseCudaKernel<FTestCudaKernels<double>>(FTestCudaKernels<double>* cukernel);

template dim3 FCuda__GetGridSize< FTestCudaKernels<double> >(FTestCudaKernels<double>* kernel, int intervalSize);
template dim3 FCuda__GetBlockSize< FTestCudaKernels<double> >(FTestCudaKernels<double>* cukernel);


/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

#include "../P2P/FCudaP2P.hpp"

template void FCuda__bottomPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FCudaP2P<float> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
unsigned char* containersPtr, std::size_t containersSize,
    FCudaP2P<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__upwardPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FCudaP2P<float> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
    int idxLevel, FCudaP2P<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__transferInoutPassCallbackMpi<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FCudaP2P<float> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
    int idxLevel, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, FCudaP2P<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__transferInPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FCudaP2P<float> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
    int idxLevel, FCudaP2P<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__transferInoutPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FCudaP2P<float> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
unsigned char* currentCellsDownPtr,
unsigned char* externalCellsPtr, std::size_t externalCellsSize,
unsigned char* externalCellsUpPtr,
int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
int nbOutsideInteractions,
const int* safeInteractions, int nbSafeInteractions, FCudaP2P<float>* kernel, cudaStream_t currentStream,
                                    const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__downardPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FCudaP2P<float> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
int idxLevel, FCudaP2P<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__directInoutPassCallbackMpi<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FCudaP2P<float> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize,
    const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
const int treeHeight, FCudaP2P<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__directInPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FCudaP2P<float> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    const int treeHeight, FCudaP2P<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__directInoutPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FCudaP2P<float> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
const int     safeOuterInteractions[], const int counterOuterCell,
    const OutOfBlockInteraction* insideInteractions,
    const int     safeInnterInteractions[], const int counterInnerCell, const int treeHeight, FCudaP2P<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__mergePassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FCudaP2P<float> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    FCudaP2P<float>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template FCudaP2P<float>* FCuda__BuildCudaKernel<FCudaP2P<float>>(void* kernel);
template void FCuda__ReleaseCudaKernel<FCudaP2P<float>>(FCudaP2P<float>* cukernel);

template dim3 FCuda__GetGridSize< FCudaP2P<float> >(FCudaP2P<float>* kernel, int intervalSize);
template dim3 FCuda__GetBlockSize< FCudaP2P<float> >(FCudaP2P<float>* cukernel);




template void FCuda__bottomPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FCudaP2P<double> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
unsigned char* containersPtr, std::size_t containersSize,
    FCudaP2P<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__upwardPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FCudaP2P<double> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
int idxLevel, FCudaP2P<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__transferInoutPassCallbackMpi<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FCudaP2P<double> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
    int idxLevel, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, FCudaP2P<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__transferInPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FCudaP2P<double> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
    int idxLevel, FCudaP2P<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__transferInoutPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FCudaP2P<double> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
unsigned char* currentCellsDownPtr,
unsigned char* externalCellsPtr, std::size_t externalCellsSize,
unsigned char* externalCellsUpPtr,
int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
int nbOutsideInteractions,
const int* safeInteractions, int nbSafeInteractions, FCudaP2P<double>* kernel, cudaStream_t currentStream,
                                    const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__downardPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FCudaP2P<double> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
int idxLevel, FCudaP2P<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__directInoutPassCallbackMpi<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FCudaP2P<double> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize,
    const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
const int treeHeight, FCudaP2P<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__directInPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FCudaP2P<double> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    const int treeHeight, FCudaP2P<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__directInoutPassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FCudaP2P<double> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
const int     safeOuterInteractions[], const int counterOuterCell,
    const OutOfBlockInteraction* insideInteractions,
    const int     safeInnterInteractions[], const int counterInnerCell, const int treeHeight, FCudaP2P<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__mergePassCallback<FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FCudaP2P<double> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    FCudaP2P<double>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template FCudaP2P<double>* FCuda__BuildCudaKernel<FCudaP2P<double>>(void* kernel);
template void FCuda__ReleaseCudaKernel<FCudaP2P<double>>(FCudaP2P<double>* cukernel);

template dim3 FCuda__GetGridSize< FCudaP2P<double> >(FCudaP2P<double>* kernel, int intervalSize);
template dim3 FCuda__GetBlockSize< FCudaP2P<double> >(FCudaP2P<double>* cukernel);



/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

#include "../Uniform/FUnifCuda.hpp"

template void FCuda__bottomPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,5> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
unsigned char* containersPtr, std::size_t containersSize,
    FUnifCuda<float,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__upwardPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,5> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
    int idxLevel, FUnifCuda<float,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__transferInoutPassCallbackMpi<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,5> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
    int idxLevel, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, FUnifCuda<float,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__transferInPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,5> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
    int idxLevel, FUnifCuda<float,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__transferInoutPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,5> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
unsigned char* currentCellsDownPtr,
unsigned char* externalCellsPtr, std::size_t externalCellsSize,
unsigned char* externalCellsUpPtr,
int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
int nbOutsideInteractions,
const int* safeInteractions, int nbSafeInteractions, FUnifCuda<float,5>* kernel, cudaStream_t currentStream,
                                    const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__downardPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,5> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
int idxLevel, FUnifCuda<float,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__directInoutPassCallbackMpi<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,5> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize,
    const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
const int treeHeight, FUnifCuda<float,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__directInPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,5> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    const int treeHeight, FUnifCuda<float,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__directInoutPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,5> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
const int     safeOuterInteractions[], const int counterOuterCell,
    const OutOfBlockInteraction* insideInteractions,
    const int     safeInnterInteractions[], const int counterInnerCell, const int treeHeight, FUnifCuda<float,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__mergePassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,5>,FCudaUnifCellPODLocal<float,5>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,5> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    FUnifCuda<float,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template FUnifCuda<float,5>* FCuda__BuildCudaKernel<FUnifCuda<float,5>>(void* kernel);
template void FCuda__ReleaseCudaKernel<FUnifCuda<float,5>>(FUnifCuda<float,5>* cukernel);

template dim3 FCuda__GetGridSize< FUnifCuda<float,5> >(FUnifCuda<float,5>* kernel, int intervalSize);
template dim3 FCuda__GetBlockSize< FUnifCuda<float,5> >(FUnifCuda<float,5>* cukernel);

template void FUnifCudaFillObject(void* cudaKernel, const FUnifCudaSharedData<double,5>& hostData);



template void FCuda__bottomPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,5> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
unsigned char* containersPtr, std::size_t containersSize,
    FUnifCuda<double,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__upwardPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,5> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
int idxLevel, FUnifCuda<double,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__transferInoutPassCallbackMpi<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,5> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
    int idxLevel, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, FUnifCuda<double,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__transferInPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,5> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
    int idxLevel, FUnifCuda<double,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__transferInoutPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,5> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
unsigned char* currentCellsDownPtr,
unsigned char* externalCellsPtr, std::size_t externalCellsSize,
unsigned char* externalCellsUpPtr,
int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
int nbOutsideInteractions,
const int* safeInteractions, int nbSafeInteractions, FUnifCuda<double,5>* kernel, cudaStream_t currentStream,
                                    const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__downardPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,5> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
int idxLevel, FUnifCuda<double,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__directInoutPassCallbackMpi<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,5> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize,
    const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
const int treeHeight, FUnifCuda<double,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__directInPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,5> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    const int treeHeight, FUnifCuda<double,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__directInoutPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,5> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
const int     safeOuterInteractions[], const int counterOuterCell,
    const OutOfBlockInteraction* insideInteractions,
    const int     safeInnterInteractions[], const int counterInnerCell, const int treeHeight, FUnifCuda<double,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__mergePassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,5>,FCudaUnifCellPODLocal<double,5>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,5> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    FUnifCuda<double,5>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template FUnifCuda<double,5>* FCuda__BuildCudaKernel<FUnifCuda<double,5>>(void* kernel);
template void FCuda__ReleaseCudaKernel<FUnifCuda<double,5>>(FUnifCuda<double,5>* cukernel);

template dim3 FCuda__GetGridSize< FUnifCuda<double,5> >(FUnifCuda<double,5>* kernel, int intervalSize);
template dim3 FCuda__GetBlockSize< FUnifCuda<double,5> >(FUnifCuda<double,5>* cukernel);

template void FUnifCudaFillObject(void* cudaKernel, const FUnifCudaSharedData<float,5>& hostData);




template void FCuda__bottomPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,7> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
unsigned char* containersPtr, std::size_t containersSize,
    FUnifCuda<float,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__upwardPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,7> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
    int idxLevel, FUnifCuda<float,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__transferInoutPassCallbackMpi<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,7> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
    int idxLevel, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, FUnifCuda<float,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__transferInPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,7> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
    int idxLevel, FUnifCuda<float,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__transferInoutPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,7> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
unsigned char* currentCellsDownPtr,
unsigned char* externalCellsPtr, std::size_t externalCellsSize,
unsigned char* externalCellsUpPtr,
int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
int nbOutsideInteractions,
const int* safeInteractions, int nbSafeInteractions, FUnifCuda<float,7>* kernel, cudaStream_t currentStream,
                                    const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__downardPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,7> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
int idxLevel, FUnifCuda<float,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__directInoutPassCallbackMpi<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,7> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize,
    const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
const int treeHeight, FUnifCuda<float,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__directInPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,7> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    const int treeHeight, FUnifCuda<float,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__directInoutPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,7> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
const int     safeOuterInteractions[], const int counterOuterCell,
    const OutOfBlockInteraction* insideInteractions,
    const int     safeInnterInteractions[], const int counterInnerCell, const int treeHeight, FUnifCuda<float,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__mergePassCallback<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<float,7>,FCudaUnifCellPODLocal<float,7>>,
                                        FCudaGroupOfParticles<float,1, 4, float>, FCudaGroupAttachedLeaf<float,1, 4, float>, FUnifCuda<float,7> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    FUnifCuda<float,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template FUnifCuda<float,7>* FCuda__BuildCudaKernel<FUnifCuda<float,7>>(void* kernel);
template void FCuda__ReleaseCudaKernel<FUnifCuda<float,7>>(FUnifCuda<float,7>* cukernel);

template dim3 FCuda__GetGridSize< FUnifCuda<float,7> >(FUnifCuda<float,7>* kernel, int intervalSize);
template dim3 FCuda__GetBlockSize< FUnifCuda<float,7> >(FUnifCuda<float,7>* cukernel);

template void FUnifCudaFillObject(void* cudaKernel, const FUnifCudaSharedData<double,7>& hostData);



template void FCuda__bottomPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,7> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
unsigned char* containersPtr, std::size_t containersSize,
    FUnifCuda<double,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__upwardPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,7> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
int idxLevel, FUnifCuda<double,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__transferInoutPassCallbackMpi<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,7> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
    int idxLevel, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, FUnifCuda<double,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__transferInPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,7> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
    int idxLevel, FUnifCuda<double,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__transferInoutPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,7> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize,
unsigned char* currentCellsDownPtr,
unsigned char* externalCellsPtr, std::size_t externalCellsSize,
unsigned char* externalCellsUpPtr,
int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
int nbOutsideInteractions,
const int* safeInteractions, int nbSafeInteractions, FUnifCuda<double,7>* kernel, cudaStream_t currentStream,
                                    const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__downardPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,7> >
    (unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
int idxLevel, FUnifCuda<double,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template void FCuda__directInoutPassCallbackMpi<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,7> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize,
    const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
const int treeHeight, FUnifCuda<double,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template void FCuda__directInPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,7> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    const int treeHeight, FUnifCuda<double,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__directInoutPassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,7> >
    (unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
const int     safeOuterInteractions[], const int counterOuterCell,
    const OutOfBlockInteraction* insideInteractions,
    const int     safeInnterInteractions[], const int counterInnerCell, const int treeHeight, FUnifCuda<double,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template void FCuda__mergePassCallback<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>, FCudaGroupOfCells<FBasicCellPOD, FCudaUnifCellPODPole<double,7>,FCudaUnifCellPODLocal<double,7>>,
                                        FCudaGroupOfParticles<double,1, 4, double>, FCudaGroupAttachedLeaf<double,1, 4, double>, FUnifCuda<double,7> >
    (unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    FUnifCuda<double,7>* kernel, cudaStream_t currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template FUnifCuda<double,7>* FCuda__BuildCudaKernel<FUnifCuda<double,7>>(void* kernel);
template void FCuda__ReleaseCudaKernel<FUnifCuda<double,7>>(FUnifCuda<double,7>* cukernel);

template dim3 FCuda__GetGridSize< FUnifCuda<double,7> >(FUnifCuda<double,7>* kernel, int intervalSize);
template dim3 FCuda__GetBlockSize< FUnifCuda<double,7> >(FUnifCuda<double,7>* cukernel);

template void FUnifCudaFillObject(void* cudaKernel, const FUnifCudaSharedData<float,7>& hostData);
