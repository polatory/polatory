#ifndef FSTARPUCPTINTERACTIONSWRAPPER_HPP
#define FSTARPUCPTINTERACTIONSWRAPPER_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Core/FCoreCommon.hpp"
#include "../../Utils/FQuickSort.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../../Utils/FLog.hpp"
#include "../../Utils/FTic.hpp"
#include "../../Utils/FAssert.hpp"

#include "../Core/FOutOfBlockInteraction.hpp"

#ifdef SCALFMM_USE_MPI
#include "../../Utils/FMpi.hpp"
#endif

#include <vector>
#include <memory>

#include <omp.h>

//extern "C"{
#include <starpu.h>
//}

#if defined(STARPU_USE_MPI) && defined(SCALFMM_USE_MPI)
//extern "C"{
#include <starpu_mpi.h>
//}
#endif

#include "FStarPUUtils.hpp"

#include "../../Utils/FTaskTimer.hpp"

template <class CellContainerClass, class CellClass, class KernelClass,
          class ParticleGroupClass, class ParticleContainerClass>
class FStarPUCptInteractionsWrapper {
public:
    enum InterTypeEnum {
        INTER_P2M = 0,
        INTER_M2M,
        INTER_M2L,
        INTER_M2L_EXT,
        INTER_M2L_MPI,
        INTER_L2L,
        INTER_L2P,
        INTER_P2P,
        INTER_P2P_EXT,
        INTER_P2P_MPI,
        INTER_NB
    };

protected:
    typedef FStarPUCptInteractionsWrapper<CellContainerClass, CellClass, KernelClass, ParticleGroupClass, ParticleContainerClass> ThisClass;

    const int treeHeight;
    KernelClass* kernels[STARPU_MAXCPUS];        //< The kernels

    struct Stats{
        Stats()
            : nbCalls(0), nbInteractionsTotal(0), timeTotal(0), bestSpeed(std::numeric_limits<double>::max()), interactionBestSpeed(0){
        }

        void addExecution(const FSize inNbInteractions, const double inExecTime){
            nbCalls += 1;
            nbInteractionsTotal += inNbInteractions;
            timeTotal += inExecTime;
            if(double(inNbInteractions)/inExecTime < bestSpeed){
                bestSpeed = double(inNbInteractions)/inExecTime;
                interactionBestSpeed = inNbInteractions;
            }
        }

        FSize nbCalls;
        FSize nbInteractionsTotal;
        double timeTotal;
        double bestSpeed;
        FSize interactionBestSpeed;
    };

    Stats stats[STARPU_MAXCPUS][INTER_NB];
    bool computeForReal;

    const int GetWorkerId() {
        return FMath::Max(0, starpu_worker_get_id());
    }

public:
#ifdef SCALFMM_TIME_OMPTASKS
    FTaskTimer taskTimeRecorder;
#endif

    FSize getNbInteractions(const int workerId, const InterTypeEnum operation) const {
        return stats[workerId][operation].nbInteractionsTotal;
    }

    FStarPUCptInteractionsWrapper(const int inTreeHeight)
        : treeHeight(inTreeHeight), computeForReal(false)
#ifdef SCALFMM_TIME_OMPTASKS
      , taskTimeRecorder(STARPU_MAXCPUS)
#endif
    {
        memset(kernels, 0, sizeof(KernelClass*)*STARPU_MAXCPUS);
    }

    bool compute() const{
        return computeForReal;
    }

    void compute(const bool inComputeForReal){
        computeForReal = inComputeForReal;
    }

    KernelClass* getKernel(const int workerId){
        return kernels[workerId];
    }

    const KernelClass* getKernel(const int workerId) const {
        return kernels[workerId];
    }

    void initKernel(const int workerId, KernelClass* originalKernel){
        FAssertLF(kernels[workerId] == nullptr);
        kernels[workerId] = new KernelClass(*originalKernel);
#ifdef SCALFMM_TIME_OMPTASKS
        taskTimeRecorder.init(GetWorkerId());
#endif
    }

    void releaseKernel(const int workerId){
        delete kernels[workerId];
        kernels[workerId] = nullptr;
    }

    ~FStarPUCptInteractionsWrapper(){
        for(int idxKernel = 0 ; idxKernel < STARPU_MAXCPUS ; ++idxKernel ){
            FAssertLF(kernels[idxKernel] == nullptr);
        }
    }

    static void bottomPassCallback(void *buffers[], void *cl_arg){
        CellContainerClass leafCells((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                            STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                            (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                            nullptr);
        ParticleGroupClass containers((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                            STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                            nullptr);

        FStarPUPtrInterface* worker = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &intervalSize);
        worker->get<ThisClass>(FSTARPU_CPU_IDX)->bottomPassPerform(&leafCells, &containers);
    }

    void bottomPassPerform(CellContainerClass* leafCells, ParticleGroupClass* containers){
        FTIME_TASKS(FTaskTimer::ScopeEvent taskTime(GetWorkerId(), &taskTimeRecorder, leafCells->getStartingIndex() * 20 * 8, "P2M"));
        FAssertLF(leafCells->getNumberOfCellsInBlock() == containers->getNumberOfLeavesInBlock());
        KernelClass*const kernel = kernels[GetWorkerId()];
        Stats& currentStat = stats[GetWorkerId()][INTER_P2M];
        FTic timer;
        FSize nbInteractions = 0;

        for(int leafIdx = 0 ; leafIdx < leafCells->getNumberOfCellsInBlock() ; ++leafIdx){
            CellClass cell = leafCells->getUpCell(leafIdx);
            ParticleContainerClass particles = containers->template getLeaf<ParticleContainerClass>(leafIdx);
            FAssertLF(leafCells->getCellMortonIndex(leafIdx) == containers->getLeafMortonIndex(leafIdx));
            if(computeForReal) kernel->P2M(&cell, &particles);
            nbInteractions += particles.getNbParticles();
        }

        currentStat.addExecution(nbInteractions, timer.tacAndElapsed());
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Upward Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void upwardPassCallback(void *buffers[], void *cl_arg){
        CellContainerClass currentCells((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                                        STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                                        nullptr);

        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &intervalSize);

        CellContainerClass subCellGroup(
                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                        STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]),
                        nullptr);

        worker->get<ThisClass>(FSTARPU_CPU_IDX)->upwardPassPerform(&currentCells, &subCellGroup, idxLevel);
    }

    void upwardPassPerform(CellContainerClass*const currentCells,
                           CellContainerClass* subCellGroup,
                           const int idxLevel){
        KernelClass*const kernel = kernels[GetWorkerId()];
        Stats& currentStat = stats[GetWorkerId()][INTER_M2M];
        FTic timer;
        FSize nbInteractions = 0;

        const MortonIndex firstParent = FMath::Max(currentCells->getStartingIndex(), subCellGroup->getStartingIndex()>>3);
        const MortonIndex lastParent = FMath::Min(currentCells->getEndingIndex()-1, (subCellGroup->getEndingIndex()-1)>>3);
        FTIME_TASKS(FTaskTimer::ScopeEvent taskTime(GetWorkerId(), &taskTimeRecorder, ((lastParent * 20) + idxLevel) * 8 + 1, "M2M"));

        int idxParentCell = currentCells->getCellIndex(firstParent);
        FAssertLF(idxParentCell != -1);

        int idxChildCell = subCellGroup->getFistChildIdx(firstParent);
        FAssertLF(idxChildCell != -1);
        CellClass childData[8];

        while(true){
            CellClass cell = currentCells->getUpCell(idxParentCell);
            FAssertLF(cell.getMortonIndex() == currentCells->getCellMortonIndex(idxParentCell));
            const CellClass* child[8] = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};

            FAssertLF(cell.getMortonIndex() == (subCellGroup->getCellMortonIndex(idxChildCell)>>3));

            do{
                const int idxChild = ((subCellGroup->getCellMortonIndex(idxChildCell)) & 7);
                FAssertLF(child[idxChild] == nullptr);
                childData[idxChild] = subCellGroup->getUpCell(idxChildCell);
                FAssertLF(subCellGroup->getCellMortonIndex(idxChildCell) == childData[idxChild].getMortonIndex());
                child[idxChild] = &childData[idxChild];

                idxChildCell += 1;
                nbInteractions += 1;
            }while(idxChildCell != subCellGroup->getNumberOfCellsInBlock() && cell.getMortonIndex() == (subCellGroup->getCellMortonIndex(idxChildCell)>>3));

            if(computeForReal) kernel->M2M(&cell, child, idxLevel);

            if(currentCells->getCellMortonIndex(idxParentCell) == lastParent){
                break;
            }

            idxParentCell += 1;
        }

        currentStat.addExecution(nbInteractions, timer.tacAndElapsed());
    }


    /////////////////////////////////////////////////////////////////////////////////////
    /// Transfer Pass Mpi
    /////////////////////////////////////////////////////////////////////////////////////
#if defined(STARPU_USE_MPI) && defined(SCALFMM_USE_MPI)
    static void transferInoutPassCallbackMpi(void *buffers[], void *cl_arg){
        CellContainerClass currentCells((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                                        STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                                        nullptr,
                                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]));
        CellContainerClass externalCells((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                                        STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]),
                                        nullptr);

        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        const std::vector<OutOfBlockInteraction>* outsideInteractions;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &outsideInteractions, &intervalSize);

        worker->get<ThisClass>(FSTARPU_CPU_IDX)->transferInoutPassPerformMpi(&currentCells, &externalCells, idxLevel, outsideInteractions);
    }


    void transferInoutPassPerformMpi(CellContainerClass*const currentCells,
                                  CellContainerClass*const cellsOther,
                                  const int idxLevel,
                                  const std::vector<OutOfBlockInteraction>* outsideInteractions){
        KernelClass*const kernel = kernels[GetWorkerId()];
        Stats& currentStat = stats[GetWorkerId()][INTER_M2L_MPI];
        FTic timer;
        FSize nbInteractions = 0;

        for(int outInterIdx = 0 ; outInterIdx < int(outsideInteractions->size()) ; ++outInterIdx){
            const int cellPos = cellsOther->getCellIndex((*outsideInteractions)[outInterIdx].outIndex);
            if(cellPos != -1){
                CellClass interCell = cellsOther->getUpCell(cellPos);
                FAssertLF(interCell.getMortonIndex() == (*outsideInteractions)[outInterIdx].outIndex);
                CellClass cell = currentCells->getDownCell((*outsideInteractions)[outInterIdx].insideIdxInBlock);
                FAssertLF(cell.getMortonIndex() == (*outsideInteractions)[outInterIdx].insideIndex);
                const CellClass* ptCell = &interCell;
                if(computeForReal) kernel->M2L( &cell , &ptCell, &(*outsideInteractions)[outInterIdx].relativeOutPosition, 1, idxLevel);
                nbInteractions += 1;
            }
        }

        currentStat.addExecution(nbInteractions, timer.tacAndElapsed());
    }
#endif
    /////////////////////////////////////////////////////////////////////////////////////
    /// Transfer Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void transferInPassCallback(void *buffers[], void *cl_arg){
        CellContainerClass currentCells((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                                        STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]));

        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &intervalSize);

        worker->get<ThisClass>(FSTARPU_CPU_IDX)->transferInPassPerform(&currentCells, idxLevel);
    }

    void transferInPassPerform(CellContainerClass*const currentCells, const int idxLevel){
        FTIME_TASKS(FTaskTimer::ScopeEvent taskTime(GetWorkerId(), &taskTimeRecorder, ((currentCells->getStartingIndex() *20) + idxLevel ) * 8 + 2, "M2L"));
        const MortonIndex blockStartIdx = currentCells->getStartingIndex();
        const MortonIndex blockEndIdx = currentCells->getEndingIndex();
        KernelClass*const kernel = kernels[GetWorkerId()];
        Stats& currentStat = stats[GetWorkerId()][INTER_M2L];
        FTic timer;
        FSize nbInteractions = 0;
        const CellClass* interactions[189];
        CellClass interactionsData[189];

        for(int cellIdx = 0 ; cellIdx < currentCells->getNumberOfCellsInBlock() ; ++cellIdx){
            CellClass cell = currentCells->getDownCell(cellIdx);

            FAssertLF(cell.getMortonIndex() == currentCells->getCellMortonIndex(cellIdx));

            MortonIndex interactionsIndexes[189];
            int interactionsPosition[189];
            const FTreeCoordinate coord(cell.getCoordinate());
            int counter = coord.getInteractionNeighbors(idxLevel,interactionsIndexes,interactionsPosition);

            int counterExistingCell = 0;

            for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
                if( blockStartIdx <= interactionsIndexes[idxInter] && interactionsIndexes[idxInter] < blockEndIdx ){
                    const int cellPos = currentCells->getCellIndex(interactionsIndexes[idxInter]);
                    if(cellPos != -1){
                        CellClass interCell = currentCells->getUpCell(cellPos);
                        FAssertLF(interCell.getMortonIndex() == interactionsIndexes[idxInter]);
                        interactionsPosition[counterExistingCell] = interactionsPosition[idxInter];
                        interactionsData[counterExistingCell] = interCell;
                        interactions[counterExistingCell] = &interactionsData[counterExistingCell];
                        counterExistingCell += 1;
                    }
                }
            }

            if(computeForReal) kernel->M2L( &cell , interactions, interactionsPosition, counterExistingCell, idxLevel);
            nbInteractions += counterExistingCell;
        }

        currentStat.addExecution(nbInteractions, timer.tacAndElapsed());
    }

    static void transferInoutPassCallback(void *buffers[], void *cl_arg){
        CellContainerClass currentCells((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                                        STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                                        nullptr,
                                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]));
        CellContainerClass externalCells((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                                        STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]),
                                        nullptr);

        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        const std::vector<OutOfBlockInteraction>* outsideInteractions;
        int intervalSize;
        int mode = 0;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &outsideInteractions, &intervalSize, &mode);

        worker->get<ThisClass>(FSTARPU_CPU_IDX)->transferInoutPassPerform(&currentCells, &externalCells, idxLevel, outsideInteractions, mode);
    }


    void transferInoutPassPerform(CellContainerClass*const currentCells,
                                  CellContainerClass*const cellsOther,
                                  const int idxLevel,
                                  const std::vector<OutOfBlockInteraction>* outsideInteractions,
                                  const int mode){
        KernelClass*const kernel = kernels[GetWorkerId()];
        Stats& currentStat = stats[GetWorkerId()][INTER_M2L_EXT];
        FTic timer;
        FSize nbInteractions = int(outsideInteractions->size());

        if(mode == 1){
            FTIME_TASKS(FTaskTimer::ScopeEvent taskTime(GetWorkerId(), &taskTimeRecorder, (((currentCells->getStartingIndex()+1) * (cellsOther->getStartingIndex()+2)) * 20 + idxLevel) * 8 + 3, "M2L-ext"));
            for(int outInterIdx = 0 ; outInterIdx < int(outsideInteractions->size()) ; ++outInterIdx){
                CellClass interCell = cellsOther->getUpCell((*outsideInteractions)[outInterIdx].outsideIdxInBlock);
                FAssertLF(interCell.getMortonIndex() == (*outsideInteractions)[outInterIdx].outIndex);
                CellClass cell = currentCells->getDownCell((*outsideInteractions)[outInterIdx].insideIdxInBlock);
                FAssertLF(cell.getMortonIndex() == (*outsideInteractions)[outInterIdx].insideIndex);

                const CellClass* ptCell = &interCell;
                if(computeForReal) kernel->M2L( &cell , &ptCell, &(*outsideInteractions)[outInterIdx].relativeOutPosition, 1, idxLevel);
            }
        }
        else{
            FTIME_TASKS(FTaskTimer::ScopeEvent taskTime(GetWorkerId(), &taskTimeRecorder, (((currentCells->getStartingIndex()+1) * (cellsOther->getStartingIndex()+1)) * 20 + idxLevel) * 8 + 3, "M2L-ext"));
            for(int outInterIdx = 0 ; outInterIdx < int(outsideInteractions->size()) ; ++outInterIdx){
                CellClass cell = cellsOther->getUpCell((*outsideInteractions)[outInterIdx].insideIdxInBlock);
                FAssertLF(cell.getMortonIndex() == (*outsideInteractions)[outInterIdx].insideIndex);
                CellClass interCell = currentCells->getDownCell((*outsideInteractions)[outInterIdx].outsideIdxInBlock);
                FAssertLF(interCell.getMortonIndex() == (*outsideInteractions)[outInterIdx].outIndex);

                const int otherPos = getOppositeInterIndex((*outsideInteractions)[outInterIdx].relativeOutPosition);
                const CellClass* ptCell = &cell;
                if(computeForReal) kernel->M2L( &interCell , &ptCell, &otherPos, 1, idxLevel);
            }
        }

        currentStat.addExecution(nbInteractions, timer.tacAndElapsed());
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Downard Pass
    /////////////////////////////////////////////////////////////////////////////////////
    static void downardPassCallback(void *buffers[], void *cl_arg){
        CellContainerClass currentCells((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                                        STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                                        nullptr,
                                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]));

        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &intervalSize);

        CellContainerClass subCellGroup(
                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                        STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                        nullptr,
                        (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]));

        worker->get<ThisClass>(FSTARPU_CPU_IDX)->downardPassPerform(&currentCells, &subCellGroup, idxLevel);
    }

    void downardPassPerform(CellContainerClass*const currentCells,
                            CellContainerClass* subCellGroup,
                            const int idxLevel){
        KernelClass*const kernel = kernels[GetWorkerId()];
        Stats& currentStat = stats[GetWorkerId()][INTER_L2L];
        FTic timer;
        FSize nbInteractions = 0;

        const MortonIndex firstParent = FMath::Max(currentCells->getStartingIndex(), subCellGroup->getStartingIndex()>>3);
        const MortonIndex lastParent = FMath::Min(currentCells->getEndingIndex()-1, (subCellGroup->getEndingIndex()-1)>>3);
        FTIME_TASKS(FTaskTimer::ScopeEvent taskTime(GetWorkerId(), &taskTimeRecorder, ((lastParent * 20) + idxLevel) * 8 + 4, "L2L"));

        int idxParentCell = currentCells->getCellIndex(firstParent);
        FAssertLF(idxParentCell != -1);

        int idxChildCell = subCellGroup->getFistChildIdx(firstParent);
        FAssertLF(idxChildCell != -1);
        CellClass childData[8];

        while(true){
            CellClass cell = currentCells->getDownCell(idxParentCell);
            FAssertLF(cell.getMortonIndex() == currentCells->getCellMortonIndex(idxParentCell));
            CellClass* child[8] = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};

            FAssertLF(cell.getMortonIndex() == (subCellGroup->getCellMortonIndex(idxChildCell)>>3));

            do{
                const int idxChild = ((subCellGroup->getCellMortonIndex(idxChildCell)) & 7);
                FAssertLF(child[idxChild] == nullptr);
                childData[idxChild] = subCellGroup->getDownCell(idxChildCell);
                FAssertLF(subCellGroup->getCellMortonIndex(idxChildCell) == childData[idxChild].getMortonIndex());
                child[idxChild] = &childData[idxChild];

                idxChildCell += 1;
                nbInteractions += 1;
            }while(idxChildCell != subCellGroup->getNumberOfCellsInBlock() && cell.getMortonIndex() == (subCellGroup->getCellMortonIndex(idxChildCell)>>3));

            if(computeForReal) kernel->L2L(&cell, child, idxLevel);

            if(currentCells->getCellMortonIndex(idxParentCell) == lastParent){
                break;
            }

            idxParentCell += 1;
        }

        currentStat.addExecution(nbInteractions, timer.tacAndElapsed());
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Direct Pass MPI
    /////////////////////////////////////////////////////////////////////////////////////

#if defined(STARPU_USE_MPI) && defined(SCALFMM_USE_MPI)
    static void directInoutPassCallbackMpi(void *buffers[], void *cl_arg){
        ParticleGroupClass containers((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                                      STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                                      (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]));
        ParticleGroupClass externalContainers((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                                      STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                                      nullptr);

        FStarPUPtrInterface* worker = nullptr;
        const std::vector<OutOfBlockInteraction>* outsideInteractions = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &outsideInteractions, &intervalSize);

        worker->get<ThisClass>(FSTARPU_CPU_IDX)->directInoutPassPerformMpi(&containers, &externalContainers, outsideInteractions);
    }

    void directInoutPassPerformMpi(ParticleGroupClass* containers, ParticleGroupClass* containersOther,
                                const std::vector<OutOfBlockInteraction>* outsideInteractions){
        KernelClass*const kernel = kernels[GetWorkerId()];
        Stats& currentStat = stats[GetWorkerId()][INTER_P2P_MPI];
        FTic timer;
        FSize nbInteractions = 0;
        for(int outInterIdx = 0 ; outInterIdx < int(outsideInteractions->size()) ; ++outInterIdx){
            const int leafPos = containersOther->getLeafIndex((*outsideInteractions)[outInterIdx].outIndex);
            if(leafPos != -1){
                ParticleContainerClass interParticles = containersOther->template getLeaf<ParticleContainerClass>(leafPos);
                FAssertLF(containersOther->getLeafMortonIndex(leafPos) == (*outsideInteractions)[outInterIdx].outIndex);
                ParticleContainerClass particles = containers->template getLeaf<ParticleContainerClass>((*outsideInteractions)[outInterIdx].insideIdxInBlock);
                FAssertLF(containers->getLeafMortonIndex((*outsideInteractions)[outInterIdx].insideIdxInBlock) == (*outsideInteractions)[outInterIdx].insideIndex,
                        containers->getLeafMortonIndex((*outsideInteractions)[outInterIdx].insideIdxInBlock), " != ", (*outsideInteractions)[outInterIdx].insideIndex);
                ParticleContainerClass* ptrLeaf = &interParticles;
                if(computeForReal) kernel->P2PRemote( FTreeCoordinate((*outsideInteractions)[outInterIdx].insideIndex, treeHeight-1), &particles, &particles ,
                                   &ptrLeaf, &(*outsideInteractions)[outInterIdx].relativeOutPosition, 1);
                nbInteractions += particles.getNbParticles() * ptrLeaf.getNbParticles();
            }
        }

        currentStat.addExecution(nbInteractions, timer.tacAndElapsed());
    }
#endif
    /////////////////////////////////////////////////////////////////////////////////////
    /// Direct Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void directInPassCallback(void *buffers[], void *cl_arg){
        ParticleGroupClass containers((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                                      STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                                      (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]));

        FStarPUPtrInterface* worker = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &intervalSize);
        worker->get<ThisClass>(FSTARPU_CPU_IDX)->directInPassPerform(&containers);
    }

    void directInPassPerform(ParticleGroupClass* containers){
        FTIME_TASKS(FTaskTimer::ScopeEvent taskTime(GetWorkerId(), &taskTimeRecorder, containers->getStartingIndex()*20*8 + 5, "P2P"));
        const MortonIndex blockStartIdx = containers->getStartingIndex();
        const MortonIndex blockEndIdx = containers->getEndingIndex();
        KernelClass*const kernel = kernels[GetWorkerId()];
        Stats& currentStat = stats[GetWorkerId()][INTER_P2P];
        FTic timer;
        FSize nbInteractions = 0;

        for(int leafIdx = 0 ; leafIdx < containers->getNumberOfLeavesInBlock() ; ++leafIdx){
            ParticleContainerClass particles = containers->template getLeaf<ParticleContainerClass>(leafIdx);

            MortonIndex interactionsIndexes[26];
            int interactionsPosition[26];
            FTreeCoordinate coord(containers->getLeafMortonIndex(leafIdx), treeHeight-1);
            int counter = coord.getNeighborsIndexes(treeHeight,interactionsIndexes,interactionsPosition);

            ParticleContainerClass interactionsObjects[26];
            ParticleContainerClass* interactions[26];
            int counterExistingCell = 0;

            for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
                if( blockStartIdx <= interactionsIndexes[idxInter] && interactionsIndexes[idxInter] < blockEndIdx ){
                    const int leafPos = containers->getLeafIndex(interactionsIndexes[idxInter]);
                    if(leafPos != -1){
                        interactionsObjects[counterExistingCell] = containers->template getLeaf<ParticleContainerClass>(leafPos);
                        interactionsPosition[counterExistingCell] = interactionsPosition[idxInter];
                        interactions[counterExistingCell] = &interactionsObjects[counterExistingCell];
                        nbInteractions += particles.getNbParticles() * interactionsObjects[counterExistingCell].getNbParticles();
                        counterExistingCell += 1;
                    }
                }
            }

            nbInteractions += particles.getNbParticles() * (particles.getNbParticles()-1);
            if(computeForReal) kernel->P2P( coord, &particles, &particles , interactions, interactionsPosition, counterExistingCell);
        }

        currentStat.addExecution(nbInteractions, timer.tacAndElapsed());
    }

    static void directInoutPassCallback(void *buffers[], void *cl_arg){
        ParticleGroupClass containers((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                                      STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                                      (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]));
        ParticleGroupClass externalContainers((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                                      STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                                      (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]));

        FStarPUPtrInterface* worker = nullptr;
        const std::vector<OutOfBlockInteraction>* outsideInteractions = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &outsideInteractions, &intervalSize);

        worker->get<ThisClass>(FSTARPU_CPU_IDX)->directInoutPassPerform(&containers, &externalContainers, outsideInteractions);
    }

    void directInoutPassPerform(ParticleGroupClass* containers, ParticleGroupClass* containersOther,
                                const std::vector<OutOfBlockInteraction>* outsideInteractions){
        FTIME_TASKS(FTaskTimer::ScopeEvent taskTime(GetWorkerId(), &taskTimeRecorder, ((containersOther->getStartingIndex()+1) * (containers->getStartingIndex()+1))*20*8 + 6, "P2P-ext"));
        KernelClass*const kernel = kernels[GetWorkerId()];
        Stats& currentStat = stats[GetWorkerId()][INTER_P2P_EXT];
        FTic timer;
        FSize nbInteractions = 0;
        for(int outInterIdx = 0 ; outInterIdx < int(outsideInteractions->size()) ; ++outInterIdx){
            ParticleContainerClass interParticles = containersOther->template getLeaf<ParticleContainerClass>((*outsideInteractions)[outInterIdx].outsideIdxInBlock);
            ParticleContainerClass particles = containers->template getLeaf<ParticleContainerClass>((*outsideInteractions)[outInterIdx].insideIdxInBlock);

            FAssertLF(containersOther->getLeafMortonIndex((*outsideInteractions)[outInterIdx].outsideIdxInBlock) == (*outsideInteractions)[outInterIdx].outIndex);
            FAssertLF(containers->getLeafMortonIndex((*outsideInteractions)[outInterIdx].insideIdxInBlock) == (*outsideInteractions)[outInterIdx].insideIndex);

            ParticleContainerClass* ptrLeaf = &interParticles;
            if(computeForReal) kernel->P2POuter( FTreeCoordinate((*outsideInteractions)[outInterIdx].insideIndex, treeHeight-1),
                                &particles , &ptrLeaf, &(*outsideInteractions)[outInterIdx].relativeOutPosition, 1);
            const int otherPosition = getOppositeNeighIndex((*outsideInteractions)[outInterIdx].relativeOutPosition);
            ptrLeaf = &particles;
            if(computeForReal) kernel->P2POuter( FTreeCoordinate((*outsideInteractions)[outInterIdx].outIndex, treeHeight-1),
                                    &interParticles , &ptrLeaf, &otherPosition, 1);

            nbInteractions += 2 * interParticles.getNbParticles() * ptrLeaf.getNbParticles();
        }

        currentStat.addExecution(nbInteractions, timer.tacAndElapsed());
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Merge Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void mergePassCallback(void *buffers[], void *cl_arg){
        CellContainerClass leafCells((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                                     STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                                     nullptr,
                                     (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]));
        ParticleGroupClass containers((unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                                     STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                                     (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]));

        FStarPUPtrInterface* worker = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &intervalSize);
        worker->get<ThisClass>(FSTARPU_CPU_IDX)->mergePassPerform(&leafCells, &containers);
    }

    void mergePassPerform(CellContainerClass* leafCells, ParticleGroupClass* containers){
        FTIME_TASKS(FTaskTimer::ScopeEvent taskTime(GetWorkerId(), &taskTimeRecorder, (leafCells->getStartingIndex()*20*8) + 7, "L2P"));
        FAssertLF(leafCells->getNumberOfCellsInBlock() == containers->getNumberOfLeavesInBlock());
        KernelClass*const kernel = kernels[GetWorkerId()];
        Stats& currentStat = stats[GetWorkerId()][INTER_L2P];
        FTic timer;
        FSize nbInteractions = 0;

        for(int cellIdx = 0 ; cellIdx < leafCells->getNumberOfCellsInBlock() ; ++cellIdx){
            CellClass cell = leafCells->getDownCell(cellIdx);
            FAssertLF(cell.getMortonIndex() == leafCells->getCellMortonIndex(cellIdx));
            ParticleContainerClass particles = containers->template getLeaf<ParticleContainerClass>(cellIdx);
            FAssertLF(leafCells->getCellMortonIndex(cellIdx) == containers->getLeafMortonIndex(cellIdx));
            if(computeForReal) kernel->L2P(&cell, &particles);
            nbInteractions += particles.getNbParticles();
        }

        currentStat.addExecution(nbInteractions, timer.tacAndElapsed());
    }

    static int getOppositeNeighIndex(const int index) {
        // ((idxX+1)*3 + (idxY+1)) * 3 + (idxZ+1)
        return 27-index-1;
    }

    static int getOppositeInterIndex(const int index) {
        // ((( (xdiff+3) * 7) + (ydiff+3))) * 7 + zdiff + 3
        return 343-index-1;
    }
};



#endif // FSTARPUCPTINTERACTIONSWRAPPER_HPP

