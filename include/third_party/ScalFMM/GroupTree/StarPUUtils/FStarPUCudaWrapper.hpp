#ifndef FSTARPUCUDAWRAPPER_HPP
#define FSTARPUCUDAWRAPPER_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Core/FCoreCommon.hpp"
#include "../../Utils/FQuickSort.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../../Utils/FLog.hpp"
#include "../../Utils/FTic.hpp"
#include "../../Utils/FAssert.hpp"
#include "../../Utils/FAssert.hpp"

#include "../Core/FOutOfBlockInteraction.hpp"

#ifdef SCALFMM_USE_MPI
#include "../../Utils/FMpi.hpp"
#endif

#include <vector>
#include <memory>

#include <omp.h>

#include <starpu.h>

#if defined(STARPU_USE_MPI) && defined(SCALFMM_USE_MPI)
#include <starpu_mpi.h>
#endif

#include "../Cuda/FCudaDeviceWrapper.hpp"

#include "../Uniform/FUnifCudaCellPOD.hpp" // TODO remove

#include "FStarPUUtils.hpp"

template <class KernelClass, class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CudaCellGroupClass, class CudaParticleGroupClass, class CudaParticleContainerClass,
          class CudaKernelClass>
class FStarPUCudaWrapper {
protected:
    typedef FStarPUCudaWrapper<KernelClass, SymboleCellClass, PoleCellClass, LocalCellClass,
        CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass> ThisClass;

    const int treeHeight;
    CudaKernelClass* kernels[STARPU_MAXCUDADEVS];        //< The kernels

    static int GetWorkerId() {
        return FMath::Max(0, starpu_worker_get_id());
    }

public:
    FStarPUCudaWrapper(const int inTreeHeight): treeHeight(inTreeHeight){
        memset(kernels, 0, sizeof(CudaKernelClass*)*STARPU_MAXCUDADEVS);
    }

    CudaKernelClass* getKernel(const int workerId){
        return kernels[workerId];
    }

    const CudaKernelClass* getKernel(const int workerId) const {
        return kernels[workerId];
    }

    void initKernel(const int workerId, KernelClass* originalKernel){
        FAssertLF(kernels[workerId] == nullptr);
        kernels[workerId] = FCuda__BuildCudaKernel<CudaKernelClass>(originalKernel);
    }

    void releaseKernel(const int workerId){
        FCuda__ReleaseCudaKernel(kernels[workerId]);
        kernels[workerId] = nullptr;
    }

    ~FStarPUCudaWrapper(){
        for(int idxKernel = 0 ; idxKernel < STARPU_MAXCUDADEVS ; ++idxKernel ){
            FAssertLF(kernels[idxKernel] == nullptr);
        }
    }

    static void bottomPassCallback(void *buffers[], void *cl_arg){
        FStarPUPtrInterface* worker = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &intervalSize, &intervalSize);

        CudaKernelClass* kernel = worker->get<ThisClass>(FSTARPU_CUDA_IDX)->kernels[GetWorkerId()];

        FCuda__bottomPassCallback< SymboleCellClass, PoleCellClass, LocalCellClass,
                CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass>(
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                    STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                    STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                    kernel, starpu_cuda_get_local_stream(),
                    FCuda__GetGridSize(kernel,intervalSize),FCuda__GetBlockSize(kernel));
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Upward Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void upwardPassCallback(void *buffers[], void *cl_arg){
        FStarPUPtrInterface* worker = nullptr;
        int nbSubCellGroups = 0;
        int idxLevel = 0;
        int intervalSize = 0;
        starpu_codelet_unpack_args(cl_arg, &worker, &nbSubCellGroups, &idxLevel, &intervalSize);

        CudaKernelClass* kernel = worker->get<ThisClass>(FSTARPU_CUDA_IDX)->kernels[GetWorkerId()];

        FCuda__upwardPassCallback< SymboleCellClass, PoleCellClass, LocalCellClass,
                CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass>(
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]),
                idxLevel, kernel, starpu_cuda_get_local_stream(),
                FCuda__GetGridSize(kernel,intervalSize),FCuda__GetBlockSize(kernel));
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Transfer Pass Mpi
    /////////////////////////////////////////////////////////////////////////////////////
#if defined(STARPU_USE_MPI) && defined(SCALFMM_USE_MPI)
    static void transferInoutPassCallbackMpi(void *buffers[], void *cl_arg){
        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        const std::vector<OutOfBlockInteraction>* outsideInteractions = nullptr;
        int intervalSize = 0;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &outsideInteractions, &intervalSize);
        const int nbInteractions = int(outsideInteractions->size());

        CudaKernelClass* kernel = worker->get<ThisClass>(FSTARPU_CUDA_IDX)->kernels[GetWorkerId()];

        std::unique_ptr<int[]> safeInteractions(new int[nbInteractions+1]);
        const int nbSafeInteractions = GetClusterOfInteractionsOutside(safeInteractions.get(), outsideInteractions->data(), nbInteractions);


        FCuda__transferInoutPassCallbackMpi< SymboleCellClass, PoleCellClass, LocalCellClass,
                CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass>(
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]),
                idxLevel, outsideInteractions->data(), nbInteractions,
                safeInteractions.get(), nbSafeInteractions,
                kernel,
                starpu_cuda_get_local_stream(),
                FCuda__GetGridSize(kernel,intervalSize),FCuda__GetBlockSize(kernel));
    }
#endif
    /////////////////////////////////////////////////////////////////////////////////////
    /// Transfer Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void transferInPassCallback(void *buffers[], void *cl_arg){
        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        int intervalSize = 0;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &intervalSize);

        CudaKernelClass* kernel = worker->get<ThisClass>(FSTARPU_CUDA_IDX)->kernels[GetWorkerId()];

        FCuda__transferInPassCallback< SymboleCellClass, PoleCellClass, LocalCellClass,
                CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass>(
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                idxLevel, kernel, starpu_cuda_get_local_stream(),
                FCuda__GetGridSize(kernel,intervalSize),FCuda__GetBlockSize(kernel));
    }

    static void transferInoutPassCallback(void *buffers[], void *cl_arg){
        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        const std::vector<OutOfBlockInteraction>* outsideInteractions = nullptr;
        int intervalSize = 0;
        int mode = 0;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &outsideInteractions, &intervalSize, &mode);
        const int nbInteractions = int(outsideInteractions->size());

        CudaKernelClass* kernel = worker->get<ThisClass>(FSTARPU_CUDA_IDX)->kernels[GetWorkerId()];

        // outsideInteractions is sorted following the outIndex
        // Compute the cell interval
        const OutOfBlockInteraction* interactions;
        std::unique_ptr<int[]> safeInteractions(new int[nbInteractions+1]);
        int nbSafeInteractions = 0;
        std::unique_ptr<OutOfBlockInteraction[]> insideInteractions;
        if(mode == 0){
            interactions = outsideInteractions->data();
            nbSafeInteractions = GetClusterOfInteractionsOutside(safeInteractions.get(), outsideInteractions->data(), nbInteractions);
        }
        else{
            insideInteractions.reset(new OutOfBlockInteraction[nbInteractions]);
            memcpy(insideInteractions.get(), outsideInteractions->data(), nbInteractions*sizeof(OutOfBlockInteraction));

            FQuickSort<OutOfBlockInteraction>::QsSequential(insideInteractions.get(), nbInteractions,
                                     [](const OutOfBlockInteraction& inter1, const OutOfBlockInteraction& inter2){
                // Could be insideIndex since the block are in morton order
                return inter1.insideIdxInBlock <= inter2.insideIdxInBlock;
            });
            interactions = insideInteractions.get();

            nbSafeInteractions = GetClusterOfInteractionsInside(safeInteractions.get(), insideInteractions.get(), nbInteractions);
        }


        FCuda__transferInoutPassCallback< SymboleCellClass, PoleCellClass, LocalCellClass,
                CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass>(
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                    STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                    STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]),
                    idxLevel, mode, interactions, nbInteractions,
                    safeInteractions.get(), nbSafeInteractions,
                    kernel,
                    starpu_cuda_get_local_stream(),
                FCuda__GetGridSize(kernel,intervalSize),FCuda__GetBlockSize(kernel));
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Downard Pass
    /////////////////////////////////////////////////////////////////////////////////////
    static void downardPassCallback(void *buffers[], void *cl_arg){
        FStarPUPtrInterface* worker = nullptr;
        int nbSubCellGroups = 0;
        int idxLevel = 0;
        int intervalSize = 0;
        starpu_codelet_unpack_args(cl_arg, &worker, &nbSubCellGroups, &idxLevel, &intervalSize);

        CudaKernelClass* kernel = worker->get<ThisClass>(FSTARPU_CUDA_IDX)->kernels[GetWorkerId()];

        FCuda__downardPassCallback< SymboleCellClass, PoleCellClass, LocalCellClass,
                CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass>(
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]),
                idxLevel, kernel, starpu_cuda_get_local_stream(),
                FCuda__GetGridSize(kernel,intervalSize),FCuda__GetBlockSize(kernel));
    }
    /////////////////////////////////////////////////////////////////////////////////////
    /// Direct Pass MPI
    /////////////////////////////////////////////////////////////////////////////////////

#if defined(STARPU_USE_MPI) && defined(SCALFMM_USE_MPI)
    static void directInoutPassCallbackMpi(void *buffers[], void *cl_arg){

        FStarPUPtrInterface* worker = nullptr;
        const std::vector<OutOfBlockInteraction>* outsideInteractions = nullptr;
        int intervalSize = 0;
        starpu_codelet_unpack_args(cl_arg, &worker, &outsideInteractions, &intervalSize);
        const int nbInteractions = int(outsideInteractions->size());

        CudaKernelClass* kernel = worker->get<ThisClass>(FSTARPU_CUDA_IDX)->kernels[GetWorkerId()];

        std::unique_ptr<int[]> safeOuterInteractions(new int[nbInteractions+1]);
        const int counterOuterCell = GetClusterOfInteractionsOutside(safeOuterInteractions.get(), outsideInteractions->data(), nbInteractions);

        FCuda__directInoutPassCallbackMpi< SymboleCellClass, PoleCellClass, LocalCellClass,
                CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass>(
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                outsideInteractions->data(), nbInteractions,
                safeOuterInteractions.get(), counterOuterCell,
                worker->get<ThisClass>(FSTARPU_CUDA_IDX)->treeHeight ,kernel, starpu_cuda_get_local_stream(),
                FCuda__GetGridSize(kernel,intervalSize),FCuda__GetBlockSize(kernel));
    }
#endif
    /////////////////////////////////////////////////////////////////////////////////////
    /// Direct Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void directInPassCallback(void *buffers[], void *cl_arg){
        FStarPUPtrInterface* worker = nullptr;
        int intervalSize = 0;
        starpu_codelet_unpack_args(cl_arg, &worker, &intervalSize);
        CudaKernelClass* kernel = worker->get<ThisClass>(FSTARPU_CUDA_IDX)->kernels[GetWorkerId()];

        FCuda__directInPassCallback< SymboleCellClass, PoleCellClass, LocalCellClass,
                CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass>(
                    (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                worker->get<ThisClass>(FSTARPU_CUDA_IDX)->treeHeight, kernel, starpu_cuda_get_local_stream(),
                FCuda__GetGridSize(kernel,intervalSize),FCuda__GetBlockSize(kernel));
    }

    static int GetClusterOfInteractionsInside(int safeOuterInteractions[],
                                        const OutOfBlockInteraction outsideInteractions[],
                                        const int nbInteractions){
        safeOuterInteractions[0] = 0;
        safeOuterInteractions[1] = 0;
        int counterInnerCell = 1;
        for(int idxInter = 0 ; idxInter < int(nbInteractions) ; ++idxInter){
            if(outsideInteractions[safeOuterInteractions[counterInnerCell]].insideIdxInBlock
                    != outsideInteractions[idxInter].insideIdxInBlock){
                FAssertLF(outsideInteractions[safeOuterInteractions[counterInnerCell]].insideIdxInBlock
                        < outsideInteractions[idxInter].insideIdxInBlock);
                counterInnerCell += 1;
                FAssertLF(counterInnerCell <= nbInteractions);
                safeOuterInteractions[counterInnerCell] = safeOuterInteractions[counterInnerCell-1];
            }
            else{
                safeOuterInteractions[counterInnerCell] += 1;
            }
        }
        FAssertLF(safeOuterInteractions[counterInnerCell] == nbInteractions);
        return counterInnerCell;
    }

    static int GetClusterOfInteractionsOutside(int safeOuterInteractions[],
                                        const OutOfBlockInteraction outsideInteractions[],
                                        const int nbInteractions){
        safeOuterInteractions[0] = 0;
        safeOuterInteractions[1] = 0;
        int counterInnerCell = 1;
        for(int idxInter = 0 ; idxInter < int(nbInteractions) ; ++idxInter){
            if(outsideInteractions[safeOuterInteractions[counterInnerCell]].outsideIdxInBlock
                    != outsideInteractions[idxInter].outsideIdxInBlock){
                FAssertLF(outsideInteractions[safeOuterInteractions[counterInnerCell]].outsideIdxInBlock
                        < outsideInteractions[idxInter].outsideIdxInBlock);
                counterInnerCell += 1;
                FAssertLF(counterInnerCell <= nbInteractions);
                safeOuterInteractions[counterInnerCell] = safeOuterInteractions[counterInnerCell-1];
            }
            else{
                safeOuterInteractions[counterInnerCell] += 1;
            }
        }
        FAssertLF(safeOuterInteractions[counterInnerCell] == nbInteractions);
        return counterInnerCell;
    }

    static void directInoutPassCallback(void *buffers[], void *cl_arg){
        FStarPUPtrInterface* worker = nullptr;
        const std::vector<OutOfBlockInteraction>* outsideInteractions = nullptr;
        int intervalSize = 0;
        starpu_codelet_unpack_args(cl_arg, &worker, &outsideInteractions, &intervalSize);
        const int nbInteractions = int(outsideInteractions->size());

        CudaKernelClass* kernel = worker->get<ThisClass>(FSTARPU_CUDA_IDX)->kernels[GetWorkerId()];

        // outsideInteractions is sorted following the outIndex
        // Compute the cell interval
        std::unique_ptr<int[]> safeOuterInteractions(new int[nbInteractions+1]);
        const int counterOuterCell = GetClusterOfInteractionsOutside(safeOuterInteractions.get(), outsideInteractions->data(), nbInteractions);

        std::unique_ptr<OutOfBlockInteraction[]> insideInteractions(new OutOfBlockInteraction[nbInteractions]);
        memcpy(insideInteractions.get(), outsideInteractions->data(), nbInteractions*sizeof(OutOfBlockInteraction));

        FQuickSort<OutOfBlockInteraction>::QsSequential(insideInteractions.get(), nbInteractions,
                                 [](const OutOfBlockInteraction& inter1, const OutOfBlockInteraction& inter2){
            // Could be insideIndex since the block are in morton order
            return inter1.insideIdxInBlock <= inter2.insideIdxInBlock;
        });

        std::unique_ptr<int[]> safeInnterInteractions(new int[nbInteractions+1]);
        const int counterInnerCell = GetClusterOfInteractionsInside(safeInnterInteractions.get(), insideInteractions.get(), nbInteractions);

        FCuda__directInoutPassCallback< SymboleCellClass, PoleCellClass, LocalCellClass,
                CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass>(
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]),
                outsideInteractions->data(), nbInteractions,
                safeOuterInteractions.get(), counterOuterCell,
                insideInteractions.get(),
                safeInnterInteractions.get(), counterInnerCell,
                worker->get<ThisClass>(FSTARPU_CUDA_IDX)->treeHeight,
                kernel, starpu_cuda_get_local_stream(),
                FCuda__GetGridSize(kernel,intervalSize),FCuda__GetBlockSize(kernel));
    }


    /////////////////////////////////////////////////////////////////////////////////////
    /// Merge Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void mergePassCallback(void *buffers[], void *cl_arg){
        FStarPUPtrInterface* worker = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &intervalSize);

        CudaKernelClass* kernel = worker->get<ThisClass>(FSTARPU_CUDA_IDX)->kernels[GetWorkerId()];

        FCuda__mergePassCallback< SymboleCellClass, PoleCellClass, LocalCellClass,
                CudaCellGroupClass, CudaParticleGroupClass, CudaParticleContainerClass, CudaKernelClass>(
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]),
                STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]),
                (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]),
                kernel, starpu_cuda_get_local_stream(),
                FCuda__GetGridSize(kernel,intervalSize),FCuda__GetBlockSize(kernel));
    }
};


#endif // FSTARPUCUDAWRAPPER_HPP

