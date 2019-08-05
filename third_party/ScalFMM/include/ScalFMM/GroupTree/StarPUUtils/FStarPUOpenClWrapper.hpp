

#ifndef FSTARPUOPENCLWRAPPER_HPP
#define FSTARPUOPENCLWRAPPER_HPP


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

#include <starpu.h>

#if defined(STARPU_USE_MPI) && defined(SCALFMM_USE_MPI)
#include <starpu_mpi.h>
#endif

#include "FStarPUUtils.hpp"

#ifndef CL_VERSION_2_0
#warning should use opencl 2.0
#define CL_MEM_HOST_NO_ACCESS 0
#endif


template <class KernelClass, class OpenCLKernelClass>
class FStarPUOpenClWrapper {
protected:
    typedef FStarPUOpenClWrapper<KernelClass, OpenCLKernelClass> ThisClass;

    const int treeHeight;
    OpenCLKernelClass* kernels[STARPU_MAXOPENCLDEVS];        //< The kernels

    static int GetWorkerId() {
        return FMath::Max(0, starpu_worker_get_id());
    }

public:
    FStarPUOpenClWrapper(const int inTreeHeight): treeHeight(inTreeHeight){
        memset(kernels, 0, sizeof(OpenCLKernelClass*)*STARPU_MAXOPENCLDEVS);
    }

    OpenCLKernelClass* getKernel(const int workerId){
        return kernels[workerId];
    }

    const OpenCLKernelClass* getKernel(const int workerId) const {
        return kernels[workerId];
    }

    void initKernel(const int workerId, KernelClass* originalKernel){
        FAssertLF(kernels[workerId] == nullptr);
        kernels[workerId] = new OpenCLKernelClass(treeHeight);
        kernels[workerId]->initDeviceFromKernel(*originalKernel);
    }

    void releaseKernel(const int workerId){
        kernels[workerId]->releaseKernel();
        delete kernels[workerId];
        kernels[workerId] = nullptr;
    }

    ~FStarPUOpenClWrapper(){
        for(int idxKernel = 0 ; idxKernel < STARPU_MAXOPENCLDEVS ; ++idxKernel ){
            FAssertLF(kernels[idxKernel] == nullptr);
        }
    }

    static void bottomPassCallback(void *buffers[], void *cl_arg){
        cl_mem leafCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]));
        size_t leafCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);
        cl_mem leafCellsUpPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[1]));

        cl_mem containersPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[2]));
        size_t containersSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        FStarPUPtrInterface* worker = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &intervalSize);
        OpenCLKernelClass* kernel = worker->get<ThisClass>(FSTARPU_OPENCL_IDX)->kernels[GetWorkerId()];

        kernel->bottomPassPerform(leafCellsPtr, leafCellsSize, leafCellsUpPtr, containersPtr, containersSize,
                                  intervalSize);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Upward Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void upwardPassCallback(void *buffers[], void *cl_arg){
        cl_mem currentCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]));
        size_t currentCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);
        cl_mem currentCellsUpPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[1]));

        FStarPUPtrInterface* worker = nullptr;
        int nbSubCellGroups = 0;
        int idxLevel = 0;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &nbSubCellGroups, &idxLevel, &intervalSize);

        cl_mem otherCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[2]));
        size_t otherCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);
        cl_mem otherCellsUpPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[3]));

        OpenCLKernelClass* kernel = worker->get<ThisClass>(FSTARPU_OPENCL_IDX)->kernels[GetWorkerId()];
        kernel->upwardPassPerform(currentCellsPtr, currentCellsSize, currentCellsUpPtr,
                                  otherCellsPtr, otherCellsSize, otherCellsUpPtr, idxLevel,
                                  intervalSize);
    }


    /////////////////////////////////////////////////////////////////////////////////////
    /// Transfer Pass Mpi
    /////////////////////////////////////////////////////////////////////////////////////
#if defined(STARPU_USE_MPI) && defined(SCALFMM_USE_MPI)
    static void transferInoutPassCallbackMpi(void *buffers[], void *cl_arg){
        cl_mem currentCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]));
        size_t currentCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);
        cl_mem currentCellsDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[1]));

        cl_mem externalCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[2]));
        size_t externalCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);
        cl_mem externalCellsUpPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[3]));

        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        const std::vector<OutOfBlockInteraction>* outsideInteractions;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &outsideInteractions, &intervalSize);

        OpenCLKernelClass* kernel = worker->get<ThisClass>(FSTARPU_OPENCL_IDX)->kernels[GetWorkerId()];
        cl_int errcode_ret;
        cl_mem outsideInteractionsCl = clCreateBuffer(kernel->getOpenCLContext(),
           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
           outsideInteractions->size()*sizeof(OutOfBlockInteraction),
           const_cast<OutOfBlockInteraction*>(outsideInteractions->data()), &errcode_ret);
        FAssertLF(outsideInteractionsCl && errcode_ret == CL_SUCCESS);

        kernel->transferInoutPassPerformMpi(currentCellsPtr,
                    currentCellsSize, currentCellsDownPtr,
                    externalCellsPtr, externalCellsSize, externalCellsUpPtr,
                    idxLevel, outsideInteractionsCl, outsideInteractions->size(),
                                            intervalSize);

        clReleaseMemObject(outsideInteractionsCl);
    }

#endif
    /////////////////////////////////////////////////////////////////////////////////////
    /// Transfer Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void transferInPassCallback(void *buffers[], void *cl_arg){
        cl_mem currentCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]));
        size_t currentCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);
        cl_mem currentCellsUpPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[1]));
        cl_mem currentCellsDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[2]));

        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &intervalSize);

        OpenCLKernelClass* kernel = worker->get<ThisClass>(FSTARPU_OPENCL_IDX)->kernels[GetWorkerId()];
        kernel->transferInPassPerform(currentCellsPtr, currentCellsSize, currentCellsUpPtr, currentCellsDownPtr, idxLevel,
                                      intervalSize);
    }

    static void transferInoutPassCallback(void *buffers[], void *cl_arg){
        cl_mem currentCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]));
        size_t currentCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);
        cl_mem currentCellsDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[1]));

        cl_mem externalCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[2]));
        size_t externalCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);
        cl_mem externalCellsUpPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[3]));

        FStarPUPtrInterface* worker = nullptr;
        int idxLevel = 0;
        const std::vector<OutOfBlockInteraction>* outsideInteractions;
        int intervalSize;
        int mode = 0;
        starpu_codelet_unpack_args(cl_arg, &worker, &idxLevel, &outsideInteractions, &intervalSize, &mode);

        OpenCLKernelClass* kernel = worker->get<ThisClass>(FSTARPU_OPENCL_IDX)->kernels[GetWorkerId()];
        cl_int errcode_ret;
        cl_mem outsideInteractionsCl = clCreateBuffer(kernel->getOpenCLContext(),
           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
           outsideInteractions->size()*sizeof(OutOfBlockInteraction),
           const_cast<OutOfBlockInteraction*>(outsideInteractions->data()), &errcode_ret);
        FAssertLF(outsideInteractionsCl && errcode_ret == CL_SUCCESS);

        kernel->transferInoutPassPerform(currentCellsPtr, currentCellsSize, currentCellsDownPtr,
                                         externalCellsPtr, externalCellsSize, externalCellsUpPtr,
                                         idxLevel, mode, outsideInteractionsCl, outsideInteractions->size(),
                                         intervalSize);

        clReleaseMemObject(outsideInteractionsCl);
    }


    /////////////////////////////////////////////////////////////////////////////////////
    /// Downard Pass
    /////////////////////////////////////////////////////////////////////////////////////
    static void downardPassCallback(void *buffers[], void *cl_arg){
        cl_mem currentCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]));
        size_t currentCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);
        cl_mem currentCellsDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[1]));

        FStarPUPtrInterface* worker = nullptr;
        int nbSubCellGroups = 0;
        int idxLevel = 0;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &nbSubCellGroups, &idxLevel, &intervalSize);

        cl_mem otherCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[2]));
        size_t otherCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);
        cl_mem otherCellsDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[3]));

        OpenCLKernelClass* kernel = worker->get<ThisClass>(FSTARPU_OPENCL_IDX)->kernels[GetWorkerId()];
        kernel->downardPassPerform(currentCellsPtr, currentCellsSize, currentCellsDownPtr,
                                   otherCellsPtr, otherCellsSize, otherCellsDownPtr, idxLevel,
                                   intervalSize);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Direct Pass MPI
    /////////////////////////////////////////////////////////////////////////////////////

#if defined(STARPU_USE_MPI) && defined(SCALFMM_USE_MPI)
    static void directInoutPassCallbackMpi(void *buffers[], void *cl_arg){
        cl_mem containersPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]));
        size_t containersSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);
        cl_mem containersDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[1]));

        cl_mem externalContainersPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[2]));
        size_t externalContainersSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        FStarPUPtrInterface* worker = nullptr;
        const std::vector<OutOfBlockInteraction>* outsideInteractions = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &outsideInteractions, &intervalSize);

        OpenCLKernelClass* kernel = worker->get<ThisClass>(FSTARPU_OPENCL_IDX)->kernels[GetWorkerId()];
        cl_int errcode_ret;
        cl_mem outsideInteractionsCl = clCreateBuffer(kernel->getOpenCLContext(),
           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
           outsideInteractions->size()*sizeof(OutOfBlockInteraction),
           const_cast<OutOfBlockInteraction*>(outsideInteractions->data()), &errcode_ret);
        FAssertLF(outsideInteractionsCl && errcode_ret == CL_SUCCESS);

        kernel->directInoutPassPerformMpi(containersPtr, containersSize, containersDownPtr,
              externalContainersPtr, externalContainersSize, outsideInteractionsCl, outsideInteractions->size(),
                                          intervalSize);

        clReleaseMemObject(outsideInteractionsCl);
    }
#endif
    /////////////////////////////////////////////////////////////////////////////////////
    /// Direct Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void directInPassCallback(void *buffers[], void *cl_arg){
        cl_mem containersPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]));
        size_t containerSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);
        cl_mem containersDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[1]));

        FStarPUPtrInterface* worker = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &intervalSize);
        OpenCLKernelClass* kernel = worker->get<ThisClass>(FSTARPU_OPENCL_IDX)->kernels[GetWorkerId()];
        kernel->directInPassPerform(containersPtr, containerSize, containersDownPtr,
                                    intervalSize);
    }

    static void directInoutPassCallback(void *buffers[], void *cl_arg){
        cl_mem containersPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]));
        size_t containerSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);
        cl_mem containersDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[1]));

        cl_mem externalContainersPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[2]));
        size_t externalContainersSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);
        cl_mem externalContainersDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[3]));

        FStarPUPtrInterface* worker = nullptr;
        const std::vector<OutOfBlockInteraction>* outsideInteractions = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &outsideInteractions, &intervalSize);

        OpenCLKernelClass* kernel = worker->get<ThisClass>(FSTARPU_OPENCL_IDX)->kernels[GetWorkerId()];
        cl_int errcode_ret;
        cl_mem outsideInteractionsCl = clCreateBuffer(kernel->getOpenCLContext(),
           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
           outsideInteractions->size()*sizeof(OutOfBlockInteraction),
           const_cast<OutOfBlockInteraction*>(outsideInteractions->data()), &errcode_ret);
        FAssertLF(outsideInteractionsCl && errcode_ret == CL_SUCCESS);

        kernel->directInoutPassPerform(containersPtr, containerSize, containersDownPtr,
                                       externalContainersPtr, externalContainersSize, externalContainersDownPtr,
                                       outsideInteractionsCl, outsideInteractions->size(),
                                       intervalSize);

        clReleaseMemObject(outsideInteractionsCl);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Merge Pass
    /////////////////////////////////////////////////////////////////////////////////////

    static void mergePassCallback(void *buffers[], void *cl_arg){
        cl_mem leafCellsPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[0]));
        size_t leafCellsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);
        cl_mem leafCellsDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[1]));

        cl_mem containersPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[2]));
        size_t containersSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);
        cl_mem containersDownPtr = ((cl_mem)STARPU_VARIABLE_GET_DEV_HANDLE(buffers[3]));

        FStarPUPtrInterface* worker = nullptr;
        int intervalSize;
        starpu_codelet_unpack_args(cl_arg, &worker, &intervalSize);
        OpenCLKernelClass* kernel = worker->get<ThisClass>(FSTARPU_OPENCL_IDX)->kernels[GetWorkerId()];
        kernel->mergePassPerform(leafCellsPtr, leafCellsSize, leafCellsDownPtr,
                                 containersPtr, containersSize, containersDownPtr,
                                 intervalSize);
    }
};


#endif // FSTARPUOPENCLWRAPPER_HPP

