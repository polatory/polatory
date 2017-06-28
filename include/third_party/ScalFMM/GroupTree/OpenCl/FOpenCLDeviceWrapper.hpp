#ifndef FOPENCLDEVICEWRAPPER_HPP
#define FOPENCLDEVICEWRAPPER_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Core/FCoreCommon.hpp"
#include "../../Utils/FQuickSort.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../../Utils/FLog.hpp"
#include "../../Utils/FTic.hpp"
#include "../../Utils/FAssert.hpp"

#include "../Core/FOutOfBlockInteraction.hpp"

#include "FEmptyOpenCLCode.hpp"

#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

#include <starpu.h>


template <class OriginalKernelClass, class KernelFilenameClass = FEmptyOpenCLCode>
class FOpenCLDeviceWrapper {
protected:
    static void SetKernelArgs(cl_kernel& /*kernel*/, const int /*pos*/){
    }
    template <class ParamClass, class... Args>
    static void SetKernelArgs(cl_kernel& kernel, const int pos, ParamClass* param, Args... args){
        FAssertLF(clSetKernelArg(kernel, pos, sizeof(*param), param) == 0,
                  "Error when assigning opencl argument ", pos);
        SetKernelArgs(kernel, pos+1, args...);
    }

    int workerId;
    int workerDevid;

    struct starpu_opencl_program opencl_code;

    cl_context context;

    cl_kernel kernel_bottomPassPerform;
    cl_command_queue queue_bottomPassPerform;

    cl_kernel kernel_upwardPassPerform;
    cl_command_queue queue_upwardPassPerform;
#ifdef SCALFMM_USE_MPI
    cl_kernel kernel_transferInoutPassPerformMpi;
    cl_command_queue queue_transferInoutPassPerformMpi;
#endif
    cl_kernel kernel_transferInPassPerform;
    cl_command_queue queue_transferInPassPerform;

    cl_kernel kernel_transferInoutPassPerform;
    cl_command_queue queue_transferInoutPassPerform;

    cl_kernel kernel_downardPassPerform;
    cl_command_queue queue_downardPassPerform;
#ifdef SCALFMM_USE_MPI
    cl_kernel kernel_directInoutPassPerformMpi;
    cl_command_queue queue_directInoutPassPerformMpi;
#endif
    cl_kernel kernel_directInoutPassPerform;
    cl_command_queue queue_directInoutPassPerform;

    cl_kernel kernel_directInPassPerform;
    cl_command_queue queue_directInPassPerform;

    cl_kernel kernel_mergePassPerform;
    cl_command_queue queue_mergePassPerform;

    cl_mem user_data;

    int treeHeight;

    KernelFilenameClass kernelFilename;

public:
    FOpenCLDeviceWrapper(const int inTreeHeight) : workerId(0) , workerDevid(0), user_data(0), treeHeight(inTreeHeight){
        workerId = starpu_worker_get_id();
        workerDevid = starpu_worker_get_devid(workerId);

        const char* filename = kernelFilename.getKernelCode(workerDevid);
        if(filename){
            starpu_opencl_get_context (workerDevid, &context);

            const int err = starpu_opencl_load_opencl_from_string(filename, &opencl_code, "-cl-std=CL2.0 -cl-mad-enable -Werror");
            if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

            FAssertLF( starpu_opencl_load_kernel(&kernel_bottomPassPerform, &queue_bottomPassPerform, &opencl_code, "FOpenCL__bottomPassPerform", workerDevid) == CL_SUCCESS);
            FAssertLF( starpu_opencl_load_kernel(&kernel_upwardPassPerform, &queue_upwardPassPerform, &opencl_code, "FOpenCL__upwardPassPerform", workerDevid) == CL_SUCCESS);
#ifdef SCALFMM_USE_MPI
            FAssertLF( starpu_opencl_load_kernel(&kernel_transferInoutPassPerformMpi, &queue_transferInoutPassPerformMpi, &opencl_code, "FOpenCL__transferInoutPassPerformMpi", workerDevid) == CL_SUCCESS);
#endif
            FAssertLF( starpu_opencl_load_kernel(&kernel_transferInPassPerform, &queue_transferInPassPerform, &opencl_code, "FOpenCL__transferInPassPerform", workerDevid) == CL_SUCCESS);
            FAssertLF( starpu_opencl_load_kernel(&kernel_transferInoutPassPerform, &queue_transferInoutPassPerform, &opencl_code, "FOpenCL__transferInoutPassPerform", workerDevid) == CL_SUCCESS);
            FAssertLF( starpu_opencl_load_kernel(&kernel_downardPassPerform, &queue_downardPassPerform, &opencl_code, "FOpenCL__downardPassPerform", workerDevid) == CL_SUCCESS);
#ifdef SCALFMM_USE_MPI
            FAssertLF( starpu_opencl_load_kernel(&kernel_directInoutPassPerformMpi, &queue_directInoutPassPerformMpi, &opencl_code, "FOpenCL__directInoutPassPerformMpi", workerDevid) == CL_SUCCESS);
#endif
            FAssertLF( starpu_opencl_load_kernel(&kernel_directInoutPassPerform, &queue_directInoutPassPerform, &opencl_code, "FOpenCL__directInoutPassPerform", workerDevid) == CL_SUCCESS);
            FAssertLF( starpu_opencl_load_kernel(&kernel_directInPassPerform, &queue_directInPassPerform, &opencl_code, "FOpenCL__directInPassPerform", workerDevid) == CL_SUCCESS);
            FAssertLF( starpu_opencl_load_kernel(&kernel_mergePassPerform, &queue_mergePassPerform, &opencl_code, "FOpenCL__mergePassPerform", workerDevid) == CL_SUCCESS);
        }
        kernelFilename.releaseKernelCode();
    }

    virtual void initDeviceFromKernel(const OriginalKernelClass& /*originalKernel*/){
    }

    virtual void releaseKernel(){
        int err;
        err = starpu_opencl_release_kernel(kernel_bottomPassPerform);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        err = starpu_opencl_release_kernel(kernel_upwardPassPerform);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
#ifdef SCALFMM_USE_MPI
        err = starpu_opencl_release_kernel(kernel_transferInoutPassPerformMpi);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
#endif
        err = starpu_opencl_release_kernel(kernel_transferInPassPerform);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        err = starpu_opencl_release_kernel(kernel_transferInoutPassPerform);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        err = starpu_opencl_release_kernel(kernel_downardPassPerform);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
#ifdef SCALFMM_USE_MPI
        err = starpu_opencl_release_kernel(kernel_directInoutPassPerformMpi);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
#endif
        err = starpu_opencl_release_kernel(kernel_directInoutPassPerform);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        err = starpu_opencl_release_kernel(kernel_directInPassPerform);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        err = starpu_opencl_release_kernel(kernel_mergePassPerform);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        err = starpu_opencl_unload_opencl(&opencl_code);
        if(err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }

    virtual ~FOpenCLDeviceWrapper(){
    }

    cl_context& getOpenCLContext(){
        return context;
    }

    void bottomPassPerform(cl_mem leafCellsPtr,  size_t leafCellsSize, cl_mem leafCellsUpPtr, cl_mem containersPtr,  size_t containersSize,
                           const int intervalSize){
        SetKernelArgs(kernel_bottomPassPerform, 0, &leafCellsPtr,  &leafCellsSize, &leafCellsUpPtr,
                      &containersPtr,  &containersSize, &user_data);
        const int err = clEnqueueNDRangeKernel(queue_bottomPassPerform, kernel_bottomPassPerform, kernelFilename.getNbDims(), NULL,
                                               kernelFilename.getNbGroups(intervalSize), kernelFilename.getGroupSize(), 0, NULL, NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }


    void upwardPassPerform(cl_mem currentCellsPtr,  size_t currentCellsSize, cl_mem currentCellsUpPtr,
                           cl_mem subCellGroupsPtr,  size_t subCellGroupsSize, cl_mem subCellGroupsUpPtr,
                           int idxLevel, const int intervalSize){

        SetKernelArgs(kernel_upwardPassPerform, 0, &currentCellsPtr, &currentCellsSize, &currentCellsUpPtr,
                      &subCellGroupsPtr,  &subCellGroupsSize, &subCellGroupsUpPtr, &idxLevel, &user_data);
        const int err = clEnqueueNDRangeKernel(queue_upwardPassPerform, kernel_upwardPassPerform, kernelFilename.getNbDims(), NULL,
                                               kernelFilename.getNbGroups(intervalSize), kernelFilename.getGroupSize(), 0, NULL, NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }
#ifdef SCALFMM_USE_MPI
    void transferInoutPassPerformMpi(cl_mem currentCellsPtr, size_t currentCellsSize, cl_mem currentCellsDownPtr,
                                     cl_mem externalCellsPtr,  size_t externalCellsSize, cl_mem externalCellsUpPtr,
                                     int idxLevel, cl_mem outsideInteractionsCl, size_t  outsideInteractionsSize,
                                     const int intervalSize){
        SetKernelArgs(kernel_transferInoutPassPerformMpi, 0, &currentCellsPtr,&currentCellsSize, &currentCellsDownPtr,
                      &externalCellsPtr,  &externalCellsSize, &externalCellsUpPtr,
                      &idxLevel, &outsideInteractionsCl, &outsideInteractionsSize, &user_data);
        const int err = clEnqueueNDRangeKernel(queue_transferInoutPassPerformMpi, kernel_transferInoutPassPerformMpi, kernelFilename.getNbDims(), NULL,
                                               kernelFilename.getNbGroups(intervalSize), kernelFilename.getGroupSize(), 0, NULL, NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }
#endif
    void transferInPassPerform(cl_mem currentCellsPtr, size_t currentCellsSize,
                               cl_mem currentCellsUpPtr, cl_mem currentCellsDownPtr, int idxLevel, const int intervalSize){
        SetKernelArgs(kernel_transferInPassPerform, 0, &currentCellsPtr, &currentCellsSize, &currentCellsUpPtr,
                      &currentCellsDownPtr, &idxLevel, &user_data);
        const int err = clEnqueueNDRangeKernel(queue_transferInPassPerform, kernel_transferInPassPerform, kernelFilename.getNbDims(), NULL,
                                               kernelFilename.getNbGroups(intervalSize), kernelFilename.getGroupSize(), 0, NULL, NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }

    void transferInoutPassPerform(cl_mem currentCellsPtr, size_t currentCellsSize, cl_mem currentCellsUpPtr,
                                  cl_mem externalCellsPtr, size_t externalCellsSize, cl_mem externalCellsDownPtr,
                                  int idxLevel, const int mode, cl_mem outsideInteractionsCl, size_t outsideInteractionsSize, const int intervalSize){
        SetKernelArgs(kernel_transferInoutPassPerform, 0, &currentCellsPtr,&currentCellsSize, &currentCellsUpPtr,
                      &externalCellsPtr, &externalCellsSize, &externalCellsDownPtr,
                      &idxLevel, &mode, &outsideInteractionsCl,&outsideInteractionsSize, &user_data);
        const int err = clEnqueueNDRangeKernel(queue_transferInoutPassPerform, kernel_transferInoutPassPerform, kernelFilename.getNbDims(), NULL,
                                               kernelFilename.getNbGroups(intervalSize), kernelFilename.getGroupSize(), 0, NULL, NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }

    void downardPassPerform(cl_mem currentCellsPtr, size_t currentCellsSize, cl_mem currentCellsDownPtr,
                            cl_mem subCellGroupsPtr,  size_t subCellGroupsSize, cl_mem subCellGroupsDownPtr,
                            int idxLevel, const int intervalSize){
        SetKernelArgs(kernel_downardPassPerform, 0, &currentCellsPtr, &currentCellsSize, &currentCellsDownPtr,
                      &subCellGroupsPtr, &subCellGroupsSize, &subCellGroupsDownPtr, &idxLevel, &user_data);
        const int err = clEnqueueNDRangeKernel(queue_downardPassPerform, kernel_downardPassPerform, kernelFilename.getNbDims(), NULL,
                                               kernelFilename.getNbGroups(intervalSize), kernelFilename.getGroupSize(), 0, NULL, NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }
#ifdef SCALFMM_USE_MPI
    void directInoutPassPerformMpi(cl_mem containersPtr, size_t containersSize, cl_mem containersDownPtr,
                                   cl_mem externalContainersPtr,  size_t externalContainersSize, cl_mem outsideInteractionsCl,
                                   size_t outsideInteractionsSize, const int intervalSize){
        SetKernelArgs(kernel_directInoutPassPerformMpi, 0, &containersPtr, &containersSize, &containersDownPtr,
                      &externalContainersPtr, &externalContainersSize, &outsideInteractionsCl,&outsideInteractionsSize, &treeHeight, &user_data);
        const int err = clEnqueueNDRangeKernel(queue_directInoutPassPerformMpi, kernel_directInoutPassPerformMpi, kernelFilename.getNbDims(), NULL,
                                               kernelFilename.getNbGroups(intervalSize), kernelFilename.getGroupSize(), 0, NULL, NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }
#endif
    void directInPassPerform(cl_mem containersPtr,  size_t containerSize, cl_mem containersDownPtr, const int intervalSize){
        SetKernelArgs(kernel_directInPassPerform, 0, &containersPtr, &containerSize, &containersDownPtr, &treeHeight, &user_data);
        const int err = clEnqueueNDRangeKernel(queue_directInPassPerform, kernel_directInPassPerform, kernelFilename.getNbDims(), NULL,
                                               kernelFilename.getNbGroups(intervalSize), kernelFilename.getGroupSize(), 0, NULL, NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }

    void directInoutPassPerform(cl_mem containersPtr, size_t containerSize, cl_mem containersDownPtr,
                                cl_mem externalContainersPtr, size_t externalContainersSize, cl_mem externalContainersDownPtr,
                                cl_mem outsideInteractionsCl, size_t  outsideInteractionsSize, const int intervalSize){
        SetKernelArgs(kernel_directInoutPassPerform, 0, &containersPtr, &containerSize, &containersDownPtr,
                      &externalContainersPtr, &externalContainersSize, &externalContainersDownPtr,
                      &outsideInteractionsCl, &outsideInteractionsSize, &treeHeight, &user_data);
        const int err = clEnqueueNDRangeKernel(queue_directInoutPassPerform, kernel_directInoutPassPerform, kernelFilename.getNbDims(), NULL,
                                               kernelFilename.getNbGroups(intervalSize), kernelFilename.getGroupSize(), 0, NULL, NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }

    void mergePassPerform(cl_mem leafCellsPtr, size_t leafCellsSize, cl_mem leafCellsDownPtr,
                          cl_mem containersPtr, size_t containersSize, cl_mem containersDownPtr, const int intervalSize){
        SetKernelArgs(kernel_mergePassPerform, 0, &leafCellsPtr, &leafCellsSize, &leafCellsDownPtr,
                      &containersPtr, &containersSize, &containersDownPtr, &user_data);
        const int err = clEnqueueNDRangeKernel(queue_mergePassPerform, kernel_mergePassPerform, kernelFilename.getNbDims(), NULL,
                                               kernelFilename.getNbGroups(intervalSize), kernelFilename.getGroupSize(), 0, NULL, NULL);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }
};

#endif // FOPENCLDEVICEWRAPPER_HPP

