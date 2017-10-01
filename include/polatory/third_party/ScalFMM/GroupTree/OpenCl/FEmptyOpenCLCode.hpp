#ifndef FEMPTYOPENCLCODE_HPP
#define FEMPTYOPENCLCODE_HPP

// Return the same thing as FEmptyKernel.cl

#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

class FEmptyOpenCLCode{

public:
    FEmptyOpenCLCode(){
    }

    const char* getKernelCode(const int /*inDevId*/){
        const char* kernelcode =
                "typedef long long int MortonIndex; \
                #define DefaultStructAlign " FStarPUDefaultAlignStr "\
                \
                struct OutOfBlockInteraction{\
                    MortonIndex outIndex;\
                    MortonIndex insideIndex;\
                    int relativeOutPosition;\
                    int insideIdxInBlock;\
                    int outsideIdxInBlock;\
                } __attribute__ ((aligned (DefaultStructAlign)));\
                __kernel void FOpenCL__bottomPassPerform(__global unsigned char* leafCellsPtr, size_t leafCellsSize,__global unsigned char* leafCellsUpPtr,\
                    __global unsigned char* containersPtr, size_t containersSize,\
                    __global void* userkernel ){\
                }\
                __kernel void FOpenCL__upwardPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize, __global unsigned char* currentCellsUpPtr,\
                    __global unsigned char* childCellsPtr, size_t childCellsSize, __global unsigned char* childCellsUpPtr,\
                    int idxLevel, __global void* userkernel){\
                }\
                __kernel  void FOpenCL__transferInoutPassPerformMpi(__global unsigned char* currentCellsPtr, size_t currentCellsSize, __global unsigned char* currentCellsDownPtr,\
                    __global unsigned char* externalCellsPtr, size_t externalCellsSize, __global unsigned char* externalCellsUpPtr,\
                    int idxLevel, const __global struct OutOfBlockInteraction* outsideInteractions,\
                    size_t nbOutsideInteractions, __global void* userkernel){\
                }\
                __kernel  void FOpenCL__transferInPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize,\
                    __global unsigned char* currentCellsUpPtr, __global unsigned char* currentCellsDownPtr,\
                    int idxLevel, __global void* userkernel){\
                }\
                __kernel void FOpenCL__transferInoutPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize,\
                    __global unsigned char*  currentCellsUpPtr,\
                    __global unsigned char* externalCellsPtr, size_t externalCellsSize,\
                    __global unsigned char* externalCellsDownPtr,\
                    int idxLevel, int mode, const __global struct OutOfBlockInteraction* outsideInteractions,\
                    size_t nbOutsideInteractions, __global void* userkernel){\
                }\
                __kernel void FOpenCL__downardPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize, __global unsigned char* currentCellsDownPtr,\
                    __global unsigned char* childCellsPtr, size_t childCellsSize, __global unsigned char* childCellsDownPtr,\
                    int idxLevel, __global void* userkernel){\
                }\
                __kernel void FOpenCL__directInoutPassPerformMpi(__global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,\
                    __global unsigned char* externalContainersPtr, size_t externalContainersSize, __global unsigned char* outsideInteractionsCl,\
                    const __global struct OutOfBlockInteraction* outsideInteractions,\
                    size_t nbOutsideInteractions, const int treeHeight, __global void* userkernel){\
                }\
                __kernel void FOpenCL__directInPassPerform(__global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,\
                    const int treeHeight, __global void* userkernel){\
                }\
                __kernel void FOpenCL__directInoutPassPerform(__global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,\
                    __global unsigned char* externalContainersPtr, size_t externalContainersSize, __global unsigned char* externalContainersDownPtr,\
                    const __global struct OutOfBlockInteraction* outsideInteractions,\
                    size_t nbOutsideInteractions, const int treeHeight, __global void* userkernel){\
                }\
                __kernel void FOpenCL__mergePassPerform(__global unsigned char* leafCellsPtr, size_t leafCellsSize, __global unsigned char* leafCellsDownPtr,\
                    __global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,\
                    __global void* userkernel){\
                }";
        return kernelcode;
    }

    void releaseKernelCode(){
    }

    unsigned int getNbDims() const {
        return 0;
    }

    const size_t* getNbGroups(const int /*inSizeInterval*/) const {
        return nullptr;
    }

    const size_t* getGroupSize() const {
        return nullptr;
    }
};

#endif // FEMPTYOPENCLCODE_HPP

