/** This file contains the prototype for a kernel in opencl */
// @SCALFMM_PRIVATE

typedef long long int MortonIndex;

#define DefaultStructAlign ___DefaultStructAlign___

struct OutOfBlockInteraction{
    MortonIndex outIndex;
    MortonIndex insideIndex;
    int relativeOutPosition;
} __attribute__ ((aligned (DefaultStructAlign)));
struct Uptr9{
    __global unsigned char* ptrs[9];
} __attribute__ ((aligned (DefaultStructAlign)));
struct size_t9{
    size_t v[9];
}__attribute__ ((aligned (DefaultStructAlign)));
__kernel void FOpenCL__bottomPassPerform(__global unsigned char* leafCellsPtr, size_t leafCellsSize,__global unsigned char* leafCellsUpPtr,
                                         __global unsigned char* containersPtr, size_t containersSize,
                                         __global void* userkernel ){
}
__kernel void FOpenCL__upwardPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize, __global unsigned char* currentCellsUpPtr,
                                  struct Uptr9 subCellGroupsPtr, struct size_t9 subCellGroupsSize, struct Uptr9 subCellGroupsUpPtr,
                                  int nbSubCellGroups, int idxLevel, __global void* userkernel){
}
__kernel  void FOpenCL__transferInoutPassPerformMpi(__global unsigned char* currentCellsPtr, size_t currentCellsSize, __global unsigned char* currentCellsDownPtr,
                                             __global unsigned char* externalCellsPtr, size_t externalCellsSize, __global unsigned char* externalCellsUpPtr,
                                             int idxLevel, const __global struct OutOfBlockInteraction* outsideInteractions,
                                             size_t nbOutsideInteractions, __global void* userkernel){
}
__kernel  void FOpenCL__transferInPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize,
                                        __global unsigned char* currentCellsUpPtr, __global unsigned char* currentCellsDownPtr,
                                       int idxLevel, __global void* userkernel){
}
__kernel void FOpenCL__transferInoutPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize,
                                         __global unsigned char*  currentCellsUpPtr, __global unsigned char*  currentCellsDownPtr,
                                         __global unsigned char* externalCellsPtr, size_t externalCellsSize,
                                         __global unsigned char* externalCellsUpPtr, __global unsigned char* externalCellsDownPtr,
                                         int idxLevel, const __global struct OutOfBlockInteraction* outsideInteractions,
                                         size_t nbOutsideInteractions, __global void* userkernel){
}
__kernel void FOpenCL__downardPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize, __global unsigned char* currentCellsDownPtr,
                                   struct Uptr9 subCellGroupsPtr, struct size_t9 subCellGroupsSize, struct Uptr9 subCellGroupsDownPtr,
                                   int nbSubCellGroups, int idxLevel, __global void* userkernel){
}
__kernel void FOpenCL__directInoutPassPerformMpi(__global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,
                                          __global unsigned char* externalContainersPtr, size_t externalContainersSize, __global unsigned char* outsideInteractionsCl,
                                          const __global struct OutOfBlockInteraction* outsideInteractions,
                                          size_t nbOutsideInteractions, const int treeHeight, __global void* userkernel){
}
__kernel void FOpenCL__directInPassPerform(__global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,
                                    const int treeHeight, __global void* userkernel){
}
__kernel void FOpenCL__directInoutPassPerform(__global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,
                                       __global unsigned char* externalContainersPtr, size_t externalContainersSize, __global unsigned char* externalContainersDownPtr,
                                       const __global struct OutOfBlockInteraction* outsideInteractions,
                                       size_t nbOutsideInteractions, const int treeHeight, __global void* userkernel){
}
__kernel void FOpenCL__mergePassPerform(__global unsigned char* leafCellsPtr, size_t leafCellsSize, __global unsigned char* leafCellsDownPtr,
                                 __global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,
                                 __global void* userkernel){
}

