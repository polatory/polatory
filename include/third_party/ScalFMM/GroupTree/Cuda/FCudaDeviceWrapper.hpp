#ifndef FCUDADEVICEWRAPPER_HPP
#define FCUDADEVICEWRAPPER_HPP


#include "../../Utils/FGlobal.hpp"
#include "../Core/FOutOfBlockInteraction.hpp"
#include "FCudaStructParams.hpp"

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
void FCuda__bottomPassCallback(unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsUpPtr,
    unsigned char* containersPtr, std::size_t containersSize,
    CudaKernelClass* kernel, cudaStream_t 	currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
void FCuda__upwardPassCallback(
    unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsUpPtr,
        unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsUpPtr,
     int idxLevel, CudaKernelClass* kernel, cudaStream_t 	currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
void FCuda__transferInoutPassCallbackMpi(
    unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize, unsigned char* externalCellsUpPtr,
    int idxLevel, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int* safeInteractions, int nbSafeInteractions, CudaKernelClass* kernel, cudaStream_t 	currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
void FCuda__transferInPassCallback(
    unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr, unsigned char* currentCellsDownPtr,
    int idxLevel, CudaKernelClass* kernel, cudaStream_t 	currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
void FCuda__transferInoutPassCallback(
    unsigned char* currentCellsPtr, std::size_t currentCellsSize,
    unsigned char* currentCellsUpPtr,
    unsigned char* externalCellsPtr, std::size_t externalCellsSize,
    unsigned char* externalCellsDownPtr,
    int idxLevel, int mode, const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions,
        const int* safeInteractions, int nbSafeInteractions, CudaKernelClass* kernel, cudaStream_t 	currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
void FCuda__downardPassCallback(
    unsigned char* currentCellsPtr, std::size_t currentCellsSize, unsigned char* currentCellsDownPtr,
        unsigned char* childCellsPtr, std::size_t childCellsSize, unsigned char* childCellsDownPtr,
        int idxLevel, CudaKernelClass* kernel, cudaStream_t 	currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#ifdef SCALFMM_USE_MPI
template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
void FCuda__directInoutPassCallbackMpi(
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize,
    const OutOfBlockInteraction* outsideInteractions,
    int nbOutsideInteractions, const int safeOuterInteractions[], const int counterOuterCell,
        const int treeHeight, CudaKernelClass* kernel, cudaStream_t 	currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);
#endif
template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
void FCuda__directInPassCallback(
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    const int treeHeight, CudaKernelClass* kernel, cudaStream_t 	currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
void FCuda__directInoutPassCallback(
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    unsigned char* externalContainersPtr, std::size_t externalContainersSize, unsigned char* externalContainersDownPtr,
    const OutOfBlockInteraction* outsideInteractions, int nbOutsideInteractions,
    const int     safeOuterInteractions[], const int counterOuterCell,
        const OutOfBlockInteraction* insideInteractions,
        const int     safeInnterInteractions[], const int counterInnerCell,
    const int treeHeight, CudaKernelClass* kernel, cudaStream_t 	currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass,
          class CellContainerClass, class ParticleContainerGroupClass, class ParticleGroupClass, class CudaKernelClass>
void FCuda__mergePassCallback(
    unsigned char* leafCellsPtr, std::size_t leafCellsSize, unsigned char* leafCellsDownPtr,
    unsigned char* containersPtr, std::size_t containersSize, unsigned char* containersDownPtr,
    CudaKernelClass* kernel, cudaStream_t 	currentStream,
                                        const dim3 inGridSize, const dim3 inBlocksSize);

template <class CudaKernelClass>
CudaKernelClass* FCuda__BuildCudaKernel(void*);

template <class CudaKernelClass>
void FCuda__ReleaseCudaKernel(CudaKernelClass*);

template <class CudaKernelClass>
dim3 FCuda__GetGridSize(CudaKernelClass* kernel, int intervalSize);

template <class CudaKernelClass>
dim3 FCuda__GetBlockSize(CudaKernelClass* kernel);

#endif
