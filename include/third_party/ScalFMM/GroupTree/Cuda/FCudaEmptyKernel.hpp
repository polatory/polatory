#ifndef FCUDAEMPTYKERNEL_HPP
#define FCUDAEMPTYKERNEL_HPP

#include "FCudaGlobal.hpp"
#include "FCudaGroupAttachedLeaf.hpp"
#include "FCudaEmptyCellSymb.hpp"
#include "FCudaCompositeCell.hpp"

/**
 * This class defines what should be a Cuda kernel.
 */
template <class FReal>
class FCudaEmptyKernel {
protected:
public:
    typedef FCudaGroupAttachedLeaf<FReal,0,0,int> ContainerClass;
    typedef FCudaCompositeCell<FCudaEmptyCellSymb,int,int> CellClass;

    __device__ void P2M(CellClass /*pole*/, const ContainerClass* const /*particles*/) {
    }

    __device__ void M2M(CellClass  /*pole*/, const CellClass  /*child*/[8], const int /*level*/) {
    }

    __device__ void M2L(CellClass  /*pole*/, const CellClass* /*distantNeighbors*/,
                        const int* /*neighPositions*/,
        const int /*size*/, const int /*level*/) {
    }

    __device__ void L2L(const CellClass  /*local*/, CellClass  /*child*/[8], const int /*level*/) {
    }

    __device__ void L2P(const CellClass  /*local*/, ContainerClass*const /*particles*/){
    }

    __device__ void P2P(const int3& ,
                 ContainerClass* const  /*targets*/, const ContainerClass* const  /*sources*/,
                 ContainerClass* const /*directNeighborsParticles*/,
                        const int* /*neighborPositions*/, const int ){
    }

    __device__ void P2POuter(const int3& ,
                 ContainerClass* const  /*targets*/,
                 ContainerClass* const /*directNeighborsParticles*/,
                              const int* /*neighborPositions*/,const int ){
    }

    __device__ void P2PRemote(const int3& ,
                 ContainerClass* const  /*targets*/, const ContainerClass* const  /*sources*/,
                 ContainerClass* const /*directNeighborsParticles*/,
                              const int* /*neighborPositions*/,const int ){
    }

    __host__ static FCudaEmptyKernel* InitKernelKernel(void*){
        return nullptr;
    }

    __host__ static void ReleaseKernel(FCudaEmptyKernel* /*todealloc*/){
        // nothing to do
    }

    __host__ static dim3 GetGridSize(const int /*intervalSize*/){
        return 0;
    }

    __host__ static dim3 GetBlocksSize(){
        return 0;
    }
};

#endif // FCUDAEMPTYKERNEL_HPP

