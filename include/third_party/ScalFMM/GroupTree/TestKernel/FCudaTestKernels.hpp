#ifndef FCUDATESTKERNEL_HPP
#define FCUDATESTKERNEL_HPP

#include "../Cuda/FCudaGlobal.hpp"
#include "../Cuda/FCudaGroupAttachedLeaf.hpp"
#include "../Cuda/FCudaCompositeCell.hpp"
// We need to describe this cell
#include "FTestCellPOD.hpp"

template <class FReal>
class FTestCudaKernels {
public:
    typedef FCudaCompositeCell<FTestCellPODCore, FTestCellPODData, FTestCellPODData> CellClass;
    typedef FCudaGroupAttachedLeaf<FReal, 0, 1, long long int>  ContainerClass;

    /** Before upward */
    __device__ void P2M(CellClass pole, const ContainerClass* const particles) {
        // the pole represents all particles under
        if(threadIdx.x == 0){
            *pole.up += particles->getNbParticles();
        }
    }

    /** During upward */
    __device__ void M2M(CellClass  pole, const CellClass  child[8], const int /*level*/) {
        if(threadIdx.x == 0) {
            // A parent represents the sum of the child
            for(int idx = 0 ; idx < 8 ; ++idx){
                if(child[idx].symb){
                    *pole.up += *child[idx].up;
                }
            }
        }
    }

    /** Before Downward */
    __device__ void M2L(CellClass  local, const CellClass* distantNeighbors,
                const int* /*neighPositions*/, const int size, const int /*level*/) {
        if(threadIdx.x == 0) {
            // The pole is impacted by what represent other poles
            for(int idx = 0 ; idx < size ; ++idx){
                *local.down += *distantNeighbors[idx].up;
            }
        }
    }

    /** During Downward */
    __device__ void L2L(const CellClass local, CellClass  child[8], const int /*level*/) {
        if(threadIdx.x == 0) {
            // Each child is impacted by the father
            for(int idx = 0 ; idx < 8 ; ++idx){
                if(child[idx].symb){
                    *child[idx].down += *local.down;
                }
            }
        }
    }

    /** After Downward */
    __device__ void L2P(const CellClass local, ContainerClass*const particles){
        if(threadIdx.x == 0) {
            // The particles is impacted by the parent cell
            long long int*const particlesAttributes = particles->template getAttribute<0>();
            for(FSize idxPart = 0 ; idxPart < particles->getNbParticles() ; ++idxPart){
                particlesAttributes[idxPart] += *local.down;
            }
        }
    }


    /** After Downward */
    __device__ void P2P(const int3& ,
                        ContainerClass* const  targets, const ContainerClass* const  sources,
                        ContainerClass* const directNeighborsParticles,
                        const int* /*neighborPositions*/,
                        const int counter){
        if(threadIdx.x == 0) {
            // Each particles targeted is impacted by the particles sources
            long long int inc = sources->getNbParticles();
            if(targets == sources){
                inc -= 1;
            }
            for(int idx = 0 ; idx < counter ; ++idx){
                inc += directNeighborsParticles[idx].getNbParticles();
            }

            long long int*const particlesAttributes = targets->template getAttribute<0>();
            for(FSize idxPart = 0 ; idxPart < targets->getNbParticles() ; ++idxPart){
                particlesAttributes[idxPart] += inc;
            }
        }
    }

    /** After Downward */
    __device__ void P2PRemote(const int3& ,
                              ContainerClass* const  targets,
                              const ContainerClass* const  sources,
                              ContainerClass* const directNeighborsParticles,
                              const int* /*neighborPositions*/,
                              const int counter){
        if(threadIdx.x == 0) {
            // Each particles targeted is impacted by the particles sources
            long long int inc = 0;
            for(int idx = 0 ; idx < counter ; ++idx){
                inc += directNeighborsParticles[idx].getNbParticles();
            }

            long long int*const particlesAttributes = targets->template getAttribute<0>();
            for(FSize idxPart = 0 ; idxPart < targets->getNbParticles() ; ++idxPart){
                particlesAttributes[idxPart] += inc;
            }
        }
    }

    __device__ void P2POuter(const int3& ,
                             ContainerClass* const  targets,
                             ContainerClass* const directNeighborsParticles,
                             const int* /*neighborPositions*/,
                             const int counter){
        if(threadIdx.x == 0) {
            // Each particles targeted is impacted by the particles sources
            long long int inc = 0;
            for(int idx = 0 ; idx < counter ; ++idx){
                inc += directNeighborsParticles[idx].getNbParticles();
            }

            long long int*const particlesAttributes = targets->template getAttribute<0>();
            for(FSize idxPart = 0 ; idxPart < targets->getNbParticles() ; ++idxPart){
                particlesAttributes[idxPart] += inc;
            }
        }
    }

    __host__ static FTestCudaKernels* InitKernelKernel(void*){
        return nullptr;
    }

    __host__ static void ReleaseKernel(FTestCudaKernels* /*todealloc*/){
        // nothing to do
    }

    __host__ static dim3 GetGridSize(const int /*intervalSize*/){
        return 1;
    }

    __host__ static dim3 GetBlocksSize(){
        return 1;
    }
};


#endif // FCUDATESTKERNEL_HPP

