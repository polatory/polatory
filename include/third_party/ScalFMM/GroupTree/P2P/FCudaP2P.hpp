#ifndef FCUDAP2P_HPP
#define FCUDAP2P_HPP

#include "../Cuda/FCudaGlobal.hpp"
#include "../Cuda/FCudaGroupAttachedLeaf.hpp"
#include "../Cuda/FCudaEmptyCellSymb.hpp"
#include "../Cuda/FCudaCompositeCell.hpp"


#define Min(x,y) ((x)<(y)?(x):(y))
#define Max(x,y) ((x)>(y)?(x):(y))

/**
 * This class defines what should be a Cuda kernel.
 */
template <class FReal>
class FCudaP2P {
protected:
public:

    __device__ void DirectComputation(const FReal& targetX, const FReal& targetY, const FReal& targetZ,const  FReal& targetPhys,
                           FReal& forceX, FReal& forceY,FReal&  forceZ, FReal& potential,
                           const FReal& sourcesX, const FReal& sourcesY, const FReal& sourcesZ, const FReal& sourcesPhys) const {
        FReal dx = sourcesX - targetX;
        FReal dy = sourcesY - targetY;
        FReal dz = sourcesZ - targetZ;

        FReal inv_square_distance = FReal(1.0) / (dx*dx + dy*dy + dz*dz);
        FReal inv_distance = sqrt(inv_square_distance);

        inv_square_distance *= inv_distance;
        inv_square_distance *= targetPhys * sourcesPhys;

        dx *= inv_square_distance;
        dy *= inv_square_distance;
        dz *= inv_square_distance;

        forceX += dx;
        forceY += dy;
        forceZ += dz;
        potential += inv_distance * sourcesPhys;
    }

    static double DSqrt(const double val){
        return sqrt(val);
    }

    static float FSqrt(const float val){
        return sqrtf(val);
    }

    typedef FCudaGroupAttachedLeaf<FReal,1,4,FReal> ContainerClass;
    typedef FCudaCompositeCell<FCudaEmptyCellSymb,int,int> CellClass;

    static const int NB_THREAD_GROUPS = 30; // 2 x 15
    static const int THREAD_GROUP_SIZE = 256;
    static const int SHARED_MEMORY_SIZE = 512;// 49152

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

    __device__ void P2P(const int3& pos,
                        ContainerClass* const  targets, const ContainerClass* const  sources,
                        ContainerClass* const directNeighborsParticles,
                        const int* neighborPositions, const int counter){
        // Compute with other
        P2PRemote(pos, targets, sources, directNeighborsParticles, neighborPositions, counter);
        // Compute inside
        const int nbLoops = (targets->getNbParticles()+blockDim.x-1)/blockDim.x;
        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            const int idxPart = (idxLoop*blockDim.x)+threadIdx.x;
            const bool threadCompute = (idxPart < targets->getNbParticles());

            FReal targetX, targetY, targetZ, targetPhys;
            FReal forceX = 0, forceY = 0, forceZ = 0, potential = 0;

            targetX = (threadCompute? targets->getPositions()[0][idxPart] : 0);
            targetY = (threadCompute? targets->getPositions()[1][idxPart] : 0);
            targetZ = (threadCompute? targets->getPositions()[2][idxPart] : 0);
            targetPhys = (threadCompute? targets->getAttribute(0)[idxPart] : 0);

            for(int idxCopy = 0 ; idxCopy < targets->getNbParticles() ; idxCopy += SHARED_MEMORY_SIZE){
                __shared__ FReal sourcesX[SHARED_MEMORY_SIZE];
                __shared__ FReal sourcesY[SHARED_MEMORY_SIZE];
                __shared__ FReal sourcesZ[SHARED_MEMORY_SIZE];
                __shared__ FReal sourcesPhys[SHARED_MEMORY_SIZE];

                const int nbCopies = Min(SHARED_MEMORY_SIZE, targets->getNbParticles()-idxCopy);
                for(int idx = threadIdx.x ; idx < nbCopies ; idx += blockDim.x){
                    sourcesX[idx] = targets->getPositions()[0][idx+idxCopy];
                    sourcesY[idx] = targets->getPositions()[1][idx+idxCopy];
                    sourcesZ[idx] = targets->getPositions()[2][idx+idxCopy];
                    sourcesPhys[idx] = targets->getAttribute(0)[idx+idxCopy];
                }

                __syncthreads();

                if(threadCompute){
                    int leftCopies = nbCopies;
                    if(idxCopy <= idxPart && idxPart < idxCopy + nbCopies){
                        leftCopies = idxPart - idxCopy;
                    }

                    // Left Part
                    for(int otherIndex = 0; otherIndex < leftCopies - 3; otherIndex += 4) { // unrolling x4
                        DirectComputation(targetX, targetY, targetZ, targetPhys,
                                          forceX, forceY, forceZ, potential,
                                          sourcesX[otherIndex], sourcesY[otherIndex], sourcesZ[otherIndex], sourcesPhys[otherIndex]);
                        DirectComputation(targetX, targetY, targetZ, targetPhys,
                                          forceX, forceY, forceZ, potential,
                                          sourcesX[otherIndex+1], sourcesY[otherIndex+1], sourcesZ[otherIndex+1], sourcesPhys[otherIndex+1]);
                        DirectComputation(targetX, targetY, targetZ, targetPhys,
                                          forceX, forceY, forceZ, potential,
                                          sourcesX[otherIndex+2], sourcesY[otherIndex+2], sourcesZ[otherIndex+2], sourcesPhys[otherIndex+2]);
                        DirectComputation(targetX, targetY, targetZ, targetPhys,
                                          forceX, forceY, forceZ, potential,
                                          sourcesX[otherIndex+3], sourcesY[otherIndex+3], sourcesZ[otherIndex+3], sourcesPhys[otherIndex+3]);
                    }

                    for(int otherIndex = (leftCopies/4) * 4; otherIndex < leftCopies; ++otherIndex) { // if nk%4 is not zero
                        DirectComputation(targetX, targetY, targetZ, targetPhys,
                                          forceX, forceY, forceZ, potential,
                                          sourcesX[otherIndex], sourcesY[otherIndex], sourcesZ[otherIndex], sourcesPhys[otherIndex]);
                    }
                    // Right Part
                    for(int otherIndex = leftCopies+1; otherIndex < nbCopies - 3; otherIndex += 4) { // unrolling x4
                        DirectComputation(targetX, targetY, targetZ, targetPhys,
                                          forceX, forceY, forceZ, potential,
                                          sourcesX[otherIndex], sourcesY[otherIndex], sourcesZ[otherIndex], sourcesPhys[otherIndex]);
                        DirectComputation(targetX, targetY, targetZ, targetPhys,
                                          forceX, forceY, forceZ, potential,
                                          sourcesX[otherIndex+1], sourcesY[otherIndex+1], sourcesZ[otherIndex+1], sourcesPhys[otherIndex+1]);
                        DirectComputation(targetX, targetY, targetZ, targetPhys,
                                          forceX, forceY, forceZ, potential,
                                          sourcesX[otherIndex+2], sourcesY[otherIndex+2], sourcesZ[otherIndex+2], sourcesPhys[otherIndex+2]);
                        DirectComputation(targetX, targetY, targetZ, targetPhys,
                                          forceX, forceY, forceZ, potential,
                                          sourcesX[otherIndex+3], sourcesY[otherIndex+3], sourcesZ[otherIndex+3], sourcesPhys[otherIndex+3]);
                    }

                    for(int otherIndex = leftCopies+1 + ((nbCopies-(leftCopies+1))/4)*4 ; otherIndex < nbCopies; ++otherIndex) { // if nk%4 is not zero
                        DirectComputation(targetX, targetY, targetZ, targetPhys,
                                          forceX, forceY, forceZ, potential,
                                          sourcesX[otherIndex], sourcesY[otherIndex], sourcesZ[otherIndex], sourcesPhys[otherIndex]);
                    }
                }

                __syncthreads();
            }

            if( threadCompute ){
                targets->getAttribute(1)[idxPart] += potential;
                targets->getAttribute(2)[idxPart] += forceX;
                targets->getAttribute(3)[idxPart] += forceY;
                targets->getAttribute(4)[idxPart] += forceZ;
            }

            __syncthreads();
        }
    }

    __device__ void P2PRemote(const int3& ,
                              ContainerClass* const  targets, const ContainerClass* const  /*sources*/,
                              ContainerClass* const directNeighborsParticles,
                              const int* /*neighborsPositions*/, const int counter){
        for(int idxNeigh = 0 ; idxNeigh < counter ; ++idxNeigh){

            const int nbLoops = (targets->getNbParticles()+blockDim.x-1)/blockDim.x;
            for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
                const int idxPart = (idxLoop*blockDim.x)+threadIdx.x;
                const bool threadCompute = (idxPart < targets->getNbParticles());

                FReal targetX, targetY, targetZ, targetPhys;
                FReal forceX = 0, forceY = 0, forceZ = 0, potential = 0;

                targetX = (threadCompute? targets->getPositions()[0][idxPart] : 0);
                targetY = (threadCompute? targets->getPositions()[1][idxPart] : 0);
                targetZ = (threadCompute? targets->getPositions()[2][idxPart] : 0);
                targetPhys = (threadCompute? targets->getAttribute(0)[idxPart] : 0);

                for(int idxCopy = 0 ; idxCopy < directNeighborsParticles[idxNeigh].getNbParticles() ; idxCopy += SHARED_MEMORY_SIZE){
                    __shared__ FReal sourcesX[SHARED_MEMORY_SIZE];
                    __shared__ FReal sourcesY[SHARED_MEMORY_SIZE];
                    __shared__ FReal sourcesZ[SHARED_MEMORY_SIZE];
                    __shared__ FReal sourcesPhys[SHARED_MEMORY_SIZE];

                    const int nbCopies = Min(SHARED_MEMORY_SIZE, directNeighborsParticles[idxNeigh].getNbParticles()-idxCopy);
                    for(int idx = threadIdx.x ; idx < nbCopies ; idx += blockDim.x){
                        sourcesX[idx] = directNeighborsParticles[idxNeigh].getPositions()[0][idx+idxCopy];
                        sourcesY[idx] = directNeighborsParticles[idxNeigh].getPositions()[1][idx+idxCopy];
                        sourcesZ[idx] = directNeighborsParticles[idxNeigh].getPositions()[2][idx+idxCopy];
                        sourcesPhys[idx] = directNeighborsParticles[idxNeigh].getAttribute(0)[idx+idxCopy];
                    }

                    __syncthreads();

                    if(threadCompute){
                        for(int otherIndex = 0; otherIndex < nbCopies - 3; otherIndex += 4) { // unrolling x4
                            DirectComputation(targetX, targetY, targetZ, targetPhys,
                                              forceX, forceY, forceZ, potential,
                                              sourcesX[otherIndex], sourcesY[otherIndex], sourcesZ[otherIndex], sourcesPhys[otherIndex]);
                            DirectComputation(targetX, targetY, targetZ, targetPhys,
                                              forceX, forceY, forceZ, potential,
                                              sourcesX[otherIndex+1], sourcesY[otherIndex+1], sourcesZ[otherIndex+1], sourcesPhys[otherIndex+1]);
                            DirectComputation(targetX, targetY, targetZ, targetPhys,
                                              forceX, forceY, forceZ, potential,
                                              sourcesX[otherIndex+2], sourcesY[otherIndex+2], sourcesZ[otherIndex+2], sourcesPhys[otherIndex+2]);
                            DirectComputation(targetX, targetY, targetZ, targetPhys,
                                              forceX, forceY, forceZ, potential,
                                              sourcesX[otherIndex+3], sourcesY[otherIndex+3], sourcesZ[otherIndex+3], sourcesPhys[otherIndex+3]);
                        }

                        for(int otherIndex = (nbCopies/4) * 4; otherIndex < nbCopies; ++otherIndex) { // if nk%4 is not zero
                            DirectComputation(targetX, targetY, targetZ, targetPhys,
                                              forceX, forceY, forceZ, potential,
                                              sourcesX[otherIndex], sourcesY[otherIndex], sourcesZ[otherIndex], sourcesPhys[otherIndex]);
                        }
                    }

                    __syncthreads();
                }

                if( threadCompute ){
                    targets->getAttribute(1)[idxPart] += potential;
                    targets->getAttribute(2)[idxPart] += forceX;
                    targets->getAttribute(3)[idxPart] += forceY;
                    targets->getAttribute(4)[idxPart] += forceZ;
                }


                __syncthreads();
            }
        }
    }

    __device__ void P2POuter(const int3& ,
                             ContainerClass* const  targets,
                             ContainerClass* const directNeighborsParticles,
                             const int* /*neighborsPositions*/, const int counter){
        for(int idxNeigh = 0 ; idxNeigh < counter ; ++idxNeigh){

            const int nbLoops = (targets->getNbParticles()+blockDim.x-1)/blockDim.x;
            for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
                const int idxPart = (idxLoop*blockDim.x)+threadIdx.x;
                const bool threadCompute = (idxPart < targets->getNbParticles());

                FReal targetX, targetY, targetZ, targetPhys;
                FReal forceX = 0, forceY = 0, forceZ = 0, potential = 0;

                targetX = (threadCompute? targets->getPositions()[0][idxPart] : 0);
                targetY = (threadCompute? targets->getPositions()[1][idxPart] : 0);
                targetZ = (threadCompute? targets->getPositions()[2][idxPart] : 0);
                targetPhys = (threadCompute? targets->getAttribute(0)[idxPart] : 0);

                for(int idxCopy = 0 ; idxCopy < directNeighborsParticles[idxNeigh].getNbParticles() ; idxCopy += SHARED_MEMORY_SIZE){
                    __shared__ FReal sourcesX[SHARED_MEMORY_SIZE];
                    __shared__ FReal sourcesY[SHARED_MEMORY_SIZE];
                    __shared__ FReal sourcesZ[SHARED_MEMORY_SIZE];
                    __shared__ FReal sourcesPhys[SHARED_MEMORY_SIZE];

                    const int nbCopies = Min(SHARED_MEMORY_SIZE, directNeighborsParticles[idxNeigh].getNbParticles()-idxCopy);
                    for(int idx = threadIdx.x ; idx < nbCopies ; idx += blockDim.x){
                        sourcesX[idx] = directNeighborsParticles[idxNeigh].getPositions()[0][idx+idxCopy];
                        sourcesY[idx] = directNeighborsParticles[idxNeigh].getPositions()[1][idx+idxCopy];
                        sourcesZ[idx] = directNeighborsParticles[idxNeigh].getPositions()[2][idx+idxCopy];
                        sourcesPhys[idx] = directNeighborsParticles[idxNeigh].getAttribute(0)[idx+idxCopy];
                    }

                    __syncthreads();

                    if(threadCompute){
                        for(int otherIndex = 0; otherIndex < nbCopies - 3; otherIndex += 4) { // unrolling x4
                            DirectComputation(targetX, targetY, targetZ, targetPhys,
                                              forceX, forceY, forceZ, potential,
                                              sourcesX[otherIndex], sourcesY[otherIndex], sourcesZ[otherIndex], sourcesPhys[otherIndex]);
                            DirectComputation(targetX, targetY, targetZ, targetPhys,
                                              forceX, forceY, forceZ, potential,
                                              sourcesX[otherIndex+1], sourcesY[otherIndex+1], sourcesZ[otherIndex+1], sourcesPhys[otherIndex+1]);
                            DirectComputation(targetX, targetY, targetZ, targetPhys,
                                              forceX, forceY, forceZ, potential,
                                              sourcesX[otherIndex+2], sourcesY[otherIndex+2], sourcesZ[otherIndex+2], sourcesPhys[otherIndex+2]);
                            DirectComputation(targetX, targetY, targetZ, targetPhys,
                                              forceX, forceY, forceZ, potential,
                                              sourcesX[otherIndex+3], sourcesY[otherIndex+3], sourcesZ[otherIndex+3], sourcesPhys[otherIndex+3]);
                        }

                        for(int otherIndex = (nbCopies/4) * 4; otherIndex < nbCopies; ++otherIndex) { // if nk%4 is not zero
                            DirectComputation(targetX, targetY, targetZ, targetPhys,
                                              forceX, forceY, forceZ, potential,
                                              sourcesX[otherIndex], sourcesY[otherIndex], sourcesZ[otherIndex], sourcesPhys[otherIndex]);
                        }
                    }

                    __syncthreads();
                }

                if( threadCompute ){
                    targets->getAttribute(1)[idxPart] += potential;
                    targets->getAttribute(2)[idxPart] += forceX;
                    targets->getAttribute(3)[idxPart] += forceY;
                    targets->getAttribute(4)[idxPart] += forceZ;
                }

                __syncthreads();
            }
        }
    }

    __host__ static FCudaP2P* InitKernelKernel(void*){
        return nullptr;
    }

    __host__ static void ReleaseKernel(FCudaP2P* /*todealloc*/){
        // nothing to do
    }

    __host__ static dim3 GetGridSize(const int /*intervalSize*/){
        return NB_THREAD_GROUPS;
    }

    __host__ static dim3 GetBlocksSize(){
        return THREAD_GROUP_SIZE;
    }
};

#endif // FCUDAP2P_HPP

