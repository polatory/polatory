// See LICENCE file at project root
#ifndef FPARTITIONSMAPPING_HPP
#define FPARTITIONSMAPPING_HPP

#include "Utils/FGlobal.hpp"
#include "Utils/FMpi.hpp"
#include "Containers/FVector.hpp"
#include "FLeafBalance.hpp"
#include "Files/FMpiTreeBuilder.hpp"

template <class FReal>
class FPartitionsMapping {
protected:
    FMpi::FComm comm;

    //! The number of particles from the initial decomposition
    FSize nbParticlesInitial;
    //! The number of particles from the scalfmm decomposition
    FSize nbParticlesWorking;

    std::unique_ptr<FSize[]> nbParticlesSentToOthers;
    std::unique_ptr<FSize[]> offsetNbParticlesSentToOthers;
    std::unique_ptr<FSize[]> nbParticlesRecvFromOthers;
    std::unique_ptr<FSize[]> offsetNbParticlesRecvFromOthers;

    std::unique_ptr<FSize[]> mappingToOthers;


public:
    FPartitionsMapping(const FMpi::FComm& inComm)
        : comm(inComm), nbParticlesInitial(0), nbParticlesWorking(0) {
    }

    void setComm(const FMpi::FComm& inComm){
        comm = inComm;
    }

    template< int NbPhysicalValuesPerPart>
    struct TestParticle{
        FPoint<FReal> position;
        std::array<FReal, NbPhysicalValuesPerPart> physicalValues;
        FSize localIndex;
        int initialProcOwner;

        const FPoint<FReal>& getPosition() const {
            return position;
        }
    };

    template< int NbPhysicalValuesPerPart, class FillerClass>
    FVector<TestParticle<NbPhysicalValuesPerPart>> distributeParticles(const FSize inNbParticles,
                                                                       const FPoint<FReal>& centerOfBox, const FReal boxWidth,
                             const int TreeHeight, FillerClass filler){
        nbParticlesInitial = inNbParticles;

        ////////////////////////////////////////////////////////

        std::unique_ptr<TestParticle<NbPhysicalValuesPerPart>[]> initialParticles(new TestParticle<NbPhysicalValuesPerPart>[inNbParticles]);

        // Create the array to distribute
        for(int idxPart = 0 ; idxPart < nbParticlesInitial ; ++idxPart){
            filler(idxPart, &initialParticles[idxPart].position, &initialParticles[idxPart].physicalValues);
            initialParticles[idxPart].localIndex = idxPart;
            initialParticles[idxPart].initialProcOwner = comm.processId();
        }

        FVector<TestParticle<NbPhysicalValuesPerPart>> finalParticles;
        FLeafBalance balancer;
        FMpiTreeBuilder< FReal,TestParticle<NbPhysicalValuesPerPart> >::DistributeArrayToContainer(comm,initialParticles.get(),
                                                                    nbParticlesInitial,
                                                                    centerOfBox,
                                                                    boxWidth,
                                                                    TreeHeight,
                                                                    &finalParticles, &balancer);

        FQuickSort<TestParticle<NbPhysicalValuesPerPart>,FSize>::QsOmp(finalParticles.data(), finalParticles.getSize(),
                          [](const TestParticle<NbPhysicalValuesPerPart>& p1,
                          const TestParticle<NbPhysicalValuesPerPart>& p2){
            return p1.initialProcOwner < p2.initialProcOwner
                    || (p1.initialProcOwner == p2.initialProcOwner
                    && p1.localIndex < p2.localIndex);
        });

        ////////////////////////////////////////////////////////

        nbParticlesWorking = finalParticles.getSize();

        nbParticlesRecvFromOthers.reset(new FSize[comm.processCount()]);
        memset(nbParticlesRecvFromOthers.get(), 0 , sizeof(FSize)*comm.processCount());

        for(FSize idxPart = 0 ; idxPart < finalParticles.getSize() ; ++idxPart){
            // Count the particles received from each proc
            nbParticlesRecvFromOthers[finalParticles[idxPart].initialProcOwner] += 1;

        }

        offsetNbParticlesRecvFromOthers.reset(new FSize[comm.processCount()+1]);
        offsetNbParticlesRecvFromOthers[0] = 0;

        for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
            offsetNbParticlesRecvFromOthers[idxProc+1] = offsetNbParticlesRecvFromOthers[idxProc]
                                    + nbParticlesRecvFromOthers[idxProc];
        }

        ////////////////////////////////////////////////////////

        std::unique_ptr<FSize[]> nbParticlesRecvFromOthersAllAll(new FSize[comm.processCount()*comm.processCount()]);
        // Exchange how many each proc receive from aother
        FMpi::MpiAssert( MPI_Allgather( nbParticlesRecvFromOthers.get(), comm.processCount(), FMpi::GetType(FSize()),
                                        nbParticlesRecvFromOthersAllAll.get(), comm.processCount(),
                                        FMpi::GetType(FSize()), comm.getComm()),  __LINE__ );

        ////////////////////////////////////////////////////////

        nbParticlesSentToOthers.reset(new FSize[comm.processCount()]);
        FSize checkerSent = 0;
        offsetNbParticlesSentToOthers.reset(new FSize[comm.processCount()+1]);
        offsetNbParticlesSentToOthers[0] = 0;

        for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
            nbParticlesSentToOthers[idxProc] = nbParticlesRecvFromOthersAllAll[comm.processCount()*idxProc + comm.processId()];
            checkerSent += nbParticlesSentToOthers[idxProc];
            offsetNbParticlesSentToOthers[idxProc+1] = offsetNbParticlesSentToOthers[idxProc]
                                                        + nbParticlesSentToOthers[idxProc];
        }
        // I must have send what I was owning at the beginning
        FAssertLF(checkerSent == nbParticlesInitial);

        ////////////////////////////////////////////////////////

        std::unique_ptr<FSize[]> localIdsRecvOrdered(new FSize[nbParticlesWorking]);

        // We list the local id in order of the different proc
        for(FSize idxPart = 0 ; idxPart < finalParticles.getSize() ; ++idxPart){
            const int procOwner = finalParticles[idxPart].initialProcOwner;
            localIdsRecvOrdered[idxPart] = finalParticles[idxPart].localIndex;
            FAssertLF(offsetNbParticlesRecvFromOthers[procOwner] <= idxPart
                      && idxPart < offsetNbParticlesRecvFromOthers[procOwner+1]);
        }

        ////////////////////////////////////////////////////////

        std::unique_ptr<FSize[]> localIdsSendOrdered(new FSize[nbParticlesInitial]);
        std::unique_ptr<MPI_Request[]> requests(new MPI_Request[comm.processCount()*2]);

        int iterRequest = 0;
        for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
            if(idxProc == comm.processId()){
                FAssertLF(nbParticlesRecvFromOthers[idxProc] == nbParticlesSentToOthers[idxProc]);
                memcpy(&localIdsSendOrdered[offsetNbParticlesSentToOthers[idxProc]],
                       &localIdsRecvOrdered[offsetNbParticlesRecvFromOthers[idxProc]],
                       sizeof(FSize)*nbParticlesRecvFromOthers[idxProc]);
            }
            else{
                const FSize nbRecvFromProc = nbParticlesRecvFromOthers[idxProc];
                if(nbRecvFromProc){
                    FMpi::MpiAssert( MPI_Isend(&localIdsRecvOrdered[offsetNbParticlesRecvFromOthers[idxProc]],
                              int(nbRecvFromProc),
                              FMpi::GetType(FSize()), idxProc,
                              99, comm.getComm(), &requests[iterRequest++]), __LINE__ );
                }
                const FSize nbSendToProc = nbParticlesSentToOthers[idxProc];
                if(nbSendToProc){
                    FMpi::MpiAssert( MPI_Irecv(&localIdsSendOrdered[offsetNbParticlesSentToOthers[idxProc]],
                              int(nbSendToProc),
                              FMpi::GetType(FSize()), idxProc,
                              99, comm.getComm(), &requests[iterRequest++]), __LINE__  );
                }
            }
        }

        FMpi::MpiAssert( MPI_Waitall( iterRequest, requests.get(), MPI_STATUSES_IGNORE), __LINE__  );

        ////////////////////////////////////////////////////////

        mappingToOthers.reset(new FSize[nbParticlesInitial]);
        for(FSize idxPart = 0; idxPart < nbParticlesInitial ; ++idxPart){
            mappingToOthers[localIdsSendOrdered[idxPart]] = idxPart;
        }

        return std::move(finalParticles);
    }

    ////////////////////////////////////////////////////////

    // physicalValues must be of size nbParticlesInitial
    template< int NbPhysicalValuesPerPart>
    std::unique_ptr<std::array<FReal, NbPhysicalValuesPerPart>[]> distributeData(
            const std::array<FReal, NbPhysicalValuesPerPart> physicalValues[]){
        std::unique_ptr<std::array<FReal, NbPhysicalValuesPerPart>[]> physicalValuesRorder(
                    new std::array<FReal, NbPhysicalValuesPerPart>[nbParticlesInitial]);

        for(FSize idxPart = 0; idxPart < nbParticlesInitial ; ++idxPart){
           physicalValuesRorder[mappingToOthers[idxPart]] = physicalValues[idxPart];
        }

        // Allocate the array to store the physical values of my working interval
        std::unique_ptr<std::array<FReal, NbPhysicalValuesPerPart>[]> recvPhysicalValues(new std::array<FReal, NbPhysicalValuesPerPart>[nbParticlesWorking]);

        std::unique_ptr<MPI_Request[]> requests(new MPI_Request[comm.processCount()*2]);
        int iterRequest = 0;
        for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
            if(idxProc == comm.processId()){
                FAssertLF(nbParticlesRecvFromOthers[idxProc] == nbParticlesSentToOthers[idxProc]);
                memcpy(&recvPhysicalValues[offsetNbParticlesRecvFromOthers[idxProc]],
                       &physicalValuesRorder[offsetNbParticlesSentToOthers[idxProc]],
                       sizeof(std::array<FReal, NbPhysicalValuesPerPart>)*nbParticlesRecvFromOthers[idxProc]);
            }
            else{
                const FSize nbSendToProc = nbParticlesSentToOthers[idxProc];
                if(nbSendToProc){
                    FMpi::MpiAssert( MPI_Isend(
                              const_cast<std::array<FReal, NbPhysicalValuesPerPart>*>(&physicalValuesRorder[offsetNbParticlesSentToOthers[idxProc]]),
                              int(nbSendToProc*sizeof(std::array<FReal, NbPhysicalValuesPerPart>)),
                              MPI_BYTE, idxProc,
                              2222, comm.getComm(), &requests[iterRequest++]), __LINE__  );;
                }
                const FSize nbRecvFromProc = nbParticlesRecvFromOthers[idxProc];
                if(nbRecvFromProc){
                    FMpi::MpiAssert( MPI_Irecv(
                              (void*)&recvPhysicalValues[offsetNbParticlesRecvFromOthers[idxProc]],
                              int(nbRecvFromProc*sizeof(std::array<FReal, NbPhysicalValuesPerPart>)),
                              MPI_INT, idxProc,
                              2222, comm.getComm(), &requests[iterRequest++]), __LINE__  );;
                }
            }
        }

        FMpi::MpiAssert( MPI_Waitall( iterRequest, requests.get(), MPI_STATUSES_IGNORE), __LINE__  );

        return std::move(recvPhysicalValues);
    }

    ////////////////////////////////////////////////////////

    // resValues must be of size nbParticlesWorking
    template< int NbResValuesPerPart>
    std::unique_ptr<std::array<FReal, NbResValuesPerPart>[]> getResultingData(
            const std::array<FReal, NbResValuesPerPart> resValues[]){
        // First allocate the array to store the result
        std::unique_ptr<std::array<FReal, NbResValuesPerPart>[]> recvPhysicalValues(
                    new std::array<FReal, NbResValuesPerPart>[nbParticlesInitial]);

        std::unique_ptr<MPI_Request[]> requests(new MPI_Request[comm.processCount()*2]);
        int iterRequest = 0;
        for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
            if(idxProc == comm.processId()){
                FAssertLF(nbParticlesRecvFromOthers[idxProc] == nbParticlesSentToOthers[idxProc]);
                memcpy(&recvPhysicalValues[offsetNbParticlesSentToOthers[idxProc]],
                       &resValues[offsetNbParticlesRecvFromOthers[idxProc]],
                       sizeof(std::array<FReal, NbResValuesPerPart>)*nbParticlesRecvFromOthers[idxProc]);
            }
            else{
                // I originaly receive nbRecvFromProc, so I should
                // send nbRecvFromProc back to the real owner
                const FSize nbRecvFromProc = nbParticlesRecvFromOthers[idxProc];
                if(nbRecvFromProc){
                    FMpi::MpiAssert( MPI_Isend(
                              const_cast<std::array<FReal, NbResValuesPerPart>*>(&resValues[offsetNbParticlesRecvFromOthers[idxProc]]),
                              int(nbRecvFromProc*sizeof(std::array<FReal, NbResValuesPerPart>)),
                              MPI_BYTE, idxProc,
                              1111, comm.getComm(), &requests[iterRequest++]), __LINE__  );;
                }
                // I sent nbSendToProc to idxProc,
                // so I should receive nbSendToProc in my interval
                const FSize nbSendToProc = nbParticlesSentToOthers[idxProc];
                if(nbSendToProc){
                    FMpi::MpiAssert( MPI_Irecv(
                              &recvPhysicalValues[offsetNbParticlesSentToOthers[idxProc]],
                              int(nbSendToProc*sizeof(std::array<FReal, NbResValuesPerPart>)),
                              MPI_BYTE, idxProc,
                              1111, comm.getComm(), &requests[iterRequest++]), __LINE__  );;
                }
            }
        }

        FMpi::MpiAssert( MPI_Waitall( iterRequest, requests.get(), MPI_STATUSES_IGNORE), __LINE__  );


        std::unique_ptr<std::array<FReal, NbResValuesPerPart>[]> recvPhysicalValuesOrder(
                    new std::array<FReal, NbResValuesPerPart>[nbParticlesInitial]);

        for(FSize idxPart = 0; idxPart < nbParticlesInitial ; ++idxPart){
           recvPhysicalValuesOrder[idxPart] = recvPhysicalValues[mappingToOthers[idxPart]];
        }

        return std::move(recvPhysicalValuesOrder);
    }

    ////////////////////////////////////////////////////////

    FSize getNbParticlesWorking() const{
        return nbParticlesWorking;
    }

    FSize getMappingResultToLocal(const FSize inIdx) const{
        return mappingToOthers[inIdx];
    }

};

#endif
