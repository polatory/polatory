// See LICENCE file at project root
#ifndef FMPITREEBUILDER_H
#define FMPITREEBUILDER_H

#include "../Utils/FMpi.hpp"
#include "../Utils/FQuickSortMpi.hpp"
#include "../Utils/FBitonicSort.hpp"
#include "../Utils/FTic.hpp"
#include "../Utils/FEnv.hpp"

#include "../Utils/FMemUtils.hpp"

#include "../Containers/FVector.hpp"

#include "../Utils/FLeafBalance.hpp"
#include "../Utils/FEqualize.hpp"

#include "../Containers/FCoordinateComputer.hpp"

/**
 * This class manage the loading of particles for the mpi version.
 * It work in several steps.
 * First it load the data from a file or an array and sort them amon the MPI processes.
 * Then, it carrefully manage if a leaf is shared by multiple processes.
 * Finally it balances the data using an external interval builder.
 *
 */
template<class FReal, class ParticleClass>
class FMpiTreeBuilder{
private:
    static const bool VerboseLog;

    /** To keep the leaves information after the sort */
    struct LeafInfo {
        MortonIndex mindex;
        FSize nbParts;
        FSize startingPoint;
    };

public:
    /** What sorting algorithm to use */
    enum SortingType{
        QuickSort,
        BitonicSort,
    };


    /**
     * A particle may not have a MortonIndex Method (set/get morton index)
     * But in this algorithm they are sorted based on their morton indexes.
     * So an IndexedParticle is storing a real particle + its index.
     */
    struct IndexedParticle{
    public:
        MortonIndex index;
        ParticleClass particle;

        operator MortonIndex() const {
            return this->index;
        }
    };

    //////////////////////////////////////////////////////////////////////////
    // Methods to sort the particles
    //////////////////////////////////////////////////////////////////////////


    /** Get an array of particles sorted from their morton indexes */
    template <class LoaderClass>
    static void GetSortedParticlesFromLoader( const FMpi::FComm& communicator, LoaderClass& loader, const SortingType sortingType,
                                              const int TreeHeight, IndexedParticle**const outputSortedParticles, FSize* const outputNbParticlesSorted){
        // Allocate the particles array
        IndexedParticle*const originalParticlesUnsorted = new IndexedParticle[loader.getNumberOfParticles()];
        FMemUtils::memset(originalParticlesUnsorted, 0, sizeof(IndexedParticle) * loader.getNumberOfParticles());

        FPoint<FReal> boxCorner(loader.getCenterOfBox() - (loader.getBoxWidth()/2));
        FTreeCoordinate host;
        const FReal boxWidthAtLeafLevel = loader.getBoxWidth() / FReal(1 << (TreeHeight - 1) );

        // Fill the array and compute the morton index
        for(FSize idxPart = 0 ; idxPart < loader.getNumberOfParticles() ; ++idxPart){
            loader.fillParticle(originalParticlesUnsorted[idxPart].particle);
            host.setX( FCoordinateComputer::GetTreeCoordinate<FReal>( originalParticlesUnsorted[idxPart].particle.getPosition().getX() - boxCorner.getX(), loader.getBoxWidth(), boxWidthAtLeafLevel,
                                           TreeHeight ));
            host.setY( FCoordinateComputer::GetTreeCoordinate<FReal>( originalParticlesUnsorted[idxPart].particle.getPosition().getY() - boxCorner.getY(), loader.getBoxWidth(), boxWidthAtLeafLevel,
                                           TreeHeight ));
            host.setZ( FCoordinateComputer::GetTreeCoordinate<FReal>( originalParticlesUnsorted[idxPart].particle.getPosition().getZ() - boxCorner.getZ(), loader.getBoxWidth(), boxWidthAtLeafLevel,
                                           TreeHeight ));

            originalParticlesUnsorted[idxPart].index = host.getMortonIndex();
        }

        // Sort particles
        if(sortingType == QuickSort){
            FQuickSortMpi<IndexedParticle,MortonIndex, FSize>::QsMpi(originalParticlesUnsorted, loader.getNumberOfParticles(), *outputSortedParticles, *outputNbParticlesSorted,communicator);
            delete [] (originalParticlesUnsorted);
        }
        else {
            FBitonicSort<IndexedParticle,MortonIndex, FSize>::Sort( originalParticlesUnsorted, loader.getNumberOfParticles(), communicator );
            *outputSortedParticles = originalParticlesUnsorted;
            *outputNbParticlesSorted = loader.getNumberOfParticles();
        }
    }

    /** Get an array of particles sorted from their morton indexes */
    static void GetSortedParticlesFromArray( const FMpi::FComm& communicator, const ParticleClass inOriginalParticles[], const FSize originalNbParticles, const SortingType sortingType,
                                             const FPoint<FReal>& centerOfBox, const FReal boxWidth,
                                             const int TreeHeight, IndexedParticle**const outputSortedParticles, FSize* const outputNbParticlesSorted){
        // Allocate the particles array
        IndexedParticle*const originalParticlesUnsorted = new IndexedParticle[originalNbParticles];
        FMemUtils::memset(originalParticlesUnsorted, 0, sizeof(IndexedParticle) * originalNbParticles);

        FPoint<FReal> boxCorner(centerOfBox - (boxWidth/2));
        FTreeCoordinate host;
        const FReal boxWidthAtLeafLevel = boxWidth / FReal(1 << (TreeHeight - 1) );

        FLOG(FTic counterTime);

        // Fill the array and compute the morton index
        for(FSize idxPart = 0 ; idxPart < originalNbParticles ; ++idxPart){
            originalParticlesUnsorted[idxPart].particle = inOriginalParticles[idxPart];
            host.setX( FCoordinateComputer::GetTreeCoordinate<FReal>( originalParticlesUnsorted[idxPart].particle.getPosition().getX() - boxCorner.getX(), boxWidth, boxWidthAtLeafLevel,
                                           TreeHeight ));
            host.setY( FCoordinateComputer::GetTreeCoordinate<FReal>( originalParticlesUnsorted[idxPart].particle.getPosition().getY() - boxCorner.getY(), boxWidth, boxWidthAtLeafLevel,
                                           TreeHeight ));
            host.setZ( FCoordinateComputer::GetTreeCoordinate<FReal>( originalParticlesUnsorted[idxPart].particle.getPosition().getZ() - boxCorner.getZ(), boxWidth, boxWidthAtLeafLevel,
                                           TreeHeight ));

            originalParticlesUnsorted[idxPart].index = host.getMortonIndex();
        }

        FLOG( FLog::Controller << "Particles Distribution: "  << "\tPrepare particles ("  << counterTime.tacAndElapsed() << "s)\n"; FLog::Controller.flush(); );

        // Sort particles
        if(sortingType == QuickSort){
            FQuickSortMpi<IndexedParticle,MortonIndex, FSize>::QsMpi(originalParticlesUnsorted, originalNbParticles, outputSortedParticles, outputNbParticlesSorted,communicator);
            delete [] (originalParticlesUnsorted);
        }
        else {
            FBitonicSort<IndexedParticle,MortonIndex, FSize>::Sort( originalParticlesUnsorted, originalNbParticles, communicator );
            *outputSortedParticles = originalParticlesUnsorted;
            *outputNbParticlesSorted = originalNbParticles;
        }
    }


    //////////////////////////////////////////////////////////////////////////
    // To merge the leaves
    //////////////////////////////////////////////////////////////////////////

    static void MergeSplitedLeaves(const FMpi::FComm& communicator, IndexedParticle** workingArray, FSize* workingSize,
                                   FSize ** leavesOffsetInParticles, ParticleClass** particlesArrayInLeafOrder, FSize* const leavesSize){
        const int myRank = communicator.processId();
        const int nbProcs = communicator.processCount();

        FVector<LeafInfo> leavesInfo;
        { // Get the information of the leaves
            leavesInfo.clear();
            if((*workingSize)){
                leavesInfo.push({(*workingArray)[0].index, 1, 0});
                for(FSize idxPart = 1 ; idxPart < (*workingSize) ; ++idxPart){
                    if(leavesInfo.data()[leavesInfo.getSize()-1].mindex == (*workingArray)[idxPart].index){
                        leavesInfo.data()[leavesInfo.getSize()-1].nbParts += 1;
                    }
                    else{
                        leavesInfo.push({(*workingArray)[idxPart].index, 1, idxPart});
                    }
                }
            }
        }

        if(nbProcs != 1){
            // Some leaf might be divived on several processes, we should move them to the first process
            const MortonIndex noDataFlag = std::numeric_limits<MortonIndex>::max();

            LeafInfo borderLeavesState[2] = { {noDataFlag, 0, 0}, {noDataFlag, 0, 0} };
            if( (*workingSize) != 0 ){
                borderLeavesState[0] = leavesInfo[0];
                borderLeavesState[1] = leavesInfo[leavesInfo.getSize()-1];
                FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] First " << borderLeavesState[0].mindex << "\n"; FLog::Controller.flush(); );
                FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Last " << borderLeavesState[1].mindex << "\n"; FLog::Controller.flush(); );
            }

            std::unique_ptr<LeafInfo[]> allProcFirstLeafStates(new LeafInfo[nbProcs*2]);
            FMpi::MpiAssert(MPI_Allgather(&borderLeavesState, sizeof(LeafInfo)*2, MPI_BYTE,
                                          allProcFirstLeafStates.get(), sizeof(LeafInfo)*2, MPI_BYTE, communicator.getComm()),__LINE__);

            std::vector<MPI_Request> requests;

            // Find what to send/recv from who
            bool hasSentFirstLeaf = false;
            if( (*workingSize) != 0 ){
                // Find the owner of the leaf
                int idProcToSendTo = myRank;
                while(0 < idProcToSendTo &&
                      (allProcFirstLeafStates[(idProcToSendTo-1)*2 + 1].mindex == borderLeavesState[0].mindex
                       || allProcFirstLeafStates[(idProcToSendTo-1)*2 + 1].mindex == noDataFlag)){
                    FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] idProcToSendTo "
                         << idProcToSendTo << " allProcFirstLeafStates[(idProcToSendTo-1)*2 + 1].mindex " <<
                         allProcFirstLeafStates[(idProcToSendTo-1)*2 + 1].mindex << " borderLeavesState[0].mindex " <<
                         borderLeavesState[0].mindex << "\n"; FLog::Controller.flush(); );
                    idProcToSendTo -= 1;
                }
                // We found someone
                if(idProcToSendTo != myRank && allProcFirstLeafStates[(idProcToSendTo)*2 + 1].mindex == borderLeavesState[0].mindex){
                    // Post and send message for the first leaf
                    FMpi::ISendSplit(&(*workingArray)[0], borderLeavesState[0].nbParts, idProcToSendTo,
                            FMpi::TagExchangeIndexs, communicator, &requests);
                    FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] send " << borderLeavesState[0].nbParts << " to " << idProcToSendTo << "\n"; FLog::Controller.flush(); );
                    hasSentFirstLeaf = true;
                }
            }

            bool hasExtendLastLeaf = false;
            std::vector<IndexedParticle> receivedParticles;

            {
                // Count all the particle of our first leaf on other procs
                FSize totalNbParticlesToRecv = 0;
                int idProcToRecvFrom = myRank;
                if(!hasSentFirstLeaf || borderLeavesState[0].mindex != borderLeavesState[1].mindex){
                    while(idProcToRecvFrom+1 < nbProcs &&
                          (borderLeavesState[1].mindex == allProcFirstLeafStates[(idProcToRecvFrom+1)*2].mindex
                           || allProcFirstLeafStates[(idProcToRecvFrom+1)*2].mindex == noDataFlag)){
                        FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] idProcToRecvFrom "
                             << idProcToRecvFrom << " allProcFirstLeafStates[(idProcToRecvFrom+1)*2].mindex " <<
                             allProcFirstLeafStates[(idProcToRecvFrom+1)*2].mindex << " borderLeavesState[1].mindex " <<
                             borderLeavesState[1].mindex << "\n"; FLog::Controller.flush(); );

                        idProcToRecvFrom += 1;
                        totalNbParticlesToRecv += allProcFirstLeafStates[(idProcToRecvFrom)*2].nbParts;
                    }
                }
                // If there are some
                if(totalNbParticlesToRecv){
                    // Alloc a received buffer
                    receivedParticles.resize(totalNbParticlesToRecv);
                    // Post the recv
                    FSize postPositionRecv = 0;
                    for(int postRecvIdx = (myRank+1); postRecvIdx <= idProcToRecvFrom ; ++postRecvIdx){
                        // If there are some on this proc
                        if(allProcFirstLeafStates[(postRecvIdx)*2].mindex != noDataFlag){
                            FMpi::IRecvSplit(&receivedParticles[postPositionRecv], allProcFirstLeafStates[(postRecvIdx)*2].nbParts, postRecvIdx,
                                    FMpi::TagExchangeIndexs, communicator, &requests);
                            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] recv " << allProcFirstLeafStates[(postRecvIdx)*2].nbParts << " from " << postRecvIdx << "\n"; FLog::Controller.flush(); );
                            // Inc the write position
                            postPositionRecv += allProcFirstLeafStates[(postRecvIdx)*2].nbParts;
                        }
                    }
                    hasExtendLastLeaf = true;
                }
            }

            // Finalize communication
            FMpi::MpiAssert(MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE),__LINE__);

            // IF we sent we need to remove the first leaf
            if(hasSentFirstLeaf){
                const FSize offsetParticles = borderLeavesState[0].nbParts;
                // Move all the particles
                for(FSize idxPart = offsetParticles ; idxPart < (*workingSize) ; ++idxPart){
                    (*workingArray)[idxPart - offsetParticles] = (*workingArray)[idxPart];
                }
                // Move all the leaf
                for(int idxLeaf = 1 ; idxLeaf < leavesInfo.getSize() ; ++idxLeaf){
                    leavesInfo[idxLeaf].startingPoint -= offsetParticles;
                    leavesInfo[idxLeaf - 1] = leavesInfo[idxLeaf];
                }
                (*workingSize) -= offsetParticles;
            }

            // If we received we need to merge both arrays
            if(hasExtendLastLeaf){
                // Allocate array
                const FSize finalParticlesNumber = (*workingSize) + receivedParticles.size();
                FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Create array "
                     << finalParticlesNumber << " particles\n"; FLog::Controller.flush(); );
                IndexedParticle* particlesWithExtension = new IndexedParticle[finalParticlesNumber];
                // Copy old data
                memcpy(particlesWithExtension, (*workingArray), (*workingSize)*sizeof(IndexedParticle));
                // Copy received data
                memcpy(particlesWithExtension + (*workingSize), receivedParticles.data(), receivedParticles.size()*sizeof(IndexedParticle));
                // Move ptr
                delete[] (*workingArray);
                (*workingArray)   = particlesWithExtension;
                (*workingSize) = finalParticlesNumber;
                leavesInfo[leavesInfo.getSize()-1].nbParts += receivedParticles.size();
            }
        }
        {//Filling the Array with leaves and parts //// COULD BE MOVED IN AN OTHER FUCTION

            (*leavesSize)    = 0; //init ptr
            (*particlesArrayInLeafOrder)   = nullptr; //init ptr
            (*leavesOffsetInParticles) = nullptr; //init ptr

            if((*workingSize)){
                //Copy all the particles
                (*particlesArrayInLeafOrder) = new ParticleClass[(*workingSize)];
                for(FSize idxPart = 0 ; idxPart < (*workingSize) ; ++idxPart){
                    memcpy(&(*particlesArrayInLeafOrder)[idxPart],&(*workingArray)[idxPart].particle,sizeof(ParticleClass));
                }
                // Assign the number of leaf
                (*leavesSize) = leavesInfo.getSize();
                // Store the offset position for each leaf
                (*leavesOffsetInParticles) = new FSize[leavesInfo.getSize() + 1];
                for(int idxLeaf = 0 ; idxLeaf < leavesInfo.getSize() ; ++idxLeaf){
                    (*leavesOffsetInParticles)[idxLeaf] = leavesInfo[idxLeaf].startingPoint;
                }
                (*leavesOffsetInParticles)[leavesInfo.getSize()] = (*workingSize);
            }
        }
    }


    //////////////////////////////////////////////////////////////////////////
    // To equalize (same number of leaves among the procs)
    //////////////////////////////////////////////////////////////////////////

    /** Put the interval into a tree */
    template <class ContainerClass>
    static void EqualizeAndFillContainer(const FMpi::FComm& communicator,  ContainerClass* particlesSaver,
                                         const FSize leavesOffsetInParticles[], const ParticleClass particlesArrayInLeafOrder[],
                                         const FSize currentNbLeaves,
                                         const FSize currentNbParts, FAbstractBalanceAlgorithm * balancer){
        const int myRank = communicator.processId();
        const int nbProcs = communicator.processCount();

        if(nbProcs == 1){
            //Just copy each part into the Particle Saver
            for(FSize idxPart =0 ; idxPart < currentNbParts ; ++idxPart){
                particlesSaver->push(particlesArrayInLeafOrder[idxPart]);
            }
        }
        else{
            // We need to know the number of leaves per procs
            std::unique_ptr<FSize[]> numberOfLeavesPerProc(new FSize[nbProcs]);
            FMpi::MpiAssert(MPI_Allgather(const_cast<FSize*>(&currentNbLeaves), 1, MPI_LONG_LONG_INT, numberOfLeavesPerProc.get(),
                                          1, MPI_LONG_LONG_INT, communicator.getComm()), __LINE__);


            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Exchange number of leaves\n"; FLog::Controller.flush(); );

            // prefix sum
            std::unique_ptr<FSize[]> diffNumberOfLeavesPerProc(new FSize[nbProcs+1]);
            diffNumberOfLeavesPerProc[0] = 0;
            for(int idxProc = 0 ; idxProc < nbProcs ; ++idxProc ){
                diffNumberOfLeavesPerProc[idxProc+1] = diffNumberOfLeavesPerProc[idxProc] + numberOfLeavesPerProc[idxProc];
            }

            const FSize totalNumberOfLeavesInSimulation  = diffNumberOfLeavesPerProc[nbProcs];
            // Compute the objective interval
            std::vector< std::pair<size_t,size_t> > allObjectives;
            allObjectives.resize(nbProcs);
            for(int idxProc = 0 ; idxProc < nbProcs ; ++idxProc){
                allObjectives[idxProc].first  = balancer->getLeft(totalNumberOfLeavesInSimulation,nbProcs,idxProc);
                allObjectives[idxProc].second = balancer->getRight(totalNumberOfLeavesInSimulation,nbProcs,idxProc);
                if(idxProc != 0) FAssertLF(allObjectives[idxProc].first == allObjectives[idxProc-1].second);
            }

            // Ask for the pack to send
            std::pair<size_t, size_t> myCurrentInter = {diffNumberOfLeavesPerProc[myRank], diffNumberOfLeavesPerProc[myRank+1]};
            const std::vector<FEqualize::Package> packsToSend = FEqualize::GetPackToSend(myCurrentInter, allObjectives);

            FAssertLF((currentNbLeaves == 0 && packsToSend.size() == 0) ||
                      (currentNbLeaves != 0 && packsToSend.size() && FSize(packsToSend[packsToSend.size()-1].elementTo) == currentNbLeaves));

            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Previous currentNbLeaves (" << currentNbLeaves << ")\n"; FLog::Controller.flush(); );
            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Get my interval (" << packsToSend.size() << ")\n"; FLog::Controller.flush(); );
            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Send data\n"; FLog::Controller.flush(); );

            // Store the requests
            std::vector<MPI_Request> requestsNbParts;
            requestsNbParts.reserve(packsToSend.size());

            // Send every thing except for me or if size == 0
            FSize totalSend = 0;
            FSize sendToMe = 0;
            for(unsigned int idxPack = 0; idxPack< packsToSend.size() ; ++idxPack){
                const FEqualize::Package& pack = packsToSend[idxPack];

                if(idxPack != 0) FAssertLF(packsToSend[idxPack].elementFrom == packsToSend[idxPack-1].elementTo);
                FAssertLF(FSize(pack.elementTo) <= FSize(currentNbParts));
                FAssertLF(pack.elementFrom <= pack.elementTo);
                const long long int nbPartsPerPackToSend = leavesOffsetInParticles[pack.elementTo]-leavesOffsetInParticles[pack.elementFrom];
                totalSend += nbPartsPerPackToSend;

                if(pack.idProc != myRank && 0 < (pack.elementTo-pack.elementFrom)){
                    // If not to me and if there is something to send
                    FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] pre-send to " << pack.idProc << " nb " << nbPartsPerPackToSend
                         << " from " << pack.elementFrom << " to " << pack.elementTo << " \n"; FLog::Controller.flush(); );
                    // Send the size of the data
                    requestsNbParts.emplace_back();
                    FMpi::MpiAssert(MPI_Isend((void*)&nbPartsPerPackToSend,1,MPI_LONG_LONG_INT,pack.idProc,
                                              FMpi::TagExchangeIndexs, communicator.getComm(), &requestsNbParts.back()),__LINE__);

                }
                else {
                    sendToMe = nbPartsPerPackToSend;
                    FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] skip " << idxPack
                         << " from " << pack.elementFrom << " to " << pack.elementTo <<  " \n"; FLog::Controller.flush(); );
                }
            }
            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Send done \n"; FLog::Controller.flush(); );
            // Ensure everything has been proceed
            FAssertLF(totalSend == currentNbParts);

            // Compute the current intervals
            std::vector< std::pair<size_t,size_t> > allCurrentIntervals;
            allCurrentIntervals.resize(nbProcs);
            for(int idxProc = 0 ; idxProc < nbProcs ; ++idxProc){
                allCurrentIntervals[idxProc].first  = diffNumberOfLeavesPerProc[idxProc];
                allCurrentIntervals[idxProc].second = diffNumberOfLeavesPerProc[idxProc+1];
            }
            // Ask the packs to receive to fill my objective
            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Get my receive interval \n"; FLog::Controller.flush(); );
            std::pair<size_t, size_t> myObjective = allObjectives[myRank];
            const std::vector<FEqualize::Package> packsToRecv = FEqualize::GetPackToRecv(myObjective, allCurrentIntervals);

            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] recv nb particles \n"; FLog::Controller.flush(); );
            // Count the number of parts to receive
            std::unique_ptr<FSize[]> nbPartsPerPackToRecv(new FSize[packsToRecv.size()]);
            for(unsigned int idxPack = 0; idxPack < packsToRecv.size(); ++idxPack){
                const FEqualize::Package& pack = packsToRecv[idxPack];

                if(idxPack != 0) FAssertLF(packsToRecv[idxPack].elementFrom == packsToRecv[idxPack-1].elementTo);

                if(pack.idProc != myRank && 0 < (pack.elementTo-pack.elementFrom)){
                    FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] pre-recv from " << pack.idProc << " \n"; FLog::Controller.flush(); );
                    // We need to know how much particles to receive
                    requestsNbParts.emplace_back();
                    FMpi::MpiAssert(MPI_Irecv(&nbPartsPerPackToRecv[idxPack], 1, MPI_LONG_LONG_INT, pack.idProc,
                                              FMpi::TagExchangeIndexs, communicator.getComm(), &requestsNbParts.back()), __LINE__);
                }
                else{
                    if(pack.idProc == myRank){
                        // Take my own data
                        const FSize sourcePosition = FMath::Max(myObjective.first, myCurrentInter.first) - myCurrentInter.first;
                        const FSize nbLeavesToCopy = pack.elementTo-pack.elementFrom;
                        nbPartsPerPackToRecv[idxPack] = leavesOffsetInParticles[sourcePosition+nbLeavesToCopy] - leavesOffsetInParticles[sourcePosition];
                        FAssertLF(nbPartsPerPackToRecv[idxPack] == sendToMe);
                        FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] skip recv " <<
                             idxPack << " nb " << nbPartsPerPackToRecv[idxPack] << " \n"; FLog::Controller.flush(); );
                    }
                    else{
                        // Nothing to receive from this so avoid communication
                        nbPartsPerPackToRecv[idxPack] = 0;
                    }
                }
            }

            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Wait \n"; FLog::Controller.flush(); );

            FMpi::MpiAssert(MPI_Waitall(int(requestsNbParts.size()), requestsNbParts.data(), MPI_STATUSES_IGNORE), __LINE__);

            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Wait Done \n"; FLog::Controller.flush(); );

            std::vector<MPI_Request> requestsParts;

            for(unsigned int idxPack = 0; idxPack< packsToSend.size() ; ++idxPack){
                const FEqualize::Package& pack = packsToSend[idxPack];
                if(pack.idProc != myRank && 0 < (pack.elementTo-pack.elementFrom)){
                    const long long int nbPartsPerPackToSend = leavesOffsetInParticles[pack.elementTo]-leavesOffsetInParticles[pack.elementFrom];

                    FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] send to "
                          << pack.idProc << " nb " << nbPartsPerPackToSend << " \n"; FLog::Controller.flush(); );

                    FMpi::ISendSplit(&particlesArrayInLeafOrder[leavesOffsetInParticles[pack.elementFrom]],
                                    nbPartsPerPackToSend,
                                    pack.idProc,
                                    FMpi::TagExchangeIndexs + 1,
                                    communicator,
                                    &requestsParts);
                }
            }


            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] barrier after all send \n"; FLog::Controller.flush(); );

            ////////////////////////////////////////////////////////////////
            // Count the number of leaf to receive
            ////////////////////////////////////////////////////////////////
            FSize totalPartsToReceive = 0;
            for(unsigned int idxPack = 0; idxPack < packsToRecv.size(); ++idxPack){
                totalPartsToReceive += nbPartsPerPackToRecv[idxPack];
            }

            std::unique_ptr<ParticleClass[]> particlesRecvBuffer(new ParticleClass[totalPartsToReceive]);

            ////////////////////////////////////////////////////////////////
            // Post all the receive and copy mine
            // it is based on the nbPartsPerPackToRecv array
            ////////////////////////////////////////////////////////////////
            if(totalPartsToReceive){
                FSize offsetToRecv = 0;
                for(unsigned int idxPack = 0; idxPack < packsToRecv.size(); ++idxPack){
                    const FEqualize::Package& pack = packsToRecv[idxPack];
                    // If it is not from me
                    if(pack.idProc != myRank && 0 < (pack.elementTo-pack.elementFrom)){
                        FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] recv from "
                              << pack.idProc << " nb " << nbPartsPerPackToRecv[idxPack] << " from " << pack.elementFrom << "\n"; FLog::Controller.flush(); );

                        // We store from offset, and use nbPartsPerPackToRecv has the number
                        FMpi::IRecvSplit(&particlesRecvBuffer[offsetToRecv],
                                        nbPartsPerPackToRecv[idxPack],
                                        pack.idProc,
                                        FMpi::TagExchangeIndexs + 1,
                                        communicator,
                                        &requestsParts);

                    }
                    // it is from me, just copy
                    else if(pack.idProc == myRank){
                        FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] copy "
                             << idxPack << " nb " << nbPartsPerPackToRecv[idxPack] << " from " << pack.elementFrom << " \n"; FLog::Controller.flush(); );
                        // Copy my particles
                        const FSize sourcePosition = FMath::Max(myObjective.first, myCurrentInter.first) - myCurrentInter.first;
                        // We store from offset, and use nbPartsPerPackToRecv has the number
                        // The reading position is the offset of the first leaf we own
                        memcpy(&particlesRecvBuffer[offsetToRecv], &particlesArrayInLeafOrder[leavesOffsetInParticles[sourcePosition]],
                                nbPartsPerPackToRecv[idxPack]*sizeof(ParticleClass));
                    }
                    offsetToRecv += nbPartsPerPackToRecv[idxPack];
                }
            }


            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] pre Wait \n"; FLog::Controller.flush(); );

            FMpi::Assert( MPI_Waitall(int(requestsParts.size()), requestsParts.data(), MPI_STATUSES_IGNORE),  __LINE__ );

            FLOG(if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << communicator.processId() << "] Wait Done \n"; FLog::Controller.flush(); );

            // Insert in the particle saver
            for(FSize idPartsToStore = 0 ; idPartsToStore < totalPartsToReceive ; ++idPartsToStore){
                particlesSaver->push(particlesRecvBuffer[idPartsToStore]);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // The builder function
    //////////////////////////////////////////////////////////////////////////

    template <class ContainerClass>
    static void DistributeArrayToContainer(const FMpi::FComm& communicator, const ParticleClass originalParticlesArray[], const FSize originalNbParticles,
                                           const FPoint<FReal>& boxCenter, const FReal boxWidth, const int treeHeight,
                                           ContainerClass* particleSaver, FAbstractBalanceAlgorithm* balancer, const SortingType sortingType = QuickSort){

        FLOG( FLog::Controller << "["  << communicator.processId() << "] Particles Distribution: "  << "Enter DistributeArrayToContainer\n" ; FLog::Controller.flush(); );
        FLOG( FTic timer );

        IndexedParticle* sortedParticlesArray = nullptr;
        FSize nbParticlesInArray = 0;
        // From ParticleClass get array of IndexedParticle sorted
        GetSortedParticlesFromArray(communicator, originalParticlesArray, originalNbParticles, sortingType, boxCenter, boxWidth, treeHeight,
                                    &sortedParticlesArray, &nbParticlesInArray);
        FLOG( FLog::Controller << "["  << communicator.processId() << "] Particles Distribution: "
              << "\t GetSortedParticlesFromArray is over (" << timer.tacAndElapsed() << "s) "
              << nbParticlesInArray << " particles\n"; FLog::Controller.flush(); );
        FLOG( timer.tic() );

//        for(int idx = 0 ; idx < nbParticlesInArray ; ++idx){
//            particleSaver->push(sortedParticlesArray[idx].particle);
//        }
        ParticleClass* particlesArrayInLeafOrder = nullptr;
        FSize * leavesOffsetInParticles = nullptr;
        FSize nbLeaves = 0;
        // Merge the leaves
        MergeSplitedLeaves(communicator, &sortedParticlesArray, &nbParticlesInArray, &leavesOffsetInParticles, &particlesArrayInLeafOrder, &nbLeaves);
        delete[] sortedParticlesArray;

//        for(int idx = 0 ; idx < nbParticlesInArray ; ++idx){
//            particleSaver->push(particlesArrayInLeafOrder[idx]);
//        }

        FLOG( FLog::Controller << "["  << communicator.processId() << "] Particles Distribution: "  << "\t MergeSplitedLeaves is over (" << timer.tacAndElapsed() << "s)\n"; FLog::Controller.flush(); );
        FLOG( timer.tic() );

        // Equalize and balance
        EqualizeAndFillContainer(communicator, particleSaver, leavesOffsetInParticles, particlesArrayInLeafOrder, nbLeaves,
                                 nbParticlesInArray, balancer);
        delete[] particlesArrayInLeafOrder;
        delete[] leavesOffsetInParticles;

        FLOG( FLog::Controller << "["  << communicator.processId() << "] Particles Distribution: "  << "\t EqualizeAndFillContainer is over (" << timer.tacAndElapsed() << "s)\n"; FLog::Controller.flush(); );

        FLOG( FLog::Controller << "["  << communicator.processId() << "] Particles Distribution: "  << "\t DistributeArrayToContainer is over (" << timer.cumulated() << "s)\n"; FLog::Controller.flush(); );

#ifdef SCALFMM_USE_LOG
        /** To produce stats after the Equalize phase  */
        {
            const FSize finalNbParticles = particleSaver->getSize();

            if(communicator.processId() != 0){
                FMpi::MpiAssert(MPI_Gather(const_cast<FSize*>(&finalNbParticles),1,FMpi::GetType(finalNbParticles),nullptr,
                                           1,FMpi::GetType(finalNbParticles),0,communicator.getComm()), __LINE__);
            }
            else{
                const int nbProcs = communicator.processCount();
                std::unique_ptr<FSize[]> nbPartsPerProc(new FSize[nbProcs]);

                FMpi::MpiAssert(MPI_Gather(const_cast<FSize*>(&finalNbParticles),1,FMpi::GetType(finalNbParticles),nbPartsPerProc.get(),
                                           1,FMpi::GetType(finalNbParticles),0,communicator.getComm()), __LINE__);

                FReal averageNbParticles = 0;
                FSize minNbParticles = finalNbParticles;
                FSize maxNbParticles = finalNbParticles;

                for(int idxProc = 0 ; idxProc < nbProcs ; ++idxProc){
                    maxNbParticles = FMath::Max(maxNbParticles, nbPartsPerProc[idxProc]);
                    minNbParticles = FMath::Min(minNbParticles, nbPartsPerProc[idxProc]);
                    averageNbParticles += FReal(nbPartsPerProc[idxProc]);
                }
                averageNbParticles /= float(nbProcs);

                printf("Particles Distribution: End of Equalize Phase : \n \t Min number of parts : %lld \n \t Max number of parts : %lld \n \t Average number of parts : %e \n",
                       minNbParticles,maxNbParticles,averageNbParticles);
            }
        }
#endif
    }


};


#ifdef SCALFMM_USE_LOG
template<class FReal, class ParticleClass>
const bool FMpiTreeBuilder<FReal,ParticleClass>::VerboseLog = FEnv::GetBool("SCALFMM_DEBUG_LOG", false);
#endif

#endif // FMPITREEBUILDER_H
