// See LICENCE file at project root
#ifndef FQUICKSORTMPI_HPP
#define FQUICKSORTMPI_HPP

#include "FQuickSort.hpp"
#include "FMpi.hpp"
#include "FLog.hpp"
#include "FAssert.hpp"
#include "FEnv.hpp"

#include <memory>
#include <utility>

template <class SortType, class CompareType, class IndexType = size_t>
class FQuickSortMpi : public FQuickSort< SortType, IndexType> {
#ifdef SCALFMM_USE_LOG
    static const bool VerboseLog;
#endif

    // We need a structure see the algorithm detail to know more
    struct Partition{
        IndexType lowerPart;
        IndexType greaterPart;
    };

    struct PackData {
        int idProc;
        IndexType fromElement;
        IndexType toElement;
    };


    static void Swap(SortType& value, SortType& other){
        const SortType temp = std::move(value);
        value = std::move(other);
        other = std::move(temp);
    }

    /* A local iteration of qs */
    static IndexType QsPartition(SortType array[], IndexType left, IndexType right, const CompareType& pivot){
        IndexType idx = left;
        while( idx <= right && CompareType(array[idx]) <= pivot){
            idx += 1;
        }
        left = idx;

        for( ; idx <= right ; ++idx){
            if( CompareType(array[idx]) <= pivot ){
                Swap(array[idx],array[left]);
                left += 1;
            }
        }

        return left;
    }

    static std::vector<PackData> Distribute(const int currentRank, const int currentNbProcs ,
                                            const Partition globalElementBalance[], const Partition globalElementBalanceSum[],
                                            const int procInTheMiddle, const bool inFromRightToLeft){
        // First agree on who send and who recv
        const int firstProcToSend = (inFromRightToLeft ? procInTheMiddle+1 : 0);
        const int lastProcToSend  = (inFromRightToLeft ? currentNbProcs : procInTheMiddle+1);
        const int firstProcToRecv = (inFromRightToLeft ? 0 : procInTheMiddle+1);
        const int lastProcToRecv  = (inFromRightToLeft ? procInTheMiddle+1 : currentNbProcs);
        // Get the number of element depending on the lower or greater send/recv
        const IndexType totalElementToProceed = (inFromRightToLeft ?
                                                     globalElementBalanceSum[lastProcToSend].lowerPart - globalElementBalanceSum[firstProcToSend].lowerPart :
                                                     globalElementBalanceSum[lastProcToSend].greaterPart - globalElementBalanceSum[firstProcToSend].greaterPart);
        const IndexType totalElementAlreadyOwned = (inFromRightToLeft ?
                                                        globalElementBalanceSum[lastProcToRecv].lowerPart - globalElementBalanceSum[firstProcToRecv].lowerPart :
                                                        globalElementBalanceSum[lastProcToRecv].greaterPart - globalElementBalanceSum[firstProcToRecv].greaterPart);

        const int nbProcsToRecv = (lastProcToRecv-firstProcToRecv);
        const int nbProcsToSend = (lastProcToSend-firstProcToSend);
        const IndexType totalElements = (totalElementToProceed+totalElementAlreadyOwned);

        std::vector<IndexType> nbElementsToRecvPerProc;
        nbElementsToRecvPerProc.resize(nbProcsToRecv);
        {
            // Get the number of elements each proc should recv
            IndexType totalRemainingElements = totalElements;
            IndexType totalAvailableElements = totalElementToProceed;


            for(int idxProc = firstProcToRecv; idxProc < lastProcToRecv ; ++idxProc){
                const IndexType nbElementsAlreadyOwned = (inFromRightToLeft ? globalElementBalance[idxProc].lowerPart : globalElementBalance[idxProc].greaterPart);
                const IndexType averageNbElementForRemainingProc = (totalRemainingElements)/(lastProcToRecv-idxProc);
                totalRemainingElements -= nbElementsAlreadyOwned;
                FAssertLF(totalRemainingElements >= 0);
                if(nbElementsAlreadyOwned < averageNbElementForRemainingProc && totalAvailableElements){
                    nbElementsToRecvPerProc[idxProc - firstProcToRecv] = FMath::Min(totalAvailableElements,
                                                                                    averageNbElementForRemainingProc - nbElementsAlreadyOwned);
                    totalAvailableElements -= nbElementsToRecvPerProc[idxProc - firstProcToRecv];
                    totalRemainingElements -= nbElementsToRecvPerProc[idxProc - firstProcToRecv];
                }
                else{
                    nbElementsToRecvPerProc[idxProc - firstProcToRecv] = 0;
                }
                FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentRank << "] nbElementsToRecvPerProc[" << idxProc << "] = " << nbElementsToRecvPerProc[idxProc - firstProcToRecv] << "\n"; )
            }
            FAssertLF(totalRemainingElements == 0);
        }

        // Store in an array the number of element to send
        std::vector<IndexType> nbElementsToSendPerProc;
        nbElementsToSendPerProc.resize(nbProcsToSend);
        for(int idxProc = firstProcToSend; idxProc < lastProcToSend ; ++idxProc){
            const IndexType nbElementsAlreadyOwned = (inFromRightToLeft ? globalElementBalance[idxProc].lowerPart : globalElementBalance[idxProc].greaterPart);
            nbElementsToSendPerProc[idxProc-firstProcToSend] = nbElementsAlreadyOwned;
            FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentRank << "] nbElementsToSendPerProc[" << idxProc << "] = " << nbElementsToSendPerProc[idxProc-firstProcToSend] << "\n"; )
        }

        // Compute all the send recv but keep only the ones related to currentRank
        std::vector<PackData> packs;
        int idxProcSend = 0;
        IndexType positionElementsSend = 0;
        int idxProcRecv = 0;
        IndexType positionElementsRecv = 0;
        while(idxProcSend != nbProcsToSend && idxProcRecv != nbProcsToRecv){
            if(nbElementsToSendPerProc[idxProcSend] == 0){
                idxProcSend += 1;
                positionElementsSend = 0;
            }
            else if(nbElementsToRecvPerProc[idxProcRecv] == 0){
                idxProcRecv += 1;
                positionElementsRecv = 0;
            }
            else {
                const IndexType nbElementsInPack = FMath::Min(nbElementsToSendPerProc[idxProcSend], nbElementsToRecvPerProc[idxProcRecv]);
                if(idxProcSend + firstProcToSend == currentRank){
                    PackData pack;
                    pack.idProc      = idxProcRecv + firstProcToRecv;
                    pack.fromElement = positionElementsSend;
                    pack.toElement   = pack.fromElement + nbElementsInPack;
                    packs.push_back(pack);
                }
                else if(idxProcRecv + firstProcToRecv == currentRank){
                    PackData pack;
                    pack.idProc      = idxProcSend + firstProcToSend;
                    pack.fromElement = positionElementsRecv;
                    pack.toElement   = pack.fromElement + nbElementsInPack;
                    packs.push_back(pack);
                }
                nbElementsToSendPerProc[idxProcSend] -= nbElementsInPack;
                nbElementsToRecvPerProc[idxProcRecv] -= nbElementsInPack;
                positionElementsSend += nbElementsInPack;
                positionElementsRecv += nbElementsInPack;
            }
        }

        return packs;
    }

    static void RecvDistribution(SortType ** inPartRecv, IndexType* inNbElementsRecv,
                                 const Partition globalElementBalance[], const Partition globalElementBalanceSum[],
                                 const int procInTheMiddle, const FMpi::FComm& currentComm, const bool inFromRightToLeft){
        // Compute to know what to recv
        const std::vector<PackData> whatToRecvFromWho = Distribute(currentComm.processId(), currentComm.processCount(),
                                                                   globalElementBalance, globalElementBalanceSum,
                                                                   procInTheMiddle, inFromRightToLeft);
        // Count the total number of elements to recv
        IndexType totalToRecv = 0;
        for(const PackData& pack : whatToRecvFromWho){
            totalToRecv += pack.toElement - pack.fromElement;
        }
        // Alloc buffer
        SortType* recvBuffer = new SortType[totalToRecv];

        // Recv all data
        std::vector<MPI_Request> requests;
        requests.reserve(whatToRecvFromWho.size());
        for(int idxPack = 0 ; idxPack < int(whatToRecvFromWho.size()) ; ++idxPack){
            const PackData& pack = whatToRecvFromWho[idxPack];
            FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] Recv from " << pack.idProc << " from " << pack.fromElement << " to " << pack.toElement << "\n"; );
            FAssertLF(pack.toElement <= totalToRecv);
            FMpi::IRecvSplit(&recvBuffer[pack.fromElement],
                    (pack.toElement - pack.fromElement),
                    pack.idProc,
                    FMpi::TagQuickSort,
                    currentComm,
                    &requests);

        }
        FAssertLF(whatToRecvFromWho.size() <= requests.size());
        FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << "Wait for " << requests.size() << " request \n" );
        FLOG(if(VerboseLog)  FLog::Controller.flush());
        // Wait to complete
        FMpi::Assert( MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE),  __LINE__ );
        FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] Recv Done \n"; )
                FLOG(if(VerboseLog)  FLog::Controller.flush());
        // Copy to ouput variables
        (*inPartRecv) = recvBuffer;
        (*inNbElementsRecv) = totalToRecv;
    }

    static void SendDistribution(const SortType * inPartToSend, const IndexType inNbElementsToSend,
                                 const Partition globalElementBalance[], const Partition globalElementBalanceSum[],
                                 const int procInTheMiddle, const FMpi::FComm& currentComm, const bool inFromRightToLeft){
        // Compute to know what to send
        const std::vector<PackData> whatToSendToWho = Distribute(currentComm.processId(), currentComm.processCount(),
                                                                 globalElementBalance, globalElementBalanceSum,
                                                                 procInTheMiddle, inFromRightToLeft);

        // Post send messages
        std::vector<MPI_Request> requests;
        requests.reserve(whatToSendToWho.size());
        for(int idxPack = 0 ; idxPack < int(whatToSendToWho.size()) ; ++idxPack){
            const PackData& pack = whatToSendToWho[idxPack];
            FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] Send to " << pack.idProc << " from " << pack.fromElement << " to " << pack.toElement << "\n"; );

            FMpi::ISendSplit(&inPartToSend[pack.fromElement],
                    (pack.toElement - pack.fromElement),
                    pack.idProc,
                    FMpi::TagQuickSort,
                    currentComm,
                    &requests);
        }
        FAssertLF(whatToSendToWho.size() <= requests.size());
        FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG [" << currentComm.processId() << "] Wait for " << requests.size() << " request \n" );
        FLOG(if(VerboseLog)  FLog::Controller.flush());
        // Wait to complete
        FMpi::Assert( MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE),  __LINE__ );
        FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] Send Done \n"; )
                FLOG(if(VerboseLog)  FLog::Controller.flush());
    }

    static CompareType SelectPivot(const SortType workingArray[], const IndexType currentSize, const FMpi::FComm& currentComm, bool* shouldStop){
        enum ValuesState{
            ALL_THE_SAME,
            NO_VALUES,
            AVERAGE_2
        };
        // Check if all the same
        bool allTheSame = true;
        // Check if empty
        const bool noValues = (currentSize == 0);
        // Get the local pivot if not empty
        CompareType localPivot = CompareType(0);

        if(noValues == false){
            // We need to know the max value to ensure that the pivot will be different
            CompareType maxFoundValue = CompareType(workingArray[0]);
            // We need to know the min value to ensure that the pivot will be different
            CompareType minFoundValue = CompareType(workingArray[0]);

            for(int idx = 1 ; idx < currentSize ; ++idx){
                // Keep the max
                maxFoundValue = FMath::Max(maxFoundValue , CompareType(workingArray[idx]));
                // Keep the min
                minFoundValue = FMath::Min(minFoundValue , CompareType(workingArray[idx]));
            }
            allTheSame = (maxFoundValue == minFoundValue);
            // Value equal to pivot are kept on the left so
            localPivot = ((maxFoundValue-minFoundValue)/2) + minFoundValue;
            // The pivot must be different (to ensure that the partition will return two parts)
            if( localPivot == maxFoundValue && !allTheSame){
                FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] Pivot " << localPivot << " is equal max and allTheSame equal " << allTheSame << "\n"; )
                        FLOG(if(VerboseLog)  FLog::Controller.flush());
                localPivot -= 1;
            }
        }

        FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] localPivot = " << localPivot << "\n" );
        FLOG(if(VerboseLog)  FLog::Controller.flush());

        //const int myRank = currentComm.processId();
        const int nbProcs = currentComm.processCount();
        // Exchange the pivos and the state
        std::unique_ptr<int[]> allProcStates(new int[nbProcs]);
        std::unique_ptr<CompareType[]> allProcPivots(new CompareType[nbProcs]);
        {
            int myState = (noValues?NO_VALUES:(allTheSame?ALL_THE_SAME:AVERAGE_2));
            FMpi::Assert( MPI_Allgather( &myState, 1, MPI_INT, allProcStates.get(),
                                         1, MPI_INT, currentComm.getComm()),  __LINE__ );
            FMpi::Assert( MPI_Allgather( &localPivot, sizeof(CompareType), MPI_BYTE, allProcPivots.get(),
                                         sizeof(CompareType), MPI_BYTE, currentComm.getComm()),  __LINE__ );
        }
        // Test if all the procs have ALL_THE_SAME and the same value
        bool allProcsAreSame = true;
        for(int idxProc = 0 ; idxProc < nbProcs && allProcsAreSame; ++idxProc){
            if(allProcStates[idxProc] != NO_VALUES && (allProcStates[idxProc] != ALL_THE_SAME || allProcPivots[0] != allProcPivots[idxProc])){
                allProcsAreSame = false;
            }
        }

        if(allProcsAreSame){
            // All the procs are the same so we need to stop
            (*shouldStop) = true;
            return CompareType(0);
        }
        else{
            CompareType globalPivot = 0;
            CompareType counterValuesInPivot = 0;
            // Compute the pivos
            for(int idxProc = 0 ; idxProc < nbProcs; ++idxProc){
                if(allProcStates[idxProc] != NO_VALUES){
                    globalPivot += allProcPivots[idxProc];
                    counterValuesInPivot += 1;
                }
            }
            if(counterValuesInPivot <= 1){
                (*shouldStop) = true;
                return globalPivot;
            }
            (*shouldStop) = false;
            return globalPivot/counterValuesInPivot;
        }
    }

public:

    static void QsMpi(const SortType originalArray[], const IndexType originalSize,
                      SortType** outputArray, IndexType* outputSize, const FMpi::FComm& originalComm){
        // We do not work in place, so create a new array
        IndexType currentSize = originalSize;
        SortType* workingArray = new SortType[currentSize];
        FMemUtils::memcpy(workingArray, originalArray, sizeof(SortType) * currentSize);

        // Clone the MPI group because we will reduce it after each partition
        FMpi::FComm currentComm(originalComm.getComm());

        // Parallel sharing until I am alone on the data
        while( currentComm.processCount() != 1 ){
            // Agree on the Pivot
            bool shouldStop;
            const CompareType globalPivot = SelectPivot(workingArray, currentSize, currentComm, &shouldStop);
            if(shouldStop){
                FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] shouldStop = " << shouldStop << "\n" );
                FLOG(if(VerboseLog)  FLog::Controller.flush());
                break;
            }

            FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] globalPivot = " << globalPivot << " for " << currentComm.processCount() << "\n" );
            FLOG(if(VerboseLog)  FLog::Controller.flush());

            // Split the array in two parts lower equal to pivot and greater than pivot
            const IndexType nbLowerElements = QsPartition(workingArray, 0, currentSize-1, globalPivot);
            const IndexType nbGreaterElements = currentSize - nbLowerElements;

            FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] After Partition: lower = " << nbLowerElements << " greater = " << nbGreaterElements << "\n"; )
                    FLOG(if(VerboseLog)  FLog::Controller.flush());

            const int currentRank = currentComm.processId();
            const int currentNbProcs = currentComm.processCount();

            // We need to know what each process is holding
            Partition currentElementsBalance = { nbLowerElements, nbGreaterElements};
            Partition globalElementBalance[currentNbProcs];

            // Every one in the group need to know
            FMpi::Assert( MPI_Allgather( &currentElementsBalance, sizeof(Partition), MPI_BYTE, globalElementBalance,
                                         sizeof(Partition), MPI_BYTE, currentComm.getComm()),  __LINE__ );

            // Find the number of elements lower or greater
            IndexType globalNumberOfElementsGreater = 0;
            IndexType globalNumberOfElementsLower = 0;
            Partition globalElementBalanceSum[currentNbProcs + 1];
            globalElementBalanceSum[0].lowerPart = 0;
            globalElementBalanceSum[0].greaterPart = 0;
            for(int idxProc = 0 ; idxProc < currentNbProcs ; ++idxProc){
                globalElementBalanceSum[idxProc + 1].lowerPart = globalElementBalanceSum[idxProc].lowerPart + globalElementBalance[idxProc].lowerPart;
                globalElementBalanceSum[idxProc + 1].greaterPart = globalElementBalanceSum[idxProc].greaterPart + globalElementBalance[idxProc].greaterPart;
                globalNumberOfElementsGreater += globalElementBalance[idxProc].greaterPart;
                globalNumberOfElementsLower   += globalElementBalance[idxProc].lowerPart;
            }

            FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] globalNumberOfElementsGreater = " << globalNumberOfElementsGreater << "\n"; )
                    FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] globalNumberOfElementsLower   = " << globalNumberOfElementsLower << "\n"; )
                    FLOG(if(VerboseLog)  FLog::Controller.flush());

            // The proc rank in the middle from the percentage
            int procInTheMiddle;
            if(globalNumberOfElementsLower == 0)        procInTheMiddle = -1;
            else if(globalNumberOfElementsGreater == 0) procInTheMiddle = currentNbProcs-1;
            else procInTheMiddle = int(FMath::Min(IndexType(currentNbProcs-2), (currentNbProcs*globalNumberOfElementsLower)
                                                  /(globalNumberOfElementsGreater + globalNumberOfElementsLower)));

            FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] procInTheMiddle = " << procInTheMiddle << "\n"; )
                    FLOG(if(VerboseLog)  FLog::Controller.flush());

            // Send or receive depending on the state
            if(currentRank <= procInTheMiddle){
                // I am in the group of the lower elements
                SendDistribution(workingArray + nbLowerElements, nbGreaterElements,
                                 globalElementBalance, globalElementBalanceSum, procInTheMiddle, currentComm, false);
                SortType* lowerPartRecv = nullptr;
                IndexType nbLowerElementsRecv = 0;
                RecvDistribution(&lowerPartRecv, &nbLowerElementsRecv,
                                 globalElementBalance, globalElementBalanceSum, procInTheMiddle, currentComm, true);
                // Merge previous part and just received elements
                const IndexType fullNbLowerElementsRecv = nbLowerElementsRecv + nbLowerElements;
                SortType* fullLowerPart = new SortType[fullNbLowerElementsRecv];
                memcpy(fullLowerPart, workingArray, sizeof(SortType)* nbLowerElements);
                memcpy(fullLowerPart + nbLowerElements, lowerPartRecv, sizeof(SortType)* nbLowerElementsRecv);
                delete[] workingArray;
                delete[] lowerPartRecv;
                workingArray = fullLowerPart;
                currentSize = fullNbLowerElementsRecv;
                // Reduce working group
                FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] Reduce group to " << 0 << " / " << procInTheMiddle << "\n"; )
                        FLOG(if(VerboseLog)  FLog::Controller.flush());
                currentComm.groupReduce( 0, procInTheMiddle);
                FLOG(if(VerboseLog)  FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] Done\n" );
                FLOG(if(VerboseLog)  FLog::Controller.flush());
            }
            else {
                // I am in the group of the greater elements
                SortType* greaterPartRecv = nullptr;
                IndexType nbGreaterElementsRecv = 0;
                RecvDistribution(&greaterPartRecv, &nbGreaterElementsRecv,
                                 globalElementBalance, globalElementBalanceSum, procInTheMiddle, currentComm, false);
                SendDistribution(workingArray, nbLowerElements,
                                 globalElementBalance, globalElementBalanceSum, procInTheMiddle, currentComm, true);
                // Merge previous part and just received elements
                const IndexType fullNbGreaterElementsRecv = nbGreaterElementsRecv + nbGreaterElements;
                SortType* fullGreaterPart = new SortType[fullNbGreaterElementsRecv];
                memcpy(fullGreaterPart, workingArray + nbLowerElements, sizeof(SortType)* nbGreaterElements);
                memcpy(fullGreaterPart + nbGreaterElements, greaterPartRecv, sizeof(SortType)* nbGreaterElementsRecv);
                delete[] workingArray;
                delete[] greaterPartRecv;
                workingArray = fullGreaterPart;
                currentSize = fullNbGreaterElementsRecv;
                // Reduce working group
                FLOG( if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] Reduce group to " << procInTheMiddle + 1 << " / " << currentNbProcs - 1 << "\n"; )
                        FLOG( if(VerboseLog) FLog::Controller.flush());
                currentComm.groupReduce( procInTheMiddle + 1, currentNbProcs - 1);
                FLOG( if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] Done\n"; )
                        FLOG( if(VerboseLog) FLog::Controller.flush());
            }
        }

        FLOG( if(VerboseLog) FLog::Controller << "SCALFMM-DEBUG ["  << currentComm.processId() << "] Sequential sort (currentSize = " << currentSize << ")\n"; )
        FLOG( if(VerboseLog) FLog::Controller.flush());
        // Finish by a local sort
        FQuickSort< SortType, IndexType>::QsOmp(workingArray, currentSize, [](const SortType& v1, const SortType& v2){
            return CompareType(v1) <= CompareType(v2);
        });
        (*outputSize)  = currentSize;
        (*outputArray) = workingArray;
    }
};


#ifdef SCALFMM_USE_LOG
template <class SortType, class CompareType, class IndexType>
const bool FQuickSortMpi<SortType, CompareType, IndexType>::VerboseLog = FEnv::GetBool("SCALFMM_DEBUG_LOG", false);
#endif

#endif // FQUICKSORTMPI_HPP
