// See LICENCE file at project root
//
// 2017-01-04
//   OC bottomPass, UpwardPass and downardPass done
//
// ToDO
//   1) transferPass, and directPass (L2P ans P2P)
//   2) to check and valid
//
//
// ==== CMAKE =====
// @FUSE_MPI
// ================
//
//
#ifndef FFMMALGORITHMTHREADPROC_HPP
#define FFMMALGORITHMTHREADPROC_HPP

#include <omp.h>

//
#include "../Utils/FAssert.hpp"
#include "../Utils/FLog.hpp"

#include "../Utils/FTic.hpp"
#include "Utils/FAlgorithmTimers.hpp"

#include "../Utils/FGlobal.hpp"

#include "../Containers/FBoolArray.hpp"
#include "../Containers/FOctree.hpp"
#include "../Containers/FLightOctree.hpp"
#include "../Utils/FEnv.hpp"

#include "../Containers/FBufferWriter.hpp"
#include "../Containers/FBufferReader.hpp"
#include "../Containers/FMpiBufferWriter.hpp"
#include "../Containers/FMpiBufferReader.hpp"

#include "../Utils/FMpi.hpp"
#include <sys/time.h>

#include "FCoreCommon.hpp"
#include "FP2PExclusion.hpp"

#include <memory>
#include <vector>

/**
 * @author
 *
 * Please read the license
 *
 * This class is a threaded FMM algorithm distributed using MPI. It iterates on
 * a tree and call the kernels with good arguments. It uses the inspector -
 * executor model : iterates on the tree and builds an array to work in parallel
 * on this array
 *
 * This class does not free pointers given in arguements.
 *
 * Threaded & based on the inspector-executor model
 *
 *     schedule(runtime) export OMP_NUM_THREADS=2
 *     export OMPI_CXX=`which g++-4.4`
 *     mpirun -np 2 valgrind --suppressions=/usr/share/openmpi/openmpi-valgrind.supp
 *        --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20
 *        --track-fds=yes ./Tests/testFmmAlgorithmProcTsm ../Data/testLoaderSmall.fma.tmp
 */
template<class OctreeClass, class CellClass, class ContainerClass, class KernelClass, class LeafClass, class P2PExclusionClass = FP2PMiddleExclusion>
class FFmmAlgorithmThreadProcTsm : public FAbstractAlgorithm, public FAlgorithmTimers {
private:
    OctreeClass* const tree;     ///< The octree to work on
    KernelClass** kernels;       ///< The kernels

    const FMpi::FComm comm;      ///< MPI comm
    FMpi::FComm fcomCompute;

    /// Used to store pointers to cells/leafs to work with
    typename OctreeClass::Iterator* iterArray;
    /// Used to store pointers to cells/leafs to send/rcv
    typename OctreeClass::Iterator* iterArrayComm;

    int numberOfLeafs;           ///< To store the size at the previous level
    const int MaxThreads;        ///< Max number of thread allowed by openmp
    const int nbProcessOrig;         ///< Process count
    const int idProcessOrig;         ///< Current process id
    int nbProcess;        ///< Process count
    int idProcess;        ///< Current process id
    const int OctreeHeight;      ///< Tree height

    const int userChunkSize;
    const int leafLevelSeparationCriteria;

    /** An interval is the morton index interval
     * that a proc uses (i.e. it holds data in this interval) */
    struct Interval{
        MortonIndex leftIndex;
        MortonIndex rightIndex;
    };

    /// Current process interval
    Interval*const intervals;
    /// All processes intervals
    Interval*const workingIntervalsPerLevel;

    /// Get an interval from a process id and tree level
    Interval& getWorkingInterval( int level,  int proc){
        return workingIntervalsPerLevel[OctreeHeight * proc + level];
    }

    /// Get an interval from a process id and tree level
    const Interval& getWorkingInterval( int level,  int proc) const {
        return workingIntervalsPerLevel[OctreeHeight * proc + level];
    }

    /// Does \a procIdx have work at given \a idxLevel
    /** i.e. does it hold cells and is responsible of them ? */
    bool procHasWorkAtLevel(const int idxLevel , const int idxProc) const {
        return getWorkingInterval(idxLevel, idxProc).leftIndex <= getWorkingInterval(idxLevel, idxProc).rightIndex;
    }

    /** True if the \a idxProc left cell at \a idxLevel+1 has the same parent as us for our right cell */
    bool procCoversMyRightBorderCell(const int idxLevel , const int idxProc) const {
        return (getWorkingInterval((idxLevel+1) , idProcess).rightIndex>>3) == (getWorkingInterval((idxLevel+1) ,idxProc).leftIndex >>3);
    }

    /** True if the idxProc right cell at idxLevel+1 has the same parent as us for our left cell */
    bool procCoversMyLeftBorderCell(const int idxLevel , const int idxProc) const {
        return (getWorkingInterval((idxLevel+1) , idxProc).rightIndex >>3) == (getWorkingInterval((idxLevel+1) , idProcess).leftIndex>>3);
    }

public:
    /// Get current process interval at given \a level
    Interval& getWorkingInterval( int level){
        return getWorkingInterval(level, idProcess);
    }

    /// Does the current process has some work at this level ?
    bool hasWorkAtLevel( int level){
        return idProcess == 0 || (getWorkingInterval(level, idProcess - 1).rightIndex) < (getWorkingInterval(level, idProcess).rightIndex);
    }

    /**@brief Constructor
     * @param inTree the octree to work on
     * @param inKernels the kernels to call
     *
     * An assert is launched if one of the arguments is null
     */
    FFmmAlgorithmThreadProcTsm(const FMpi::FComm& inComm, OctreeClass* const inTree,
                               KernelClass* const inKernels,
                               const int inUserChunkSize = 10, const int inLeafLevelSeperationCriteria = 1) :
        tree(inTree),
        kernels(nullptr),
        comm(inComm),
        fcomCompute(inComm),
        iterArray(nullptr),
        iterArrayComm(nullptr),
        numberOfLeafs(0),
        MaxThreads(FEnv::GetValue("SCALFMM_ALGO_NUM_THREADS",omp_get_max_threads())),
        nbProcessOrig(inComm.processCount()),
        idProcessOrig(inComm.processId()),
        nbProcess(0),
        idProcess(0),
        OctreeHeight(tree->getHeight()),
        userChunkSize(inUserChunkSize),
        leafLevelSeparationCriteria(inLeafLevelSeperationCriteria),
        intervals(new Interval[inComm.processCount()]),
        workingIntervalsPerLevel(new Interval[inComm.processCount() * tree->getHeight()]) {
        FAssertLF(tree, "tree cannot be null");
        FAssertLF(leafLevelSeparationCriteria < 3, "Separation criteria should be < 3");

        this->kernels = new KernelClass*[MaxThreads];
#pragma omp parallel num_threads(MaxThreads)
        {
#pragma omp critical (InitFFmmAlgorithmThreadProcTsm)
            {
                this->kernels[omp_get_thread_num()] = new KernelClass(*inKernels);
            }
        }

        FAbstractAlgorithm::setNbLevelsInTree(tree->getHeight());

        FLOG(FLog::Controller << "FFmmAlgorithmThreadProcTsm\n");
        FLOG(FLog::Controller << "Max threads = "  << MaxThreads << ", Procs = " << nbProcessOrig << ", I am " << idProcessOrig << ".\n");
        FLOG(FLog::Controller << "Chunck Size = " << userChunkSize << "\n");
    }

    /// Default destructor
    virtual ~FFmmAlgorithmThreadProcTsm(){
        for(int idxThread = 0 ; idxThread < MaxThreads ; ++idxThread){
            delete this->kernels[idxThread];
        }
        delete [] this->kernels;

        delete [] intervals;
        delete [] workingIntervalsPerLevel;
    }

protected:
    /**
     * To execute the fmm algorithm
     * Call this function to run the complete algorithm
     */
    void executeCore(const unsigned operationsToProceed) override {
        // We are not involve if the tree is empty
        const int iHaveParticles = (!tree->isEmpty());

        std::unique_ptr<int[]> hasParticles(new int[comm.processCount()]);
        FMpi::Assert( MPI_Allgather(const_cast<int*>(&iHaveParticles), 1,MPI_INT,
                                    hasParticles.get(), 1, MPI_INT,
                                    comm.getComm()), __LINE__);

        fcomCompute = FMpi::FComm(comm);
        fcomCompute.groupReduce(hasParticles.get());

        if(iHaveParticles){

            nbProcess = fcomCompute.processCount();
            idProcess = fcomCompute.processId();

            FLOG(FLog::Controller << "Max threads = "  << MaxThreads << ", Procs = " << nbProcess << ", I am " << idProcess << ".\n");

            // Count leaf
#ifdef SCALFMM_TRACE_ALGO
            eztrace_resume();
#endif
            this->numberOfLeafs = 0;
            {
                Interval myFullInterval;
                {//Building the interval with the first and last leaves (and count the number of leaves)
                    typename OctreeClass::Iterator octreeIterator(tree);
                    octreeIterator.gotoBottomLeft();
                    myFullInterval.leftIndex = octreeIterator.getCurrentGlobalIndex();
                    do{
                        ++this->numberOfLeafs;
                    } while(octreeIterator.moveRight());
                    myFullInterval.rightIndex = octreeIterator.getCurrentGlobalIndex();
                }
                // Allocate a number to store the pointer of the cells at a level
                iterArray     = new typename OctreeClass::Iterator[numberOfLeafs];
                iterArrayComm = new typename OctreeClass::Iterator[numberOfLeafs];
                FAssertLF(iterArray,     "iterArray     bad alloc");
                FAssertLF(iterArrayComm, "iterArrayComm bad alloc");

                // We get the leftIndex/rightIndex indexes from each procs
                FMpi::MpiAssert( MPI_Allgather( &myFullInterval, sizeof(Interval), MPI_BYTE, intervals, sizeof(Interval), MPI_BYTE, fcomCompute.getComm()),  __LINE__ );

                // Build my intervals for all levels
                std::unique_ptr<Interval[]> myIntervals(new Interval[OctreeHeight]);
                // At leaf level we know it is the full interval
                myIntervals[OctreeHeight - 1] = myFullInterval;

                // We can estimate the interval for each level by using the parent/child relation
                for(int idxLevel = OctreeHeight - 2 ; idxLevel >= 0 ; --idxLevel){
                    myIntervals[idxLevel].leftIndex = myIntervals[idxLevel+1].leftIndex >> 3;
                    myIntervals[idxLevel].rightIndex = myIntervals[idxLevel+1].rightIndex >> 3;
                }

                // Process 0 uses the estimates as real intervals, but other processes
                // should remove cells that belong to others
                if(idProcess != 0){
                    //We test for each level if process on left (idProcess-1) own cell I thought I owned
                    typename OctreeClass::Iterator octreeIterator(tree);
                    octreeIterator.gotoBottomLeft();
                    octreeIterator.moveUp();

                    // At h-1 the working limit is the parent of the right cell of the proc on the left
                    MortonIndex workingLimitAtLevel = intervals[idProcess-1].rightIndex >> 3;

                    // We check if we have no more work to do
                    int nullIntervalFromLevel = 0;

                    for(int idxLevel = OctreeHeight - 2 ; idxLevel >= 1 && nullIntervalFromLevel == 0 ; --idxLevel){
                        while(octreeIterator.getCurrentGlobalIndex() <= workingLimitAtLevel){
                            if( !octreeIterator.moveRight() ){
                                // We cannot move right we are not owner of any more cell
                                nullIntervalFromLevel = idxLevel;
                                break;
                            }
                        }
                        // If we are responsible for some cells at this level keep the first index
                        if(nullIntervalFromLevel == 0){
                            myIntervals[idxLevel].leftIndex = octreeIterator.getCurrentGlobalIndex();
                            octreeIterator.moveUp();
                            workingLimitAtLevel >>= 3;
                        }
                    }
                    // In case we are not responsible for any cells we put the leftIndex = rightIndex+1
                    for(int idxLevel = nullIntervalFromLevel ; idxLevel >= 1 ; --idxLevel){
                        myIntervals[idxLevel].leftIndex = myIntervals[idxLevel].rightIndex + 1;
                    }
                }

                // We get the leftIndex/rightIndex indexes from each procs
                FMpi::MpiAssert( MPI_Allgather( myIntervals.get(), int(sizeof(Interval)) * OctreeHeight, MPI_BYTE,
                                                workingIntervalsPerLevel, int(sizeof(Interval)) * OctreeHeight, MPI_BYTE, fcomCompute.getComm()),  __LINE__ );
            }
#ifdef SCALFMM_TRACE_ALGO
            eztrace_enter_event("P2M", EZTRACE_YELLOW);
#endif
            Timers[P2MTimer].tic();
            if(operationsToProceed & FFmmP2M) bottomPass();
            Timers[P2MTimer].tac();

#ifdef SSCALFMM_TRACE_ALGO
            eztrace_leave_event();
            eztrace_enter_event("M2M", EZTRACE_PINK);
#endif

            Timers[M2MTimer].tic();
            if(operationsToProceed & FFmmM2M) upwardPass();
            Timers[M2MTimer].tac();

#ifdef SCALFMM_TRACE_ALGO
            eztrace_leave_event();
            eztrace_enter_event("M2L", EZTRACE_GREEN);
#endif

            Timers[M2LTimer].tic();
            if(operationsToProceed & FFmmM2L) transferPass();
            Timers[M2LTimer].tac();

#ifdef SCALFMM_TRACE_ALGO
            eztrace_leave_event();
            eztrace_enter_event("L2L", EZTRACE_PINK);
#endif

            Timers[L2LTimer].tic();
            if(operationsToProceed & FFmmL2L) downardPass();
            Timers[L2LTimer].tac();

#ifdef SCALFMM_TRACE_ALGO
            eztrace_leave_event();
            eztrace_enter_event("L2P+P2P", EZTRACE_BLUE);
#endif

            Timers[NearTimer].tic();
            if( (operationsToProceed & FFmmP2P) || (operationsToProceed & FFmmL2P) ) directPass((operationsToProceed & FFmmP2P),(operationsToProceed & FFmmL2P));
            Timers[NearTimer].tac();

#ifdef SCALFMM_TRACE_ALGO
            eztrace_leave_event();
            eztrace_stop();
#endif
            // delete array
            delete []     iterArray;
            delete []     iterArrayComm;
            iterArray          = nullptr;
            iterArrayComm = nullptr;
#ifdef SCALFMM_TRACE_ALGO
            eztrace_stop();
#endif
        }
        else{
            FLOG( FLog::Controller << "\tProcess = " << comm.processId() << " has zero particles.\n" );
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    // P2M
    /////////////////////////////////////////////////////////////////////////////

    /**
     * P2M Bottom Pass
     * No communication are involved in the P2M.
     * It is similar to multi threaded version.
     * We traverse alle the octree to set if cells are  sources or targets
     */
    void bottomPass(){
        FLOG( FLog::Controller.write("\tStart Bottom Pass\n").write(FLog::Flush) );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);
        typename OctreeClass::Iterator octreeIterator(tree);

        // Copy the ptr to leaves in array
        octreeIterator.gotoBottomLeft();
        int leafs = 0;
        do{
            iterArray[leafs++] = octreeIterator;
        } while(octreeIterator.moveRight());

        const int chunkSize = FMath::Max(1 , numberOfLeafs/(omp_get_max_threads()*omp_get_max_threads()));

        FLOG(computationCounter.tic());
#pragma omp parallel num_threads(MaxThreads)
        {
            // Each thread get its own kernel
            KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
            // Parallel iteration on the leaves
#pragma omp for nowait schedule(dynamic, userChunkSize)
            for(int idxLeafs = 0 ; idxLeafs < leafs ; ++idxLeafs){
                // We need the current cell that represent the leaf
                // and the list of particles
                ContainerClass* const sources = iterArray[idxLeafs].getCurrentListSrc();
                if(sources->getNbParticles()){
                    iterArray[idxLeafs].getCurrentCell()->setSrcChildTrue();
                    myThreadkernels->P2M( iterArray[idxLeafs].getCurrentCell() , sources);
                }
                if(iterArray[idxLeafs].getCurrentListTargets()->getNbParticles()){
                    iterArray[idxLeafs].getCurrentCell()->setTargetsChildTrue();
                }
            }
        }
        FLOG(computationCounter.tac());
        FLOG( FLog::Controller << "\tFinished (@Bottom Pass (P2M) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation:                " << computationCounter.elapsed() << " s\n" );
        FLOG( FLog::Controller.flush());
    }

    /////////////////////////////////////////////////////////////////////////////
    // Upward
    /////////////////////////////////////////////////////////////////////////////

    /** M2M */
    void upwardPass(){
        FLOG( FLog::Controller.write("\tStart Upward Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);
        FLOG(FTic singleCounter);
        FLOG(FTic parallelCounter);

        // Start from leal level (height-1)
        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();
        octreeIterator.moveUp();

        for(int idxLevel = OctreeHeight - 2 ; idxLevel > FAbstractAlgorithm::lowerWorkingLevel-1 ; --idxLevel){
            octreeIterator.moveUp();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);

        // The proc to send the shared cells to
        // Starting to the proc on the left this variable will go to 0
        int currentProcIdToSendTo = (idProcess - 1);

        // There are a maximum of 1 sends and 8-1 receptions
        MPI_Request requests[8];
        MPI_Status status[8];

        MPI_Request requestsSize[8];
        MPI_Status statusSize[8];

        FSize bufferSize;
        FMpiBufferWriter sendBuffer(1);// Max = 1 + sizeof(cell)*7
        std::unique_ptr<FMpiBufferReader[]> recvBuffer(new FMpiBufferReader[7]);
        FSize recvBufferSize[7];
        CellClass recvBufferCells[7];

        // The first proc that send to me a cell
        // This variable will go to nbProcess
        int firstProcThatSend = idProcess + 1;
        FLOG(computationCounter.tic());

        // for each levels
        for(int idxLevel = FMath::Min(OctreeHeight - 2, FAbstractAlgorithm::lowerWorkingLevel - 1) ; idxLevel >= FAbstractAlgorithm::upperWorkingLevel ; --idxLevel ){
            // Does my cells are covered by my neighbors working interval and so I have no more work?
            const bool noMoreWorkForMe = (idProcess != 0 && !procHasWorkAtLevel(idxLevel+1, idProcess));
            if(noMoreWorkForMe){
                FAssertLF(procHasWorkAtLevel(idxLevel, idProcess) == false);
                break;
            }

            // Copy and count ALL the cells (even the ones outside the working interval)
            int totalNbCellsAtLevel = 0;
            do{
                iterArray[totalNbCellsAtLevel++] = octreeIterator;
            } while(octreeIterator.moveRight());
            avoidGotoLeftIterator.moveUp();
            octreeIterator = avoidGotoLeftIterator;

            int iterMpiRequests       = 0; // The iterator for send/recv requests
            int iterMpiRequestsSize   = 0; // The iterator for send/recv requests

            int nbCellsToSkip     = 0; // The number of cells to send
            // Skip all the cells that are out of my working interval
            while(nbCellsToSkip < totalNbCellsAtLevel && iterArray[nbCellsToSkip].getCurrentGlobalIndex() < getWorkingInterval(idxLevel, idProcess).leftIndex){
                ++nbCellsToSkip;
            }

            // We need to know if we will recv something in order to know if threads skip the last cell
            int nbCellsForThreads = totalNbCellsAtLevel; // totalNbCellsAtLevel or totalNbCellsAtLevel-1
            bool hasToReceive = false;
            if(idProcess != nbProcess-1 && procHasWorkAtLevel(idxLevel , idProcess)){
                // Find the first proc that may send to me
                while(firstProcThatSend < nbProcess && !procHasWorkAtLevel(idxLevel+1, firstProcThatSend) ){
                    firstProcThatSend += 1;
                }
                // Do we have to receive?
                if(firstProcThatSend < nbProcess && procHasWorkAtLevel(idxLevel+1, firstProcThatSend) && procCoversMyRightBorderCell(idxLevel, firstProcThatSend) ){
                    hasToReceive = true;
                    // Threads do not compute the last cell, we will do it once data are received
                    nbCellsForThreads -= 1;
                }
            }

            FLOG(parallelCounter.tic());
#pragma omp parallel num_threads(MaxThreads)
            {
                KernelClass* myThreadkernels = (kernels[omp_get_thread_num()]);
                /////////////////////////////////////////////////////////////////////////////////////////
                //This single section post and receive the comms, and then do the M2M associated with it.
#pragma omp single nowait
                {
                    FLOG(singleCounter.tic());
                    // Master proc never send
                    if(idProcess != 0){
                        // Skip process that have no work at that level
                        while( currentProcIdToSendTo && !procHasWorkAtLevel(idxLevel, currentProcIdToSendTo)  ){
                            --currentProcIdToSendTo;
                        }
                        // Does the next proc that has work is sharing the parent of my left cell
                        if(procHasWorkAtLevel(idxLevel, currentProcIdToSendTo) && procCoversMyLeftBorderCell(idxLevel, currentProcIdToSendTo)){
                            FAssertLF(nbCellsToSkip != 0);

                            char packageFlags = 0;
                            sendBuffer.write(packageFlags);

                            // Only the cell the most on the right out of my working interval should be taken in
                            // consideration (at pos nbCellsToSkip-1) other (x < nbCellsToSkip-1) have already been sent
                            const CellClass* const* const child = iterArray[nbCellsToSkip-1].getCurrentChild();
                            for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
                                // Check if child exists and has sources  and it was part of my working interval
                                if( child[idxChild] && getWorkingInterval((idxLevel+1), idProcess).leftIndex <= child[idxChild]->getMortonIndex() ){
                                    if( child[idxChild]->hasSrcChild()){
                                        // Add the cell to the buffer
                                        sendBuffer << int(1);
                                    }
                                    else{
                                        sendBuffer << int(0);
                                    }
                                    if( child[idxChild]->hasTargetsChild()){
                                        // Add the cell to the buffer
                                        sendBuffer << int(1);
                                    }
                                    else{
                                        sendBuffer << int(0);
                                    }
                                    if( child[idxChild]->hasSrcChild() ){
                                        child[idxChild]->serializeUp(sendBuffer);
                                    }
                                    packageFlags = char(packageFlags | (0x1 << idxChild));
                                }
                            }
                            // Add the flag as first value
                            sendBuffer.writeAt(0,packageFlags);
                            // Post the message
                            bufferSize = sendBuffer.getSize();
                            MPI_Isend(&bufferSize, 1, FMpi::GetType(bufferSize), currentProcIdToSendTo,
                                      FMpi::TagFmmM2MSize + idxLevel, fcomCompute.getComm(), &requestsSize[iterMpiRequestsSize++]);
                            FAssertLF(sendBuffer.getSize() < std::numeric_limits<int>::max());
                            MPI_Isend(sendBuffer.data(), int(sendBuffer.getSize()), MPI_BYTE, currentProcIdToSendTo,
                                      FMpi::TagFmmM2M + idxLevel, fcomCompute.getComm(), &requests[iterMpiRequests++]);
                        }
                    }

                    //Post receive, Datas needed in several parts of the section
                    int nbProcThatSendToMe = 0;

                    if(hasToReceive){
                        //Test : if the firstProcThatSend father minimal value in interval is lesser than mine
                        int idProcSource = firstProcThatSend;
                        // Find the last proc that should send to me
                        while( idProcSource < nbProcess
                               && ( !procHasWorkAtLevel(idxLevel+1, idProcSource) || procCoversMyRightBorderCell(idxLevel, idProcSource) )){
                            if(procHasWorkAtLevel(idxLevel+1, idProcSource) && procCoversMyRightBorderCell(idxLevel, idProcSource)){
                                MPI_Irecv(&recvBufferSize[nbProcThatSendToMe], 1, FMpi::GetType(recvBufferSize[nbProcThatSendToMe]),
                                          idProcSource, FMpi::TagFmmM2MSize + idxLevel, fcomCompute.getComm(), &requestsSize[iterMpiRequestsSize++]);
                                nbProcThatSendToMe += 1;
                                FAssertLF(nbProcThatSendToMe <= 7);
                            }
                            ++idProcSource;
                        }
                    }

                    //Wait For the comms, and do the work
                    // Are we sending or waiting anything?
                    if(iterMpiRequestsSize){
                        FAssertLF(iterMpiRequestsSize <= 8);
                        MPI_Waitall( iterMpiRequestsSize, requestsSize, statusSize);
                    }

                    if(hasToReceive){
                        nbProcThatSendToMe = 0;
                        //Test : if the firstProcThatSend father minimal value in interval is lesser than mine
                        int idProcSource = firstProcThatSend;
                        // Find the last proc that should send to me
                        while( idProcSource < nbProcess
                               && ( !procHasWorkAtLevel(idxLevel+1, idProcSource) || procCoversMyRightBorderCell(idxLevel, idProcSource) )){
                            if(procHasWorkAtLevel(idxLevel+1, idProcSource) && procCoversMyRightBorderCell(idxLevel, idProcSource)){
                                recvBuffer[nbProcThatSendToMe].cleanAndResize(recvBufferSize[nbProcThatSendToMe]);
                                FAssertLF(recvBufferSize[nbProcThatSendToMe] < std::numeric_limits<int>::max());
                                MPI_Irecv(recvBuffer[nbProcThatSendToMe].data(), int(recvBufferSize[nbProcThatSendToMe]), MPI_BYTE,
                                          idProcSource, FMpi::TagFmmM2M + idxLevel, fcomCompute.getComm(), &requests[iterMpiRequests++]);
                                nbProcThatSendToMe += 1;
                                FAssertLF(nbProcThatSendToMe <= 7);
                            }
                            ++idProcSource;
                        }
                    }

                    //Wait For the comms, and do the work
                    // Are we sending or waiting anything?
                    if(iterMpiRequests){
                        FAssertLF(iterMpiRequests <= 8);
                        MPI_Waitall( iterMpiRequests, requests, status);
                    }

                    // We had received something so we need to proceed the last M2M
                    if( hasToReceive ){
                        FAssertLF(iterMpiRequests != 0);
                        CellClass* currentChild[8];
                        memcpy(currentChild, iterArray[totalNbCellsAtLevel - 1].getCurrentChild(), 8 * sizeof(CellClass*));

                        // Retreive data and merge my child and the child from others
                        int positionToInsert = 0;
                        int nbRemoteWithTgt = 0;
                        int nbRemoteWithSrc = 0;
                        for(int idxProc = 0 ; idxProc < nbProcThatSendToMe ; ++idxProc){
                            unsigned packageFlags = unsigned(recvBuffer[idxProc].getValue<unsigned char>());

                            int position = 0;
                            while( packageFlags && position < 8){
                                while(!(packageFlags & 0x1)){
                                    packageFlags >>= 1;
                                    ++position;
                                }
                                FAssertLF(positionToInsert < 7);
                                FAssertLF(position < 8);
                                FAssertLF(!currentChild[position], "Already has a cell here");
                                int hasSrcChild;
                                recvBuffer[idxProc] >> hasSrcChild;
                                FAssert(hasSrcChild == 0 || hasSrcChild == 1);
                                if(hasSrcChild) nbRemoteWithSrc += 1;
                                int hasTargChild;
                                recvBuffer[idxProc] >> hasTargChild;
                                FAssert(hasTargChild == 0 || hasTargChild == 1);
                                if(hasTargChild) nbRemoteWithTgt += 1;

                                if(hasSrcChild){
                                    recvBufferCells[positionToInsert].deserializeUp(recvBuffer[idxProc]);
                                    currentChild[position] = (CellClass*) &recvBufferCells[positionToInsert];
                                    position         += 1;
                                    positionToInsert += 1;
                                }

                                packageFlags     >>= 1;
                            }

                            recvBuffer[idxProc].seek(0);
                        }
                        //////////////////////////////////////////////////////////////////////////////////////////////////////
                        //                Finally one thread computes with distant cell

                        if(nbRemoteWithSrc){
                            iterArray[totalNbCellsAtLevel - 1].getCurrentCell()->setSrcChildTrue();
                            myThreadkernels->M2M( iterArray[totalNbCellsAtLevel - 1].getCurrentCell() , currentChild, idxLevel);
                        }
                        if(nbRemoteWithTgt){
                            iterArray[totalNbCellsAtLevel - 1].getCurrentCell()->setTargetsChildTrue();
                        }
                        firstProcThatSend += nbProcThatSendToMe - 1;
                    }
                    // Reset buffer
                    sendBuffer.reset();
                    FLOG(singleCounter.tac());
                }
                /////////////////////////////////////////////////////////////////////////////////////////
                //                             End Of Single section
                ////////////////////////////////////////////////////////////////////////////////////////
                //

                ////////////////////////////////////////////////////////////////////////////////////////
                //     All threads proceed the M2M with local cells
                //
                ////////////////////////////////////////////////////////////////////////////////////////
#pragma omp for nowait schedule(dynamic, userChunkSize)
                for( int idxCell = nbCellsToSkip ; idxCell < nbCellsForThreads ; ++idxCell){
                    // We need the current cell and the child
                    // child is an array (of 8 child) that may be null
                    CellClass* potentialChild[8];
                    CellClass** const realChild   = iterArray[idxCell].getCurrentChild();
                    CellClass*  const currentCell = iterArray[idxCell].getCurrentCell();
                    int nbChildWithSrc = 0;
                    for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
                        potentialChild[idxChild] = nullptr;
                        if(realChild[idxChild]){
                            if(realChild[idxChild]->hasSrcChild()){
                                nbChildWithSrc += 1;
                                potentialChild[idxChild] = realChild[idxChild];
                            }
                            if(realChild[idxChild]->hasTargetsChild()){
                                currentCell->setTargetsChildTrue();
                            }
                        }
                    }
                    if(nbChildWithSrc){
                        currentCell->setSrcChildTrue();
                        myThreadkernels->M2M( currentCell , potentialChild, idxLevel);
                    }
                }
                ////////////////////////////////////////////////////////////////////////////////////////

            }//End of parallel section
            FLOG(parallelCounter.tac());
        }

        FLOG(counterTime.tac());
        FLOG(computationCounter.tac());
        FLOG( FLog::Controller << "\tFinished (@Upward Pass (M2M) = "  << counterTime.elapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.elapsed() << " s\n" );
        FLOG( FLog::Controller << "\t\t Single : " << singleCounter.cumulated() << " s\n" );
        FLOG( FLog::Controller << "\t\t Parallel : " << parallelCounter.cumulated() << " s\n" );
        FLOG( FLog::Controller.flush());
    }

    /////////////////////////////////////////////////////////////////////////////
    // Downard
    /////////////////////////////////////////////////////////////////////////////


    void transferPass(){
        FLOG( FLog::Controller.write("\tStart Downward Pass (M2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);
        FLOG(FTic sendCounter);
        FLOG(FTic receiveCounter);
        FLOG(FTic prepareCounter);
        FLOG(FTic gatherCounter);

        //////////////////////////////////////////////////////////////////
        // First know what to send to who
        //////////////////////////////////////////////////////////////////

        // pointer to send
        std::unique_ptr<FVector<typename OctreeClass::Iterator>[]> toSend(new FVector<typename OctreeClass::Iterator>[nbProcess * OctreeHeight]);
        // index
        long long int*const indexToSend = new long long int[nbProcess * OctreeHeight];
        memset(indexToSend, 0, sizeof(long long int) * nbProcess * OctreeHeight);
        // To know which one has need someone
        FBoolArray** const leafsNeedOther = new FBoolArray*[OctreeHeight];
        memset(leafsNeedOther, 0, sizeof(FBoolArray*) * OctreeHeight);

        // All process say to each others
        // what the will send to who
        long long int*const globalReceiveMap = new long long  int[nbProcess * nbProcess * OctreeHeight];
        memset(globalReceiveMap, 0, sizeof(long long  int) * nbProcess * nbProcess * OctreeHeight);

        FMpiBufferWriter**const sendBuffer = new FMpiBufferWriter*[nbProcess * OctreeHeight];
        memset(sendBuffer, 0, sizeof(FMpiBufferWriter*) * nbProcess * OctreeHeight);

        FMpiBufferReader**const recvBuffer = new FMpiBufferReader*[nbProcess * OctreeHeight];
        memset(recvBuffer, 0, sizeof(FMpiBufferReader*) * nbProcess * OctreeHeight);

#pragma omp parallel num_threads(MaxThreads)
        {
#pragma omp master
            {
                {
                    FLOG(prepareCounter.tic());

                    std::unique_ptr<typename OctreeClass::Iterator[]> iterArrayLocal(new typename OctreeClass::Iterator[numberOfLeafs]);

                    // To know if a leaf has been already sent to a proc
                    bool*const alreadySent = new bool[nbProcess];
                    memset(alreadySent, 0, sizeof(bool) * nbProcess);

                    typename OctreeClass::Iterator octreeIterator(tree);
                    octreeIterator.moveDown();

                    for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
                        octreeIterator.moveDown();
                    }

                    typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);
                    // for each levels
                    for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel ){

                        const int separationCriteria = (idxLevel != FAbstractAlgorithm::lowerWorkingLevel-1 ? 1 : leafLevelSeparationCriteria);

                        if(!procHasWorkAtLevel(idxLevel, idProcess)){
                            avoidGotoLeftIterator.moveDown();
                            octreeIterator = avoidGotoLeftIterator;
                            continue;
                        }

                        int numberOfCells = 0;

                        while(octreeIterator.getCurrentGlobalIndex() <  getWorkingInterval(idxLevel , idProcess).leftIndex){
                            octreeIterator.moveRight();
                        }

                        // for each cells
                        do{
                            iterArrayLocal[numberOfCells] = octreeIterator;
                            ++numberOfCells;
                        } while(octreeIterator.moveRight());
                        avoidGotoLeftIterator.moveDown();
                        octreeIterator = avoidGotoLeftIterator;

                        leafsNeedOther[idxLevel] = new FBoolArray(numberOfCells);

                        // Which cell potentialy needs other data and in the same time
                        // are potentialy needed by other
                        MortonIndex neighborsIndexes[/*189+26+1*/216];
                        for(int idxCell = 0 ; idxCell < numberOfCells ; ++idxCell){
                            // Find the M2L neigbors of a cell
                            const int counter = iterArrayLocal[idxCell].getCurrentGlobalCoordinate().getInteractionNeighbors(idxLevel, neighborsIndexes, separationCriteria);

                            memset(alreadySent, false, sizeof(bool) * nbProcess);
                            bool needOther = false;
                            // Test each negibors to know which one do not belong to us
                            for(int idxNeigh = 0 ; idxNeigh < counter ; ++idxNeigh){
                                if(neighborsIndexes[idxNeigh] < getWorkingInterval(idxLevel , idProcess).leftIndex
                                        || (getWorkingInterval(idxLevel , idProcess).rightIndex) < neighborsIndexes[idxNeigh]){
                                    int procToReceive = idProcess;
                                    while( 0 != procToReceive && neighborsIndexes[idxNeigh] < getWorkingInterval(idxLevel , procToReceive).leftIndex ){
                                        --procToReceive;
                                    }
                                    while( procToReceive != nbProcess -1 && (getWorkingInterval(idxLevel , procToReceive).rightIndex) < neighborsIndexes[idxNeigh]){
                                        ++procToReceive;
                                    }
                                    // Maybe already sent to that proc?
                                    if( !alreadySent[procToReceive]
                                            && getWorkingInterval(idxLevel , procToReceive).leftIndex <= neighborsIndexes[idxNeigh]
                                            && neighborsIndexes[idxNeigh] <= getWorkingInterval(idxLevel , procToReceive).rightIndex){

                                        needOther = true;

                                        if(iterArrayLocal[idxCell].getCurrentCell()->hasSrcChild()){
                                            alreadySent[procToReceive] = true;


                                            toSend[idxLevel * nbProcess + procToReceive].push(iterArrayLocal[idxCell]);
                                            if(indexToSend[idxLevel * nbProcess + procToReceive] == 0){
                                                indexToSend[idxLevel * nbProcess + procToReceive] = sizeof(int);
                                            }
                                            indexToSend[idxLevel * nbProcess + procToReceive] += iterArrayLocal[idxCell].getCurrentCell()->getSavedSizeUp();
                                            indexToSend[idxLevel * nbProcess + procToReceive] += sizeof(MortonIndex);
                                            indexToSend[idxLevel * nbProcess + procToReceive] += sizeof(FSize);
                                        }
                                        else{
                                            // We have not sources so we will not send it,
                                            // but we need to know if needOther is true
                                            break;
                                        }
                                    }
                                }
                            }
                            if(needOther && iterArrayLocal[idxCell].getCurrentCell()->hasTargetsChild()){
                                // only keep those with targets in the list
                                leafsNeedOther[idxLevel]->set(idxCell,true);
                            }
                        }
                    }
                    FLOG(prepareCounter.tac());

                    delete[] alreadySent;
                }

                //////////////////////////////////////////////////////////////////
                // Gather this information
                //////////////////////////////////////////////////////////////////

                FLOG(gatherCounter.tic());
                FMpi::MpiAssert( MPI_Allgather( indexToSend, nbProcess * OctreeHeight, MPI_LONG_LONG_INT, globalReceiveMap, nbProcess * OctreeHeight, MPI_LONG_LONG_INT, fcomCompute.getComm()),  __LINE__ );
                FLOG(gatherCounter.tac());

                //////////////////////////////////////////////////////////////////
                // Send and receive for real
                //////////////////////////////////////////////////////////////////

                FLOG(sendCounter.tic());
                // Then they can send and receive (because they know what they will receive)
                // To send in asynchrone way
                std::vector<MPI_Request> requests;
                requests.reserve(2 * nbProcess * OctreeHeight);

                for(int idxLevel = 2 ; idxLevel < OctreeHeight ; ++idxLevel ){
                    for(int idxProc = 0 ; idxProc < nbProcess ; ++idxProc){
                        const long long int toSendAtProcAtLevel = indexToSend[idxLevel * nbProcess + idxProc];
                        if(toSendAtProcAtLevel != 0){
                            sendBuffer[idxLevel * nbProcess + idxProc] = new FMpiBufferWriter(toSendAtProcAtLevel);

                            sendBuffer[idxLevel * nbProcess + idxProc]->write(int(toSend[idxLevel * nbProcess + idxProc].getSize()));

                            for(int idxLeaf = 0 ; idxLeaf < toSend[idxLevel * nbProcess + idxProc].getSize(); ++idxLeaf){
                                const FSize currentTell = sendBuffer[idxLevel * nbProcess + idxProc]->getSize();
                                sendBuffer[idxLevel * nbProcess + idxProc]->write(currentTell);
                                const MortonIndex cellIndex = toSend[idxLevel * nbProcess + idxProc][idxLeaf].getCurrentGlobalIndex();
                                sendBuffer[idxLevel * nbProcess + idxProc]->write(cellIndex);
                                toSend[idxLevel * nbProcess + idxProc][idxLeaf].getCurrentCell()->serializeUp(*sendBuffer[idxLevel * nbProcess + idxProc]);
                            }

                            FAssertLF(sendBuffer[idxLevel * nbProcess + idxProc]->getSize() == toSendAtProcAtLevel);

                            FMpi::ISendSplit(sendBuffer[idxLevel * nbProcess + idxProc]->data(),
                                    sendBuffer[idxLevel * nbProcess + idxProc]->getSize(), idxProc,
                                    FMpi::TagLast + idxLevel*100, fcomCompute, &requests);
                        }

                        const long long int toReceiveFromProcAtLevel = globalReceiveMap[(idxProc * nbProcess * OctreeHeight) + idxLevel * nbProcess + idProcess];
                        if(toReceiveFromProcAtLevel){
                            recvBuffer[idxLevel * nbProcess + idxProc] = new FMpiBufferReader(toReceiveFromProcAtLevel);

                            FMpi::IRecvSplit(recvBuffer[idxLevel * nbProcess + idxProc]->data(),
                                    recvBuffer[idxLevel * nbProcess + idxProc]->getCapacity(), idxProc,
                                    FMpi::TagLast + idxLevel*100, fcomCompute, &requests);
                        }
                    }
                }

                //////////////////////////////////////////////////////////////////
                // Wait received data and compute
                //////////////////////////////////////////////////////////////////

                // Wait to receive every things (and send every things)
                FMpi::MpiAssert(MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUS_IGNORE), __LINE__);

                FLOG(sendCounter.tac());
            }//End of Master region

            //////////////////////////////////////////////////////////////////
            // Do M2L
            //////////////////////////////////////////////////////////////////

#pragma omp single nowait
            {
                typename OctreeClass::Iterator octreeIterator(tree);
                octreeIterator.moveDown();

                for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
                    octreeIterator.moveDown();
                }

                typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);
                // Now we can compute all the data
                // for each levels
                for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel ){
                    const int separationCriteria = (idxLevel != FAbstractAlgorithm::lowerWorkingLevel-1 ? 1 : leafLevelSeparationCriteria);

                    if(!procHasWorkAtLevel(idxLevel, idProcess)){
                        avoidGotoLeftIterator.moveDown();
                        octreeIterator = avoidGotoLeftIterator;
                        continue;
                    }

                    int numberOfCells = 0;
                    while(octreeIterator.getCurrentGlobalIndex() <  getWorkingInterval(idxLevel , idProcess).leftIndex){
                        octreeIterator.moveRight();
                    }
                    // for each cells
                    do{
                        iterArray[numberOfCells] = octreeIterator;
                        ++numberOfCells;
                    } while(octreeIterator.moveRight());
                    avoidGotoLeftIterator.moveDown();
                    octreeIterator = avoidGotoLeftIterator;

                    FLOG(computationCounter.tic());
                    {
                        const int chunckSize = userChunkSize;
                        for(int idxCell = 0 ; idxCell < numberOfCells ; idxCell += chunckSize){
#pragma omp task default(none) shared(numberOfCells,idxLevel) firstprivate(idxCell) //+ shared(chunckSize)
                            {
                                KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
                                const CellClass* neighbors[342];
                                int neighborPositions[342];

                                const int nbCellToCompute = FMath::Min(chunckSize, numberOfCells-idxCell);
                                for(int idxCellToCompute = idxCell ; idxCellToCompute < idxCell+nbCellToCompute ; ++idxCellToCompute){
                                    if(iterArray[idxCellToCompute].getCurrentCell()->hasTargetsChild()){
                                        const int counter = tree->getInteractionNeighbors(neighbors,  neighborPositions, iterArray[idxCellToCompute].getCurrentGlobalCoordinate(), idxLevel, separationCriteria);
                                        if(counter){
                                            int counterWithSrc = 0;
                                            for(int idxRealNeighbors = 0 ; idxRealNeighbors < counter ; ++idxRealNeighbors ){
                                                if(neighbors[idxRealNeighbors]->hasSrcChild()){
                                                    neighbors[counterWithSrc] = neighbors[idxRealNeighbors];
                                                    neighborPositions[counterWithSrc] = neighborPositions[idxRealNeighbors];
                                                    ++counterWithSrc;
                                                }
                                            }
                                            myThreadkernels->M2L( iterArray[idxCellToCompute].getCurrentCell() , neighbors, neighborPositions, counterWithSrc, idxLevel);
                                        }
                                    }
                                }
                            }
                        }
                    }//End of task spawning

#pragma omp taskwait

                    for(int idxThread = 0 ; idxThread < omp_get_num_threads() ; ++idxThread){
#pragma omp task  default(none) firstprivate(idxThread,idxLevel)
                        {
                            kernels[idxThread]->finishedLevelM2L(idxLevel);
                        }
                    }
#pragma omp taskwait

                    FLOG(computationCounter.tac());
                }
            }
        }


        {
            FLOG(receiveCounter.tic());
            typename OctreeClass::Iterator octreeIterator(tree);
            octreeIterator.moveDown();

            for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
                octreeIterator.moveDown();
            }

            typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);
            // compute the second time
            // for each levels
            for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel ){
                const int separationCriteria = (idxLevel != FAbstractAlgorithm::lowerWorkingLevel-1 ? 1 : leafLevelSeparationCriteria);

                if(!procHasWorkAtLevel(idxLevel, idProcess)){
                    avoidGotoLeftIterator.moveDown();
                    octreeIterator = avoidGotoLeftIterator;
                    continue;
                }

                // put the received data into a temporary tree
                FLightOctree<CellClass> tempTree;
                for(int idxProc = 0 ; idxProc < nbProcess ; ++idxProc){
                    if(recvBuffer[idxLevel * nbProcess + idxProc]){
                        const int toReceiveFromProcAtLevel = recvBuffer[idxLevel * nbProcess + idxProc]->template getValue<int>();

                        for(int idxCell = 0 ; idxCell < toReceiveFromProcAtLevel ; ++idxCell){
                            const FSize currentTell = recvBuffer[idxLevel * nbProcess + idxProc]->tell();
                            const FSize verifCurrentTell = recvBuffer[idxLevel * nbProcess + idxProc]->template getValue<FSize>();
                            FAssertLF(currentTell == verifCurrentTell, currentTell, " ", verifCurrentTell);

                            const MortonIndex cellIndex = recvBuffer[idxLevel * nbProcess + idxProc]->template getValue<MortonIndex>();

                            CellClass* const newCell = new CellClass;
                            newCell->setMortonIndex(cellIndex);
                            newCell->deserializeUp(*recvBuffer[idxLevel * nbProcess + idxProc]);

                            tempTree.insertCell(cellIndex, idxLevel, newCell);
                        }

                        FAssertLF(globalReceiveMap[(idxProc * nbProcess * OctreeHeight) + idxLevel * nbProcess + idProcess] ==
                                recvBuffer[idxLevel * nbProcess + idxProc]->tell());
                    }
                }

                // take cells from our octree only if they are
                // linked to received data
                int numberOfCells = 0;
                int realCellId = 0;

                while(octreeIterator.getCurrentGlobalIndex() <  getWorkingInterval(idxLevel , idProcess).leftIndex){
                    octreeIterator.moveRight();
                }
                // for each cells
                do{
                    // copy cells that need data from others
                    if(leafsNeedOther[idxLevel]->get(realCellId++)){
                        iterArray[numberOfCells++] = octreeIterator;
                    }
                } while(octreeIterator.moveRight());
                avoidGotoLeftIterator.moveDown();
                octreeIterator = avoidGotoLeftIterator;

                delete leafsNeedOther[idxLevel];
                leafsNeedOther[idxLevel] = nullptr;

                // Compute this cells
                FLOG(computationCounter.tic());
#pragma omp parallel num_threads(MaxThreads)
                {
                    KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
                    MortonIndex neighborsIndex[/*189+26+1*/216];
                    int neighborsPosition[/*189+26+1*/216];
                    const CellClass* neighbors[342];
                    int neighborPositions[342];

#pragma omp for  schedule(dynamic, userChunkSize) nowait
                    for(int idxCell = 0 ; idxCell < numberOfCells ; ++idxCell){
                        // compute indexes
                        const int counterNeighbors = iterArray[idxCell].getCurrentGlobalCoordinate().getInteractionNeighbors(idxLevel, neighborsIndex, neighborsPosition, separationCriteria);

                        FAssertLF(iterArray[idxCell].getCurrentCell()->hasTargetsChild());

                        int counter = 0;
                        // does we receive this index from someone?
                        for(int idxNeig = 0 ;idxNeig < counterNeighbors ; ++idxNeig){
                            if(neighborsIndex[idxNeig] < (getWorkingInterval(idxLevel , idProcess).leftIndex)
                                    || (getWorkingInterval(idxLevel , idProcess).rightIndex) < neighborsIndex[idxNeig]){

                                CellClass*const otherCell = tempTree.getCell(neighborsIndex[idxNeig], idxLevel);

                                if(otherCell){
                                    neighbors[counter] = otherCell;
                                    neighborPositions[counter] = neighborsPosition[idxNeig];
                                    ++counter;
                                }
                            }
                        }
                        // need to compute
                        if(counter){
                            myThreadkernels->M2L( iterArray[idxCell].getCurrentCell() , neighbors, neighborPositions, counter, idxLevel);
                        }
                    }

                    myThreadkernels->finishedLevelM2L(idxLevel);
                }
                FLOG(computationCounter.tac());
            }
            FLOG(receiveCounter.tac());
        }

        for(int idxComm = 0 ; idxComm < nbProcess * OctreeHeight; ++idxComm){
            delete sendBuffer[idxComm];
            delete recvBuffer[idxComm];
        }
        for(int idxComm = 0 ; idxComm < OctreeHeight; ++idxComm){
            delete leafsNeedOther[idxComm];
        }
        delete[] sendBuffer;
        delete[] recvBuffer;
        delete[] indexToSend;
        delete[] leafsNeedOther;
        delete[] globalReceiveMap;


        FLOG( FLog::Controller << "\tFinished (@Downward Pass (M2L) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.cumulated() << " s\n" );
        FLOG( FLog::Controller << "\t\t Send : " << sendCounter.cumulated() << " s\n" );
        FLOG( FLog::Controller << "\t\t Receive : " << receiveCounter.cumulated() << " s\n" );
        FLOG( FLog::Controller << "\t\t Gather : " << gatherCounter.cumulated() << " s\n" );
        FLOG( FLog::Controller << "\t\t Prepare : " << prepareCounter.cumulated() << " s\n" );
        FLOG( FLog::Controller.flush());

    }

    //////////////////////////////////////////////////////////////////
    // ---------------- L2L ---------------
    //////////////////////////////////////////////////////////////////

    void downardPass(){ // second L2L
        FLOG( FLog::Controller.write("\tStart Downward Pass (L2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);
        FLOG(FTic prepareCounter);
        FLOG(FTic waitCounter);

        // Start from leal level - 1
        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.moveDown();

        for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
            octreeIterator.moveDown();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);

        // Max 1 receive and 7 send (but 7 times the same data)
        MPI_Request*const requests     = new MPI_Request[8];
        MPI_Request*const requestsSize = new MPI_Request[8];

        const int heightMinusOne = FAbstractAlgorithm::lowerWorkingLevel - 1;

        FMpiBufferWriter sendBuffer;
        FMpiBufferReader recvBuffer;

        int righestProcToSendTo   = nbProcess - 1;

        // for each levels exepted leaf level
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < heightMinusOne ; ++idxLevel ){
            // If nothing to do in the next level skip the current one
            if(idProcess != 0 && !procHasWorkAtLevel(idxLevel+1, idProcess) ){
                avoidGotoLeftIterator.moveDown();
                octreeIterator = avoidGotoLeftIterator;
                continue;
            }

            // Copy all the cells in an array even the one that are out of my working interval
            int totalNbCellsAtLevel = 0;
            do{
                iterArray[totalNbCellsAtLevel++] = octreeIterator;
            } while(octreeIterator.moveRight());
            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;

            // Count the number of cells that are out of my working interval
            int nbCellsToSkip = 0;
            while(nbCellsToSkip < totalNbCellsAtLevel && iterArray[nbCellsToSkip].getCurrentGlobalIndex() < getWorkingInterval(idxLevel , idProcess).leftIndex){
                nbCellsToSkip += 1;
            }

            // Check if someone will send a cell to me
            bool hasToReceive    = false;
            int idxProcToReceive = idProcess - 1;
            if(idProcess != 0 && nbCellsToSkip){
                // Starting from my left neighbor stop at the first proc that has work to do (not null interval)
                while(idxProcToReceive && !procHasWorkAtLevel(idxLevel, idxProcToReceive) ){
                    idxProcToReceive -= 1;
                }
                // Check if we find such a proc and that it share a cell with us on the border
                if(procHasWorkAtLevel(idxLevel, idxProcToReceive) && procCoversMyLeftBorderCell(idxLevel, idxProcToReceive)){
                    hasToReceive = true;
                }
            }

#pragma omp parallel num_threads(MaxThreads)
            {
                KernelClass* myThreadkernels = (kernels[omp_get_thread_num()]);
#pragma omp single nowait
                {
                    FLOG(prepareCounter.tic());
                    int iterRequests = 0;
                    int iterRequestsSize = 0;
                    FSize recvBufferSize = 0;
                    FSize sendBufferSize;
                    // Post the receive
                    if(hasToReceive){
                        FMpi::MpiAssert( MPI_Irecv( &recvBufferSize, 1, FMpi::GetType(recvBufferSize), idxProcToReceive,
                                                    FMpi::TagFmmL2LSize + idxLevel, fcomCompute.getComm(), &requestsSize[iterRequestsSize++]), __LINE__ );
                    }

                    // We have to be sure that we are not sending if we have no work in the current level
                    if(idProcess != nbProcess - 1 && idProcess < righestProcToSendTo && procHasWorkAtLevel(idxLevel, idProcess)){
                        int idxProcSend = idProcess + 1;
                        int nbMessageSent = 0;
                        // From the proc on the right to righestProcToSendTo, check if we have to send something
                        while(idxProcSend <= righestProcToSendTo && ( !procHasWorkAtLevel(idxLevel+1, idxProcSend) || procCoversMyRightBorderCell(idxLevel, idxProcSend)) ){
                            // We know that if the proc has work at the next level it share a cell with us due to the while condition
                            if(procHasWorkAtLevel(idxLevel+1, idxProcSend)){
                                FAssertLF(procCoversMyRightBorderCell(idxLevel, idxProcSend));
                                // If first message then serialize the cell to send
                                if( nbMessageSent == 0 ){
                                    if(iterArray[totalNbCellsAtLevel - 1].getCurrentCell()->hasTargetsChild()){
                                        sendBuffer << int(1);
                                        // We send our last cell
                                        iterArray[totalNbCellsAtLevel - 1].getCurrentCell()->serializeDown(sendBuffer);
                                    }
                                    else{
                                        sendBuffer << int(0);
                                    }
                                    sendBufferSize = sendBuffer.getSize();
                                }
                                // Post the send message
                                FMpi::MpiAssert( MPI_Isend(&sendBufferSize, 1, FMpi::GetType(sendBufferSize), idxProcSend,
                                                           FMpi::TagFmmL2LSize + idxLevel, fcomCompute.getComm(), &requestsSize[iterRequestsSize++]), __LINE__);
                                FAssertLF(sendBuffer.getSize() < std::numeric_limits<int>::max());
                                FMpi::MpiAssert( MPI_Isend(sendBuffer.data(), int(sendBuffer.getSize()), MPI_BYTE, idxProcSend,
                                                           FMpi::TagFmmL2L + idxLevel, fcomCompute.getComm(), &requests[iterRequests++]), __LINE__);
                                // Inc and check the counter
                                nbMessageSent += 1;
                                FAssertLF(nbMessageSent <= 7);
                            }
                            idxProcSend += 1;
                        }
                        // Next time we will not need to go further than idxProcSend
                        righestProcToSendTo = idxProcSend;
                    }

                    // Finalize the communication
                    if(iterRequestsSize){
                        FLOG(waitCounter.tic());
                        FAssertLF(iterRequestsSize <= 8);
                        FMpi::MpiAssert(MPI_Waitall( iterRequestsSize, requestsSize, MPI_STATUSES_IGNORE), __LINE__);
                        FLOG(waitCounter.tac());
                    }

                    if(hasToReceive){
                        recvBuffer.cleanAndResize(recvBufferSize);
                        FAssertLF(recvBuffer.getCapacity() < std::numeric_limits<int>::max());
                        FMpi::MpiAssert( MPI_Irecv( recvBuffer.data(), int(recvBuffer.getCapacity()), MPI_BYTE, idxProcToReceive,
                                                    FMpi::TagFmmL2L + idxLevel, fcomCompute.getComm(), &requests[iterRequests++]), __LINE__ );
                    }

                    if(iterRequests){
                        FLOG(waitCounter.tic());
                        FAssertLF(iterRequests <= 8);
                        FMpi::MpiAssert(MPI_Waitall( iterRequests, requests, MPI_STATUSES_IGNORE), __LINE__);
                        FLOG(waitCounter.tac());
                    }

                    // If we receive something proceed the L2L
                    if(hasToReceive){
                        FAssertLF(iterRequests != 0);
                        // In this case we know that we have to perform the L2L with the last cell that are
                        // exclude from our working interval nbCellsToSkip-1
                        int hasTgtChild;
                        recvBuffer >> hasTgtChild;
                        FAssertLF(hasTgtChild == 0 || hasTgtChild == 1);
                        if(hasTgtChild){
                            iterArray[nbCellsToSkip-1].getCurrentCell()->deserializeDown(recvBuffer);
                            CellClass* potentialChild[8];
                            CellClass** const realChild   = iterArray[nbCellsToSkip-1].getCurrentChild();
                            CellClass* const currentCell  = iterArray[nbCellsToSkip-1].getCurrentCell();
                            int nbTgtChild = 0;
                            for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
                                if(realChild[idxChild] && realChild[idxChild]->hasTargetsChild()){
                                    nbTgtChild += 1;
                                    potentialChild[idxChild] = realChild[idxChild];
                                }
                                else{
                                    potentialChild[idxChild] = nullptr;
                                }
                            }
                            if(nbTgtChild){
                                myThreadkernels->L2L( currentCell , potentialChild, idxLevel);
                            }
                        }
                        //		      myThreadkernels->L2L( iterArray[nbCellsToSkip-1].getCurrentCell() , iterArray[nbCellsToSkip-1].getCurrentChild(), idxLevel);
                    }
                    FLOG(prepareCounter.tac());
                }

#pragma omp single nowait
                {
                    FLOG(computationCounter.tic());
                }
                // Threads are working on all the cell of our working interval at that level
#pragma omp for nowait  schedule(dynamic, userChunkSize)
                for(int idxCell = nbCellsToSkip ; idxCell < totalNbCellsAtLevel ; ++idxCell){
                    if( iterArray[idxCell].getCurrentCell()->hasTargetsChild() ){
                        CellClass* potentialChild[8];
                        CellClass** const realChild   = iterArray[idxCell].getCurrentChild();
                        CellClass* const currentCell  = iterArray[idxCell].getCurrentCell();
                        int nbTgtChild = 0;
                        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
                            if(realChild[idxChild] && realChild[idxChild]->hasTargetsChild()){
                                potentialChild[idxChild] = realChild[idxChild];
                                nbTgtChild += 1;
                            }
                            else{
                                potentialChild[idxChild] = nullptr;
                            }
                        }
                        if(nbTgtChild){
                            myThreadkernels->L2L( currentCell , potentialChild, idxLevel);
                        }
                        // Old form                    myThreadkernels->L2L( iterArray[idxCell].getCurrentCell() , iterArray[idxCell].getCurrentChild(), idxLevel);
                    }
                }
            }
            FLOG(computationCounter.tac());

            sendBuffer.reset();
            recvBuffer.seek(0);
        }
        delete[] requests;
        delete[] requestsSize;

        FLOG( FLog::Controller << "\tFinished (@Downward Pass (L2L) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.cumulated() << " s\n" );
        FLOG( FLog::Controller << "\t\t Prepare : " << prepareCounter.cumulated() << " s\n" );
        FLOG( FLog::Controller << "\t\t Wait : " << waitCounter.cumulated() << " s\n" );
        FLOG( FLog::Controller.flush());
    }


    /////////////////////////////////////////////////////////////////////////////
    // Direct
    /////////////////////////////////////////////////////////////////////////////
    struct LeafData{
        FTreeCoordinate coord;
        CellClass*      cell;
        ContainerClass* targets;
        ContainerClass* sources;
    };


    /** P2P */
    void directPass(const bool p2pEnabled, const bool l2pEnabled){
        FLOG( FLog::Controller.write("\tStart Direct Pass\n").write(FLog::Flush); );
        FLOG( FTic counterTime);
        FLOG( FTic prepareCounter);
        FLOG( FTic gatherCounter);
        FLOG( FTic waitCounter);
        FLOG(FTic computationCounter);
        FLOG(FTic computation2Counter);

        ///////////////////////////////////////////////////
        // Prepare data to send receive
        ///////////////////////////////////////////////////
        FLOG(prepareCounter.tic());

        FBoolArray leafsNeedOther(this->numberOfLeafs);
        int countNeedOther = 0;

        // To store the result
        OctreeClass otherP2Ptree( tree->getHeight(), tree->getSubHeight(), tree->getBoxWidth(), tree->getBoxCenter() );

        // init
        const int SizeShape = P2PExclusionClass::SizeShape;

        int shapeLeaf[SizeShape];
        memset(shapeLeaf,0,SizeShape*sizeof(int));

        LeafData* const leafsDataArray = new LeafData[this->numberOfLeafs];

        FVector<LeafData> leafsNeedOtherData(countNeedOther);

        FVector<typename OctreeClass::Iterator>*const toSend = new FVector<typename OctreeClass::Iterator>[nbProcess];
        FSize partsToSend[nbProcess];
        memset(partsToSend, 0, sizeof(FSize) * nbProcess);

#pragma omp parallel num_threads(MaxThreads)
        {
#pragma omp master // MUST WAIT to fill leafsNeedOther
            if(p2pEnabled){
                // Copy leafs
                {
                    typename OctreeClass::Iterator octreeIterator(tree);
                    octreeIterator.gotoBottomLeft();
                    int idxLeaf = 0;
                    do{
                        this->iterArray[idxLeaf++] = octreeIterator;
                    } while(octreeIterator.moveRight());
                }

                int alreadySent[nbProcess];

                //Will store the indexes of the neighbors of current cell
                MortonIndex indexesNeighbors[26];

                for(int idxLeaf = 0 ; idxLeaf < this->numberOfLeafs ; ++idxLeaf){
                    memset(alreadySent, 0, sizeof(int) * nbProcess);
                    bool needOther = false;
                    //Get the neighbors of current cell in indexesNeighbors, and their number in neighCount
                    const int neighCount = (iterArray[idxLeaf].getCurrentGlobalCoordinate()).getNeighborsIndexes(OctreeHeight,indexesNeighbors);
                    //Loop over the neighbor leafs
                    for(int idxNeigh = 0 ; idxNeigh < neighCount ; ++idxNeigh){
                        //Test if leaf belongs to someone else (false if it's mine)
                        if(indexesNeighbors[idxNeigh] < (intervals[idProcess].leftIndex) || (intervals[idProcess].rightIndex) < indexesNeighbors[idxNeigh]){
                            needOther = true;

                            if(iterArray[idxLeaf].getCurrentListSrc()->getNbParticles()){
                                // find the proc that will need current leaf
                                int procToReceive = idProcess;
                                while( procToReceive != 0 && indexesNeighbors[idxNeigh] < intervals[procToReceive].leftIndex){
                                    --procToReceive; //scroll process "before" current process
                                }

                                while( procToReceive != nbProcess - 1 && (intervals[procToReceive].rightIndex) < indexesNeighbors[idxNeigh]){
                                    ++procToReceive;//scroll process "after" current process
                                }
                                //  Test : Not Already Send && be sure someone hold this interval
                                if( !alreadySent[procToReceive] && intervals[procToReceive].leftIndex <= indexesNeighbors[idxNeigh]
                                        && indexesNeighbors[idxNeigh] <= intervals[procToReceive].rightIndex){
                                    alreadySent[procToReceive] = 1;
                                    toSend[procToReceive].push( iterArray[idxLeaf] );
                                    partsToSend[procToReceive] += iterArray[idxLeaf].getCurrentListSrc()->getSavedSize();
                                    partsToSend[procToReceive] += int(sizeof(MortonIndex));
                                }
                            }
                            else{
                                break;
                            }
                        }
                    }

                    if(iterArray[idxLeaf].getCurrentListTargets()->getNbParticles() && needOther){ //means that something need to be sent (or received)
                        leafsNeedOther.set(idxLeaf,true);
                        ++countNeedOther;
                    }
                }

                // No idea why it is mandatory there, could it be a few line before,
                for(int idxProc = 0 ; idxProc < nbProcess ; ++idxProc){
                    if(partsToSend[idxProc]){
                        partsToSend[idxProc] += int(sizeof(FSize));
                    }
                }
            }

#pragma omp barrier

#pragma omp master // nowait
            if(p2pEnabled){
                /* This a nbProcess x nbProcess matrix of integer
             * let U and V be id of processes :
             * globalReceiveMap[U*nbProcess + V] == size of information needed by V and own by U
             */
                FSize*const globalReceiveMap = new FSize[nbProcess * nbProcess];
                memset(globalReceiveMap, 0, sizeof(FSize) * nbProcess * nbProcess);

                //Share to all processus globalReceiveMap
                FLOG(gatherCounter.tic());
                FMpi::MpiAssert( MPI_Allgather( partsToSend, nbProcess, FMpi::GetType(*partsToSend),
                                                globalReceiveMap, nbProcess, FMpi::GetType(*partsToSend), fcomCompute.getComm()),  __LINE__ );
                FLOG(gatherCounter.tac());

                FMpiBufferReader**const recvBuffer = new FMpiBufferReader*[nbProcess];
                memset(recvBuffer, 0, sizeof(FMpiBufferReader*) * nbProcess);

                FMpiBufferWriter**const sendBuffer = new FMpiBufferWriter*[nbProcess];
                memset(sendBuffer, 0, sizeof(FMpiBufferWriter*) * nbProcess);

                // To send in asynchrone way
                std::vector<MPI_Request> requests;
                requests.reserve(2 * nbProcess);
                //Prepare receive
                for(int idxProc = 0 ; idxProc < nbProcess ; ++idxProc){
                    if(globalReceiveMap[idxProc * nbProcess + idProcess]){ //if idxProc has sth for me.
                        //allocate buffer of right size
                        recvBuffer[idxProc] = new FMpiBufferReader(globalReceiveMap[idxProc * nbProcess + idProcess]);

                        FMpi::IRecvSplit(recvBuffer[idxProc]->data(), recvBuffer[idxProc]->getCapacity(),
                                         idxProc, FMpi::TagFmmP2P, fcomCompute, &requests);
                    }
                }

                // Prepare send
                for(int idxProc = 0 ; idxProc < nbProcess ; ++idxProc){
                    if(toSend[idxProc].getSize() != 0){
                        sendBuffer[idxProc] = new FMpiBufferWriter(globalReceiveMap[idProcess*nbProcess+idxProc]);
                        // << is equivalent to write().
                        (*sendBuffer[idxProc]) << toSend[idxProc].getSize();
                        for(int idxLeaf = 0 ; idxLeaf < toSend[idxProc].getSize() ; ++idxLeaf){
                            (*sendBuffer[idxProc]) << toSend[idxProc][idxLeaf].getCurrentGlobalIndex();
                            toSend[idxProc][idxLeaf].getCurrentListSrc()->save(*sendBuffer[idxProc]);
                        }

                        FAssertLF(sendBuffer[idxProc]->getSize() == globalReceiveMap[idProcess*nbProcess+idxProc]);

                        FMpi::ISendSplit(sendBuffer[idxProc]->data(), sendBuffer[idxProc]->getSize(),
                                         idxProc, FMpi::TagFmmP2P, fcomCompute, &requests);

                    }
                }

                delete[] toSend;


                //////////////////////////////////////////////////////////
                // Waitsend receive
                //////////////////////////////////////////////////////////

                std::unique_ptr<MPI_Status[]> status(new MPI_Status[requests.size()]);
                // Wait data
                FLOG(waitCounter.tic());
                MPI_Waitall(int(requests.size()), requests.data(), status.get());
                FLOG(waitCounter.tac());

                for(int idxProc = 0 ; idxProc < nbProcess ; ++idxProc){
                    if(globalReceiveMap[idxProc * nbProcess + idProcess]){ //if idxProc has sth for me.
                        FAssertLF(recvBuffer[idxProc]);
                        FMpiBufferReader& currentBuffer = (*recvBuffer[idxProc]);
                        FSize nbLeaves;
                        currentBuffer >> nbLeaves;
                        for(FSize idxLeaf = 0 ; idxLeaf < nbLeaves ; ++idxLeaf){
                            MortonIndex leafIndex;
                            currentBuffer >> leafIndex;
                            otherP2Ptree.createLeaf(leafIndex)->getSrc()->restore(currentBuffer);
                        }
                        // Realease memory early
                        delete recvBuffer[idxProc];
                        recvBuffer[idxProc] = nullptr;
                    }
                }

                for(int idxProc = 0 ; idxProc < nbProcess ; ++idxProc){
                    delete sendBuffer[idxProc];
                    delete recvBuffer[idxProc];
                }
                delete[] globalReceiveMap;
            }

            ///////////////////////////////////////////////////
            // Prepare data for thread P2P
            ///////////////////////////////////////////////////

#pragma omp single // MUST WAIT!
            {
                typename OctreeClass::Iterator octreeIterator(tree);
                octreeIterator.gotoBottomLeft();

                // to store which shape for each leaf
                typename OctreeClass::Iterator* const myLeafs = new typename OctreeClass::Iterator[this->numberOfLeafs];
                int*const shapeType = new int[this->numberOfLeafs];

                for(int idxLeaf = 0 ; idxLeaf < this->numberOfLeafs ; ++idxLeaf){
                    myLeafs[idxLeaf] = octreeIterator;

                    const FTreeCoordinate& coord = octreeIterator.getCurrentCell()->getCoordinate();
                    const int shape = P2PExclusionClass::GetShapeIdx(coord);
                    shapeType[idxLeaf] = shape;

                    ++shapeLeaf[shape];

                    octreeIterator.moveRight();
                }

                int startPosAtShape[SizeShape];
                startPosAtShape[0] = 0;
                for(int idxShape = 1 ; idxShape < SizeShape ; ++idxShape){
                    startPosAtShape[idxShape] = startPosAtShape[idxShape-1] + shapeLeaf[idxShape-1];
                }

                int idxInArray = 0;
                for(int idxLeaf = 0 ; idxLeaf < this->numberOfLeafs ; ++idxLeaf, ++idxInArray){
                    const int shapePosition = shapeType[idxInArray];

                    leafsDataArray[startPosAtShape[shapePosition]].coord = myLeafs[idxInArray].getCurrentGlobalCoordinate();
                    leafsDataArray[startPosAtShape[shapePosition]].cell = myLeafs[idxInArray].getCurrentCell();
                    leafsDataArray[startPosAtShape[shapePosition]].targets = myLeafs[idxInArray].getCurrentListTargets();
                    leafsDataArray[startPosAtShape[shapePosition]].sources = myLeafs[idxInArray].getCurrentListSrc();
                    if( leafsNeedOther.get(idxLeaf) ) leafsNeedOtherData.push(leafsDataArray[startPosAtShape[shapePosition]]);

                    ++startPosAtShape[shapePosition];
                }

                delete[] shapeType;
                delete[] myLeafs;

                FLOG(prepareCounter.tac());
            }


            //////////////////////////////////////////////////////////
            // Computation P2P that DO NOT need others data
            //////////////////////////////////////////////////////////

            {
#pragma omp single nowait
                {
                    FLOG(computationCounter.tic());
                    int previous = 0;

                    for(int idxShape = 0 ; idxShape < SizeShape ; ++idxShape){
                        const int endAtThisShape = shapeLeaf[idxShape] + previous;
                        const int chunckSize = userChunkSize;

                        for(int idxLeafs = previous ; idxLeafs < endAtThisShape ; idxLeafs += chunckSize){
                            const int nbLeavesInTask = FMath::Min(endAtThisShape-idxLeafs, chunckSize);
#pragma omp task default(none) firstprivate(nbLeavesInTask,idxLeafs) //+shared(leafsDataArray)
                            {
                                KernelClass* myThreadkernels = (kernels[omp_get_thread_num()]);
                                // There is a maximum of 26 neighbors
                                ContainerClass* neighbors[26];
                                int neighborPositions[26];

                                for(int idxTaskLeaf = idxLeafs ; idxTaskLeaf < (idxLeafs + nbLeavesInTask) ; ++idxTaskLeaf){
                                    LeafData& currentIter = leafsDataArray[idxTaskLeaf];
                                    if(l2pEnabled){
                                        myThreadkernels->L2P(currentIter.cell, currentIter.targets);
                                    }
                                    if(p2pEnabled){
                                        // need the current particles and neighbors particles
                                        const int counter = tree->getLeafsNeighbors(neighbors, neighborPositions, currentIter.coord, OctreeHeight-1);
                                        myThreadkernels->P2P( currentIter.coord,currentIter.targets,
                                                              currentIter.sources, neighbors, neighborPositions, counter);
                                    }
                                }
                            }
                        }
                        previous = endAtThisShape;

#pragma omp taskwait
                    }
                    FLOG(computationCounter.tac());
                }
            }

            // Wait the come to finish (and the previous computation also)
#pragma omp barrier


            //////////////////////////////////////////////////////////
            // Computation P2P that need others data
            //////////////////////////////////////////////////////////
#pragma omp master
            { FLOG( computation2Counter.tic() ); }

            if(p2pEnabled){
                KernelClass& myThreadkernels = (*kernels[omp_get_thread_num()]);
                // There is a maximum of 26 neighbors
                ContainerClass* neighbors[26];
                MortonIndex indexesNeighbors[26];
                int indexArray[26];
                int neighborPositions[26];
                // Box limite
                FAssertLF(leafsNeedOtherData.getSize() < std::numeric_limits<int>::max());
                const int nbLeafToProceed = int(leafsNeedOtherData.getSize());

#pragma omp for  schedule(dynamic, userChunkSize)
                for(int idxLeafs = 0 ; idxLeafs < nbLeafToProceed ; ++idxLeafs){
                    LeafData currentIter = leafsNeedOtherData[idxLeafs];
                    FAssertLF(currentIter.targets->getNbParticles())

                            // need the current particles and neighbors particles
                            int counter = 0;

                    // Take possible data
                    const int nbNeigh = currentIter.coord.getNeighborsIndexes(OctreeHeight, indexesNeighbors, indexArray);

                    for(int idxNeigh = 0 ; idxNeigh < nbNeigh ; ++idxNeigh){
                        if(indexesNeighbors[idxNeigh] < (intervals[idProcess].leftIndex) || (intervals[idProcess].rightIndex) < indexesNeighbors[idxNeigh]){
                            ContainerClass*const hypotheticNeighbor = otherP2Ptree.getLeafSrc(indexesNeighbors[idxNeigh]);
                            if(hypotheticNeighbor){
                                neighbors[ counter ] = hypotheticNeighbor;
                                neighborPositions[counter] = indexArray[idxNeigh];
                                ++counter;
                            }
                        }
                    }
                    if(counter){
                        myThreadkernels.P2PRemote( currentIter.cell->getCoordinate(), currentIter.targets,
                                                   currentIter.sources, neighbors, neighborPositions, counter);
                    }

                }

            }
        }

        delete[] leafsDataArray;

        FLOG(computation2Counter.tac());


        FLOG( FLog::Controller << "\tFinished (@Direct Pass (L2P + P2P) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation L2P + P2P : " << computationCounter.elapsed() << " s\n" );
        FLOG( FLog::Controller << "\t\t Computation P2P 2 : " << computation2Counter.elapsed() << " s\n" );
        FLOG( FLog::Controller << "\t\t Prepare P2P : " << prepareCounter.elapsed() << " s\n" );
        FLOG( FLog::Controller << "\t\t Gather P2P : " << gatherCounter.elapsed() << " s\n" );
        FLOG( FLog::Controller << "\t\t Wait : " << waitCounter.elapsed() << " s\n" );
        FLOG( FLog::Controller.flush());

    }
};






#endif //FFMMALGORITHMTHREAD_HPP
