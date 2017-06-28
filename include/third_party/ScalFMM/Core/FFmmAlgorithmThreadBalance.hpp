#ifndef FFmmAlgorithmThreadBalanceBALANCE_HPP
#define FFmmAlgorithmThreadBalanceBALANCE_HPP

#include "../Utils/FAssert.hpp"
#include "../Utils/FLog.hpp"

#include "../Utils/FTic.hpp"
#include "../Utils/FGlobal.hpp"
#include "Utils/FAlgorithmTimers.hpp"

#include "../Containers/FOctree.hpp"
#include "../Utils/FEnv.hpp"

#include "FCoreCommon.hpp"
#include "FP2PExclusion.hpp"

#include <omp.h>
#include <vector>
#include <memory>

/**
* \author Berenger Bramas (berenger.bramas@inria.fr)
* \brief Implements an FMM algorithm threaded using OpenMP.
*
* Please read the license
*
* This class runs a threaded FMM algorithm.
* It balance the execution between threads.
*
* When using this algorithm the P2P is thread safe.
*
* This class does not deallocate pointers given to its constructor.
*/
template<class OctreeClass, class CellClass, class ContainerClass, class KernelClass, class LeafClass, class P2PExclusionClass = FP2PMiddleExclusion>
class FFmmAlgorithmThreadBalance : public FAbstractAlgorithm, public FAlgorithmTimers{
    OctreeClass* const tree;                  ///< The octree to work on.
    KernelClass** kernels;                    ///< The kernels.

    static const int SizeShape = P2PExclusionClass::SizeShape;

    const int MaxThreads;                     ///< The maximum number of threads.

    const int OctreeHeight;                   ///< The height of the given tree.

    const int leafLevelSeparationCriteria;

public:
    /** Class constructor
     *
     * The constructor needs the octree and the kernels used for computation.
     * \param inTree the octree to work on.
     * \param inKernels the kernels to call.
     * \param inUserChunckSize To specify the chunck size in the loops (-1 is static, 0 is N/p^2, otherwise it
     * directly used as the number of item to proceed together), default is 10
     *
     * \except An exception is thrown if one of the arguments is NULL.
     */
    FFmmAlgorithmThreadBalance(OctreeClass* const inTree, KernelClass* const inKernels,
                               const int inLeafLevelSeperationCriteria = 1)
        : tree(inTree) , kernels(nullptr),
          MaxThreads(FEnv::GetValue("SCALFMM_ALGO_NUM_THREADS",omp_get_max_threads())), OctreeHeight(tree->getHeight()),
          leafLevelSeparationCriteria(inLeafLevelSeperationCriteria) {
        FAssertLF(tree, "tree cannot be null");
        FAssertLF(leafLevelSeparationCriteria < 3, "Separation criteria should be < 3");

        this->kernels = new KernelClass*[MaxThreads];
        #pragma omp parallel num_threads(MaxThreads)
        {
            #pragma omp critical (InitFFmmAlgorithmThreadBalance)
            {
                this->kernels[omp_get_thread_num()] = new KernelClass(*inKernels);
            }
        }

        FAbstractAlgorithm::setNbLevelsInTree(OctreeHeight);
        buildThreadIntervals();

        FLOG(FLog::Controller << "FFmmAlgorithmThreadBalance (Max Thread " << omp_get_max_threads() << ")\n");
    }

    /** Default destructor */
    virtual ~FFmmAlgorithmThreadBalance(){
        for(int idxThread = 0 ; idxThread < MaxThreads ; ++idxThread){
            delete this->kernels[idxThread];
        }
        delete [] this->kernels;
    }

    /**
      * Runs the complete algorithm.
      */
    void executeCore(const unsigned operationsToProceed) override {

        Timers[P2MTimer].tic();
        if(operationsToProceed & FFmmP2M) bottomPass();
        Timers[P2MTimer].tac();

        Timers[M2MTimer].tic();
        if(operationsToProceed & FFmmM2M) upwardPass();
        Timers[M2MTimer].tac();

        Timers[M2LTimer].tic();
        if(operationsToProceed & FFmmM2L) transferPass();
        Timers[M2LTimer].tac();

        Timers[L2LTimer].tic();
        if(operationsToProceed & FFmmL2L) downardPass();
        Timers[L2LTimer].tac();

        Timers[NearTimer].tic();
        if(operationsToProceed & FFmmL2P) L2P();
        if(operationsToProceed & FFmmP2P) directPass();
        Timers[NearTimer].tac();
    }

protected:
    /////////////////////////////////////////////////////////////////////////////
    // P2M
    /////////////////////////////////////////////////////////////////////////////
    /** The workload contains what a thread need to perfom its interval of work */
    struct Workload{
        typename OctreeClass::Iterator iterator;
        int nbElements;
    };

    //< The work per thread for the P2M
    std::vector<Workload> workloadP2M;
    //< The work per level and per thread for the M2M
    std::vector<std::vector<Workload>> workloadM2M;
    //< The work per level and per thread for the M2L
    std::vector<std::vector<Workload>> workloadM2L;
    //< The work per level and per thread for the L2L
    std::vector<std::vector<Workload>> workloadL2L;
    //< The work per thread for the L2P
    std::vector<Workload> workloadL2P;
    //< The work per shape and per thread for the P2P
    std::vector<std::vector<std::pair<int,int>>> workloadP2P;

    /** This structure is needed by the thread for the P2P because of the colors */
    struct LeafData{
        MortonIndex index;
        FTreeCoordinate coord;
        ContainerClass* targets;
        ContainerClass* sources;
    };
    /** Direct access to the data for the P2P */
    std::unique_ptr<LeafData[]> leafsDataArray;

    /** This struct is used during the preparation of the interval */
    struct WorkloadTemp{
        typename OctreeClass::Iterator iterator;
        FSize amountOfWork;
    };

    /** From a vector of work (workPerElement) generate the interval */
    void generateIntervalFromWorkload(std::vector<Workload>* intervals, const FSize totalWork,
                                      WorkloadTemp* workPerElement, const FSize nbElements) const {
        // Now split between thread
        (*intervals).resize(MaxThreads);

        // Ideally each thread will have this
        const FSize idealWork = (totalWork/MaxThreads);
        ///FLOG(FLog::Controller << "[Balance] idealWork " << idealWork << "\n");

        // Assign default value for first thread
        int idxThread = 0;
        (*intervals)[idxThread].iterator = workPerElement[0].iterator;
        (*intervals)[idxThread].nbElements = 1;
        FSize assignWork = workPerElement[0].amountOfWork;

        for(int idxElement = 1 ; idxElement < nbElements ; ++idxElement){
            ///FLOG(FLog::Controller << "[Balance] idxElement " << workPerElement[idxElement].amountOfWork << "\n");
            ///FLOG(FLog::Controller << "[Balance] assignWork " << assignWork << "\n");
            // is it more balance if we add the current element to the current thread
            if(FMath::Abs((idxThread+1)*idealWork - assignWork) <
                    FMath::Abs((idxThread+1)*idealWork - assignWork - workPerElement[idxElement].amountOfWork)
                    && idxThread != MaxThreads-1){
                ///FLOG(FLog::Controller << "[Balance] Shape Thread " << idxThread << " goes from "
                ///      << (*intervals)[idxThread].iterator.getCurrentGlobalIndex() << " nb " << (*intervals)[idxThread].nbElements << "/" << nbElements << "\n");
                // if not start filling the next thread
                idxThread += 1;
                (*intervals)[idxThread].iterator = workPerElement[idxElement].iterator;
                (*intervals)[idxThread].nbElements = 0;
            }
            (*intervals)[idxThread].nbElements += 1;
            assignWork += workPerElement[idxElement].amountOfWork;
        }

        ///FLOG(FLog::Controller << "[Balance] Shape Thread " << idxThread << " goes from "
        ///      << (*intervals)[idxThread].iterator.getCurrentGlobalIndex() << " nb " << (*intervals)[idxThread].nbElements << "/" << nbElements << "\n");
    }

    void buildThreadIntervals(){
        // Reset the vectors
        workloadP2M.clear();
        workloadM2M.clear();
        workloadM2L.clear();
        workloadL2L.clear();
        workloadL2P.clear();
        workloadP2P.clear();

        // Count the number of leaves and color elements
        int shapeLeaves[SizeShape] = {0};
        int leafsNumber = 0;
        {
            typename OctreeClass::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();
            do{
                ++leafsNumber;
                const FTreeCoordinate& coord = octreeIterator.getCurrentCell()->getCoordinate();
                ++shapeLeaves[P2PExclusionClass::GetShapeIdx(coord)];
            } while(octreeIterator.moveRight());
        }

        // Allocate the working buffer
        std::unique_ptr<WorkloadTemp*[]> workloadBufferThread(new WorkloadTemp*[MaxThreads]);
        memset(workloadBufferThread.get(), 0, MaxThreads*sizeof(WorkloadTemp*));

        #pragma omp parallel num_threads(MaxThreads)
        {
            #pragma omp single
            {
                #pragma omp task
                { // Prepare P2M
                    if(workloadBufferThread[omp_get_thread_num()] == nullptr){
                        workloadBufferThread[omp_get_thread_num()] = new WorkloadTemp[leafsNumber];
                    }
                    WorkloadTemp* workloadBuffer = workloadBufferThread[omp_get_thread_num()];

                    /// FLOG(FLog::Controller << "[Balance] P2M:\n");
                    typename OctreeClass::Iterator octreeIterator(tree);
                    octreeIterator.gotoBottomLeft();
                    FSize idxLeaf = 0;
                    FSize totalWork = 0;
                    do{
                        // Keep track of tree iterator
                        workloadBuffer[idxLeaf].iterator = octreeIterator;
                        // Count the nb of particles as amount of work in the leaf
                        workloadBuffer[idxLeaf].amountOfWork = octreeIterator.getCurrentListSrc()->getNbParticles();
                        // Keep the total amount of work
                        totalWork += workloadBuffer[idxLeaf].amountOfWork;
                        ++idxLeaf;
                    } while(octreeIterator.moveRight());

                    generateIntervalFromWorkload(&workloadP2M, totalWork, workloadBuffer, idxLeaf);
                }

                #pragma omp task
                { // Prepare L2P
                    if(workloadBufferThread[omp_get_thread_num()] == nullptr){
                        workloadBufferThread[omp_get_thread_num()] = new WorkloadTemp[leafsNumber];
                    }
                    WorkloadTemp* workloadBuffer = workloadBufferThread[omp_get_thread_num()];
                    /// FLOG(FLog::Controller << "[Balance] L2P:\n");
                    typename OctreeClass::Iterator octreeIterator(tree);
                    octreeIterator.gotoBottomLeft();
                    FSize idxLeaf = 0;
                    FSize totalWork = 0;
                    do{
                        // Keep track of tree iterator
                        workloadBuffer[idxLeaf].iterator = octreeIterator;
                        // Count the nb of particles as amount of work in the leaf
                        workloadBuffer[idxLeaf].amountOfWork = octreeIterator.getCurrentListTargets()->getNbParticles();
                        // Keep the total amount of work
                        totalWork += workloadBuffer[idxLeaf].amountOfWork;
                        ++idxLeaf;
                    } while(octreeIterator.moveRight());

                    generateIntervalFromWorkload(&workloadL2P, totalWork, workloadBuffer, idxLeaf);
                }

                #pragma omp task
                {// Do it for the M2L
                    if(workloadBufferThread[omp_get_thread_num()] == nullptr){
                        workloadBufferThread[omp_get_thread_num()] = new WorkloadTemp[leafsNumber];
                    }
                    WorkloadTemp* workloadBuffer = workloadBufferThread[omp_get_thread_num()];
                    /// FLOG(FLog::Controller << "[Balance] M2L:\n");
                    workloadM2L.resize(OctreeHeight);
                    typename OctreeClass::Iterator avoidGotoLeftIterator(tree);
                    avoidGotoLeftIterator.gotoBottomLeft();

                    const CellClass* neighbors[343];

                    for(int idxLevel = OctreeHeight-1 ; idxLevel >= 2 ; --idxLevel){
                        /// FLOG(FLog::Controller << "[Balance] \t level " << idxLevel << ":\n");
                        typename OctreeClass::Iterator octreeIterator(avoidGotoLeftIterator);
                        avoidGotoLeftIterator.moveUp();

                        FSize idxCell = 0;
                        FSize totalWork = 0;
                        do{
                            // Keep track of tree iterator
                            workloadBuffer[idxCell].iterator = octreeIterator;
                            // Count the nb of M2L for this cell
                            workloadBuffer[idxCell].amountOfWork = tree->getInteractionNeighbors(neighbors, octreeIterator.getCurrentGlobalCoordinate(), idxLevel, 1);
                            // Keep the total amount of work
                            totalWork += workloadBuffer[idxCell].amountOfWork;
                            ++idxCell;
                        } while(octreeIterator.moveRight());

                        // Now split between thread
                        generateIntervalFromWorkload(&workloadM2L[idxLevel], totalWork, workloadBuffer, idxCell);
                    }
                }
                #pragma omp task
                {// Do it for the M2M L2L
                    if(workloadBufferThread[omp_get_thread_num()] == nullptr){
                        workloadBufferThread[omp_get_thread_num()] = new WorkloadTemp[leafsNumber];
                    }
                    WorkloadTemp* workloadBuffer = workloadBufferThread[omp_get_thread_num()];
                    /// FLOG(FLog::Controller << "[Balance] M2M L2L:\n");
                    workloadM2M.resize(OctreeHeight);
                    workloadL2L.resize(OctreeHeight);
                    typename OctreeClass::Iterator avoidGotoLeftIterator(tree);
                    avoidGotoLeftIterator.gotoBottomLeft();
                    avoidGotoLeftIterator.moveUp();

                    for(int idxLevel = OctreeHeight-2 ; idxLevel >= 2 ; --idxLevel){
                        /// FLOG(FLog::Controller << "[Balance] \t level " << idxLevel << ":\n");
                        typename OctreeClass::Iterator octreeIterator(avoidGotoLeftIterator);
                        avoidGotoLeftIterator.moveUp();

                        FSize idxCell = 0;
                        FSize totalWork = 0;
                        do{
                            // Keep track of tree iterator
                            workloadBuffer[idxCell].iterator = octreeIterator;
                            // Count the nb of children of the current cell
                            workloadBuffer[idxCell].amountOfWork = 0;
                            CellClass** child = octreeIterator.getCurrentChild();
                            for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
                                if(child[idxChild]) workloadBuffer[idxCell].amountOfWork += 1;
                            }
                            // Keep the total amount of work
                            totalWork += workloadBuffer[idxCell].amountOfWork;
                            ++idxCell;
                        } while(octreeIterator.moveRight());

                        // Now split between thread
                        generateIntervalFromWorkload(&workloadM2M[idxLevel], totalWork, workloadBuffer, idxCell);
                        generateIntervalFromWorkload(&workloadL2L[idxLevel], totalWork, workloadBuffer, idxCell);
                    }
                }

                #pragma omp task
                {
                    if(workloadBufferThread[omp_get_thread_num()] == nullptr){
                        workloadBufferThread[omp_get_thread_num()] = new WorkloadTemp[leafsNumber];
                    }
                    WorkloadTemp* workloadBuffer = workloadBufferThread[omp_get_thread_num()];
                    memset(workloadBuffer, 0, sizeof(struct WorkloadTemp)*leafsNumber);
                    // Prepare the P2P
                    leafsDataArray.reset(new LeafData[leafsNumber]);

                    // We need the offset for each color
                    int startPosAtShape[SizeShape] = {0};
                    for(int idxShape = 1 ; idxShape < SizeShape ; ++idxShape){
                        startPosAtShape[idxShape] = startPosAtShape[idxShape-1] + shapeLeaves[idxShape-1];
                    }

                    // Prepare each color
                    typename OctreeClass::Iterator octreeIterator(tree);
                    octreeIterator.gotoBottomLeft();

                    FSize workPerShape[SizeShape] = {0};

                    // for each leafs
                    for(int idxLeaf = 0 ; idxLeaf < leafsNumber ; ++idxLeaf){
                        const FTreeCoordinate& coord = octreeIterator.getCurrentGlobalCoordinate();
                        const int shapePosition = P2PExclusionClass::GetShapeIdx(coord);

                        const int positionToWork = startPosAtShape[shapePosition]++;

                        leafsDataArray[positionToWork].index   = octreeIterator.getCurrentGlobalIndex();
                        leafsDataArray[positionToWork].coord   = coord;
                        leafsDataArray[positionToWork].targets = octreeIterator.getCurrentListTargets();
                        leafsDataArray[positionToWork].sources = octreeIterator.getCurrentListSrc();

                        // For now the cost is simply based on the number of particles
                        const FSize nbPartInLeaf = octreeIterator.getCurrentListTargets()->getNbParticles();
                        workloadBuffer[positionToWork].amountOfWork = nbPartInLeaf*nbPartInLeaf;
                        ContainerClass* neighbors[27];
                        tree->getLeafsNeighbors(neighbors, octreeIterator.getCurrentGlobalCoordinate(), OctreeHeight-1);
                        for(int idxNeigh = 0 ; idxNeigh < 27 ; ++idxNeigh){
                            if(neighbors[idxNeigh]){
                                workloadBuffer[positionToWork].amountOfWork +=
                                         nbPartInLeaf * neighbors[idxNeigh]->getNbParticles();
                            }
                        }

                        workPerShape[shapePosition] += workloadBuffer[positionToWork].amountOfWork;

                        octreeIterator.moveRight();
                    }

                    workloadP2P.resize(SizeShape);
                    int offsetShape = 0;

                    for(int idxShape = 0 ; idxShape < SizeShape ; ++idxShape){
                        std::vector<std::pair<int,int>>* intervals = &workloadP2P[idxShape];
                        const int nbElements = shapeLeaves[idxShape];
                        const FSize totalWork = workPerShape[idxShape];

                        // Now split between thread
                        (*intervals).resize(MaxThreads, std::pair<int,int>(0,0));
                        // Ideally each thread will have this
                        const FSize idealWork = (totalWork/MaxThreads);
                        // Assign default value for first thread
                        int idxThread = 0;
                        (*intervals)[idxThread].first = offsetShape;
                        FSize assignWork = workloadBuffer[offsetShape].amountOfWork;
                        for(int idxElement = 1+offsetShape ; idxElement < nbElements+offsetShape ; ++idxElement){
                            if(FMath::Abs((idxThread+1)*idealWork - assignWork) <
                                    FMath::Abs((idxThread+1)*idealWork - assignWork - workloadBuffer[idxElement].amountOfWork)
                                    && idxThread != MaxThreads-1){
                                (*intervals)[idxThread].second = idxElement;
                                idxThread += 1;
                                (*intervals)[idxThread].first = idxElement;
                            }
                            assignWork += workloadBuffer[idxElement].amountOfWork;
                        }
                        (*intervals)[idxThread].second = nbElements + offsetShape;

                        idxThread += 1;
                        while(idxThread != MaxThreads){
                            (*intervals)[idxThread].first = nbElements+offsetShape;
                            (*intervals)[idxThread].second = nbElements+offsetShape;
                            idxThread += 1;
                        }

                        offsetShape += nbElements;
                    }
                }
            }

            #pragma omp taskwait
        }

        for(int idxThread = 0 ; idxThread < MaxThreads ; ++idxThread){
            delete[] workloadBufferThread[idxThread];
        }
    }


    /////////////////////////////////////////////////////////////////////////////
    // P2M
    /////////////////////////////////////////////////////////////////////////////

    /** Runs the P2M kernel. */
    void bottomPass(){
        FLOG( FLog::Controller.write("\tStart Bottom Pass\n").write(FLog::Flush) );
        FLOG(FTic counterTime);

        FLOG(FTic computationCounter);
        #pragma omp parallel num_threads(MaxThreads)
        {
            KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
            const int nbCellsToCompute = workloadP2M[omp_get_thread_num()].nbElements;
            typename OctreeClass::Iterator octreeIterator(workloadP2M[omp_get_thread_num()].iterator);

            for(int idxLeafs = 0 ; idxLeafs < nbCellsToCompute ; ++idxLeafs){
                // We need the current cell that represent the leaf
                // and the list of particles
                myThreadkernels->P2M( octreeIterator.getCurrentCell() , octreeIterator.getCurrentListSrc());
                octreeIterator.moveRight();
            }

            FAssertLF(omp_get_thread_num() == MaxThreads-1
                      || workloadP2M[omp_get_thread_num()+1].nbElements == 0
                      || octreeIterator.getCurrentGlobalIndex() == workloadP2M[omp_get_thread_num()+1].iterator.getCurrentGlobalIndex());
        }
        FLOG(computationCounter.tac() );

        FLOG( FLog::Controller << "\tFinished (@Bottom Pass (P2M) = "  << counterTime.tacAndElapsed() << "s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.elapsed() << " s\n" );

    }

    /////////////////////////////////////////////////////////////////////////////
    // Upward
    /////////////////////////////////////////////////////////////////////////////

    /** Runs the M2M kernel. */
    void upwardPass(){
        FLOG( FLog::Controller.write("\tStart Upward Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);

        // for each levels
        for(int idxLevel = FMath::Min(OctreeHeight - 2, FAbstractAlgorithm::lowerWorkingLevel - 1) ; idxLevel >= FAbstractAlgorithm::upperWorkingLevel ; --idxLevel ){
            FLOG(FTic counterTimeLevel);

            FLOG(computationCounter.tic());
            #pragma omp parallel num_threads(MaxThreads)
            {
                KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
                const int nbCellsToCompute = workloadM2M[idxLevel][omp_get_thread_num()].nbElements;
                typename OctreeClass::Iterator octreeIterator( workloadM2M[idxLevel][omp_get_thread_num()].iterator);

                for(int idxCell = 0 ; idxCell < nbCellsToCompute ; ++idxCell){
                    // We need the current cell and the child
                    // child is an array (of 8 child) that may be null
                    myThreadkernels->M2M( octreeIterator.getCurrentCell() , octreeIterator.getCurrentChild(), idxLevel);
                    octreeIterator.moveRight();
                }

                FAssertLF(omp_get_thread_num() == MaxThreads-1
                          || workloadM2M[idxLevel][omp_get_thread_num()+1].nbElements == 0
                          || octreeIterator.getCurrentGlobalIndex() == workloadM2M[idxLevel][omp_get_thread_num()+1].iterator.getCurrentGlobalIndex());
            }

            FLOG(computationCounter.tac());
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << "s\n" );
        }


        FLOG( FLog::Controller << "\tFinished (@Upward Pass (M2M) = "  << counterTime.tacAndElapsed() << "s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.cumulated() << " s\n" );

    }

    /////////////////////////////////////////////////////////////////////////////
    // Transfer
    /////////////////////////////////////////////////////////////////////////////

    /** Runs the M2L kernel. */
  	/** M2L  */
  void transferPass(){
#ifdef SCALFMM_USE_EZTRACE	  
    eztrace_start();
#endif
    this->transferPassWithFinalize() ;
    //
#ifdef SCALFMM_USE_EZTRACE
    eztrace_stop();
#endif
  }

    void transferPassWithFinalize(){

        FLOG( FLog::Controller.write("\tStart Downward Pass (M2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);

        // for each levels
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel ){
            const int separationCriteria = (idxLevel != FAbstractAlgorithm::lowerWorkingLevel-1 ? 1 : leafLevelSeparationCriteria);
            FLOG(FTic counterTimeLevel);
            FLOG(computationCounter.tic());
            #pragma omp parallel num_threads(MaxThreads)
            {
                KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
                const int nbCellsToCompute = workloadM2L[idxLevel][omp_get_thread_num()].nbElements;
                typename OctreeClass::Iterator octreeIterator( workloadM2L[idxLevel][omp_get_thread_num()].iterator);

                const CellClass* neighbors[342];
                int neighborPositions[342];

                for(int idxCell = 0 ; idxCell < nbCellsToCompute ; ++idxCell){
                    const int counter = tree->getInteractionNeighbors(neighbors, neighborPositions, octreeIterator.getCurrentGlobalCoordinate(), idxLevel, separationCriteria);
                    if(counter) myThreadkernels->M2L( octreeIterator.getCurrentCell() , neighbors, neighborPositions, counter, idxLevel);
                    octreeIterator.moveRight();
                }

                myThreadkernels->finishedLevelM2L(idxLevel);


                FAssertLF(omp_get_thread_num() == MaxThreads-1
                          || workloadM2L[idxLevel][omp_get_thread_num()+1].nbElements == 0
                          || octreeIterator.getCurrentGlobalIndex() == workloadM2L[idxLevel][omp_get_thread_num()+1].iterator.getCurrentGlobalIndex());
            }
            FLOG(computationCounter.tac());
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << "s\n" );
        }

        FLOG( FLog::Controller << "\tFinished (@Downward Pass (M2L) = "  << counterTime.tacAndElapsed() << "s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.cumulated() << " s\n" );
    }

    /////////////////////////////////////////////////////////////////////////////
    // Downward
    /////////////////////////////////////////////////////////////////////////////

    /** Runs the L2L kernel. */
    void downardPass(){

        FLOG( FLog::Controller.write("\tStart Downward Pass (L2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);


        const int heightMinusOne = FAbstractAlgorithm::lowerWorkingLevel - 1;
        // for each levels excepted leaf level
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < heightMinusOne ; ++idxLevel ){
            FLOG(FTic counterTimeLevel);

            FLOG(computationCounter.tic());
            #pragma omp parallel num_threads(MaxThreads)
            {
                KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
                const int nbCellsToCompute = workloadL2L[idxLevel][omp_get_thread_num()].nbElements;
                typename OctreeClass::Iterator octreeIterator( workloadL2L[idxLevel][omp_get_thread_num()].iterator);

                for(int idxCell = 0 ; idxCell < nbCellsToCompute ; ++idxCell){
                    myThreadkernels->L2L( octreeIterator.getCurrentCell() , octreeIterator.getCurrentChild(), idxLevel);
                    octreeIterator.moveRight();
                }

                FAssertLF(omp_get_thread_num() == MaxThreads-1
                          || workloadL2L[idxLevel][omp_get_thread_num()+1].nbElements == 0
                          || octreeIterator.getCurrentGlobalIndex() == workloadL2L[idxLevel][omp_get_thread_num()+1].iterator.getCurrentGlobalIndex());
            }
            FLOG(computationCounter.tac());
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << "s\n" );
        }

        FLOG( FLog::Controller << "\tFinished (@Downward Pass (L2L) = "  << counterTime.tacAndElapsed() << "s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.cumulated() << " s\n" );
    }




    /////////////////////////////////////////////////////////////////////////////
    // Direct
    /////////////////////////////////////////////////////////////////////////////

    void L2P(){
        #pragma omp parallel num_threads(MaxThreads)
        {
            KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
            const int nbCellsToCompute = workloadL2P[omp_get_thread_num()].nbElements;
            typename OctreeClass::Iterator octreeIterator(workloadL2P[omp_get_thread_num()].iterator);

            for(int idxLeafs = 0 ; idxLeafs < nbCellsToCompute ; ++idxLeafs){
                // We need the current cell that represent the leaf
                // and the list of particles
                myThreadkernels->L2P( octreeIterator.getCurrentCell() , octreeIterator.getCurrentListTargets());
                octreeIterator.moveRight();
            }

            FAssertLF(omp_get_thread_num() == MaxThreads-1
                      || workloadL2P[omp_get_thread_num()+1].nbElements == 0
                      || octreeIterator.getCurrentGlobalIndex() == workloadL2P[omp_get_thread_num()+1].iterator.getCurrentGlobalIndex());
        }
    }

    /** Runs the P2P kernel.
      *
     * \param p2pEnabled Run the P2P kernel.
     * \param l2pEnabled Run the L2P kernel.
     */
    void directPass(){
        FLOG( FLog::Controller.write("\tStart Direct Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);
        FLOG(FTic computationCounterP2P);

        #pragma omp parallel num_threads(MaxThreads)
        {
            FLOG(if(!omp_get_thread_num()) computationCounter.tic());

            KernelClass& myThreadkernels = (*kernels[omp_get_thread_num()]);
            // There is a maximum of 26 neighbors
            ContainerClass* neighbors[26];
            int neighborPositions[26];

            for(int idxShape = 0 ; idxShape < SizeShape ; ++idxShape){
                const std::pair<int,int> interval = workloadP2P[idxShape][omp_get_thread_num()];
                for(int idxLeafs = interval.first ; idxLeafs < interval.second ; ++idxLeafs){
                    LeafData& currentIter = leafsDataArray[idxLeafs];
                    // need the current particles and neighbors particles
                    FLOG(if(!omp_get_thread_num()) computationCounterP2P.tic());
                    const int counter = tree->getLeafsNeighbors(neighbors, neighborPositions, currentIter.coord, OctreeHeight-1);
                    myThreadkernels.P2P(currentIter.coord, currentIter.targets,
                                        currentIter.sources, neighbors, neighborPositions, counter);
                    FLOG(if(!omp_get_thread_num()) computationCounterP2P.tac());
                }

                FAssertLF(omp_get_thread_num() == MaxThreads-1
                          || interval.second == workloadP2P[idxShape][omp_get_thread_num()+1].first,
                        omp_get_thread_num(), " ==> ", interval.second, " != ", workloadP2P[idxShape][omp_get_thread_num()+1].first);

                #pragma omp barrier
            }
        }

        FLOG(computationCounter.tac());

        FLOG( FLog::Controller << "\tFinished (@Direct Pass (L2P + P2P) = "  << counterTime.tacAndElapsed() << "s)\n" );
        FLOG( FLog::Controller << "\t\t Computation L2P + P2P : " << computationCounter.cumulated()    << " s\n" );
        FLOG( FLog::Controller << "\t\t Computation P2P :       " << computationCounterP2P.cumulated() << " s\n" );

    }

};

#endif
