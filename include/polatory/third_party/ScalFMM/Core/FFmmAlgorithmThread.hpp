// See LICENCE file at project root
#ifndef FFMMALGORITHMTHREAD_HPP
#define FFMMALGORITHMTHREAD_HPP


#include "../Utils/FAssert.hpp"
#include "../Utils/FLog.hpp"

#include "../Utils/FTic.hpp"
#include "../Utils/FGlobal.hpp"
#include "../Utils/FAlgorithmTimers.hpp"
#include "../Utils/FEnv.hpp"

#include "../Containers/FOctree.hpp"

#include "FCoreCommon.hpp"
#include "FP2PExclusion.hpp"

#include <omp.h>

/**
* \author Berenger Bramas (berenger.bramas@inria.fr)
* \brief Implements an FMM algorithm threaded using OpenMP.
*
* Please read the license
*
* This class runs a threaded FMM algorithm.  It just iterates on a tree and call
* the kernels with good arguments.  The inspector-executor model is used : the
* class iterates on the tree and builds an array and works in parallel on this
* array.
*
* When using this algorithm the P2P is thread safe.
*
* This class does not deallocate pointers given to its constructor.
*/
template<class OctreeClass, class CellClass, class ContainerClass, class KernelClass, class LeafClass, class P2PExclusionClass = FP2PMiddleExclusion>
class FFmmAlgorithmThread : public FAbstractAlgorithm, public FAlgorithmTimers{
    OctreeClass* const tree;                  ///< The octree to work on.
    KernelClass** kernels;                    ///< The kernels.

    typename OctreeClass::Iterator* iterArray;
    int leafsNumber;

    static const int SizeShape = P2PExclusionClass::SizeShape;
    int shapeLeaf[SizeShape];

    const int MaxThreads;                     ///< The maximum number of threads.

    const int OctreeHeight;                   ///< The height of the given tree.

    int userChunkSize;

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
    FFmmAlgorithmThread(OctreeClass* const inTree, KernelClass* const inKernels,
                        const int inUserChunkSize = 10, const int inLeafLevelSeperationCriteria = 1)
        : tree(inTree) , kernels(nullptr), iterArray(nullptr), leafsNumber(0),
          MaxThreads(FEnv::GetValue("SCALFMM_ALGO_NUM_THREADS",omp_get_max_threads())), OctreeHeight(tree->getHeight()),
          userChunkSize(inUserChunkSize), leafLevelSeparationCriteria(inLeafLevelSeperationCriteria) {
        FAssertLF(tree, "tree cannot be null");
        FAssertLF(leafLevelSeparationCriteria < 3, "Separation criteria should be < 3");
        FAssertLF(0 < userChunkSize, "Chunk size should be > 0");
        
        this->kernels = new KernelClass*[MaxThreads];
        #pragma omp parallel num_threads(MaxThreads)
        {
            #pragma omp critical (InitFFmmAlgorithmThread)
            {
                this->kernels[omp_get_thread_num()] = new KernelClass(*inKernels);
            }
        }
        
        FAbstractAlgorithm::setNbLevelsInTree(tree->getHeight());
        
        FLOG(FLog::Controller << "FFmmAlgorithmThread (Max Thread " << omp_get_max_threads() << ")\n");
        FLOG(FLog::Controller << "\t static schedule " << (userChunkSize == -1 ? "static" : (userChunkSize == 0 ? "N/p^2" : std::to_string(userChunkSize))) << ")\n");
    }
    
    /** Default destructor */
    virtual ~FFmmAlgorithmThread(){
        for(int idxThread = 0 ; idxThread < MaxThreads ; ++idxThread){
            delete this->kernels[idxThread];
        }
        delete [] this->kernels;
    }
    
    template <class NumType>
    NumType getChunkSize(const NumType inSize) const {
        if(userChunkSize <= -1){
            return FMath::Max(NumType(1) , NumType(double(inSize)/double(omp_get_max_threads())) );
        } else if(userChunkSize == 0){
            return FMath::Max(NumType(1) , inSize/NumType(omp_get_max_threads()*omp_get_max_threads()));
        } else {
            return userChunkSize;
        }
    }    
    
    template <class NumType>
    void setChunkSize(const NumType size) {
            userChunkSize = size;
    }
    
protected:
    /**
      * Runs the complete algorithm.
      */
    void executeCore(const unsigned operationsToProceed) override {

        for(int idxShape = 0 ; idxShape < SizeShape ; ++idxShape){
            this->shapeLeaf[idxShape] = 0;
        }

        // Count leaf
        leafsNumber = 0;
        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();
        do{
            ++leafsNumber;
            const FTreeCoordinate& coord = octreeIterator.getCurrentCell()->getCoordinate();
            ++this->shapeLeaf[P2PExclusionClass::GetShapeIdx(coord)];

        } while(octreeIterator.moveRight());
        iterArray = new typename OctreeClass::Iterator[leafsNumber];
        FAssertLF(iterArray, "iterArray bad alloc");

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
        if( (operationsToProceed & FFmmP2P) || (operationsToProceed & FFmmL2P) ) directPass((operationsToProceed & FFmmP2P),(operationsToProceed & FFmmL2P));
        Timers[NearTimer].tac();

        delete [] iterArray;
        iterArray = nullptr;
    }

    /////////////////////////////////////////////////////////////////////////////
    // P2M
    /////////////////////////////////////////////////////////////////////////////

    /** Runs the P2M kernel. */
    void bottomPass(){
        FLOG( FLog::Controller.write("\tStart Bottom Pass\n").write(FLog::Flush) );
        FLOG(FTic counterTime);

        typename OctreeClass::Iterator octreeIterator(tree);
        int leafs = 0;
        // Iterate on leafs
        octreeIterator.gotoBottomLeft();
        do{
            iterArray[leafs] = octreeIterator;
            ++leafs;
        } while(octreeIterator.moveRight());

        const int chunkSize = this->getChunkSize(leafs);

        FLOG(FTic computationCounter);
        #pragma omp parallel num_threads(MaxThreads)
        {
            KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
            #pragma omp for nowait schedule(dynamic, chunkSize)
            for(int idxLeafs = 0 ; idxLeafs < leafs ; ++idxLeafs){
                // We need the current cell that represent the leaf
                // and the list of particles
                myThreadkernels->P2M( iterArray[idxLeafs].getCurrentCell() , iterArray[idxLeafs].getCurrentListSrc());
            }
        }
        FLOG(computationCounter.tac() );

        FLOG( FLog::Controller << "\tFinished (@Bottom Pass (P2M) = "  << counterTime.tacAndElapsed() << " s)\n" );
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

        // Start from leal level - 1
        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();
        octreeIterator.moveUp();

        for(int idxLevel = OctreeHeight - 2 ; idxLevel > FAbstractAlgorithm::lowerWorkingLevel-1 ; --idxLevel){
            octreeIterator.moveUp();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);

        // for each levels
        for(int idxLevel = FMath::Min(OctreeHeight - 2, FAbstractAlgorithm::lowerWorkingLevel - 1) ; idxLevel >= FAbstractAlgorithm::upperWorkingLevel ; --idxLevel ){
            FLOG(FTic counterTimeLevel);
            int numberOfCells = 0;
            // for each cells
            do{
                iterArray[numberOfCells] = octreeIterator;
                ++numberOfCells;
            } while(octreeIterator.moveRight());
            avoidGotoLeftIterator.moveUp();
            octreeIterator = avoidGotoLeftIterator;// equal octreeIterator.moveUp(); octreeIterator.gotoLeft();

            const int chunkSize = this->getChunkSize(numberOfCells);

            FLOG(computationCounter.tic());
            #pragma omp parallel num_threads(MaxThreads)
            {
                KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
                #pragma omp for nowait  schedule(dynamic, chunkSize)
                for(int idxCell = 0 ; idxCell < numberOfCells ; ++idxCell){
                    // We need the current cell and the child
                    // child is an array (of 8 child) that may be null
                    myThreadkernels->M2M( iterArray[idxCell].getCurrentCell() , iterArray[idxCell].getCurrentChild(), idxLevel);
                }
            }

            FLOG(computationCounter.tac());
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }


        FLOG( FLog::Controller << "\tFinished (@Upward Pass (M2M) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.cumulated() << " s\n" );

    }

    /////////////////////////////////////////////////////////////////////////////
    // Transfer
    /////////////////////////////////////////////////////////////////////////////
	/** M2L  */
  void transferPass(){
#ifdef SCALFMM_USE_EZTRACE
    eztrace_start();
#endif
    this->transferPassWithFinalize() ;
#ifdef SCALFMM_USE_EZTRACE
    eztrace_stop();
#endif
  }

    /** Runs the M2L kernel. */
    void transferPassWithFinalize(){

        FLOG( FLog::Controller.write("\tStart Downward Pass (M2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.moveDown();

        for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
            octreeIterator.moveDown();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);

        // for each levels
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel ){
            FLOG(FTic counterTimeLevel);
            const int separationCriteria = (idxLevel != FAbstractAlgorithm::lowerWorkingLevel-1 ? 1 : leafLevelSeparationCriteria);
            int numberOfCells = 0;
            // for each cells
            do{
                iterArray[numberOfCells] = octreeIterator;
                ++numberOfCells;
            } while(octreeIterator.moveRight());
            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;

            const int chunkSize = this->getChunkSize(numberOfCells);

            FLOG(computationCounter.tic());
            #pragma omp parallel num_threads(MaxThreads)
            {
                KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
                const CellClass* neighbors[342];
                int neighborPositions[342];

                #pragma omp for  schedule(dynamic, chunkSize) nowait
                for(int idxCell = 0 ; idxCell < numberOfCells ; ++idxCell){
                    const int counter = tree->getInteractionNeighbors(neighbors, neighborPositions, iterArray[idxCell].getCurrentGlobalCoordinate(), idxLevel, separationCriteria);
                    if(counter) myThreadkernels->M2L( iterArray[idxCell].getCurrentCell() , neighbors, neighborPositions, counter, idxLevel);
                }

                myThreadkernels->finishedLevelM2L(idxLevel);
            }  //Synchro end of parallel section
            FLOG(computationCounter.tac());
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }

        FLOG( FLog::Controller << "\tFinished (@Downward Pass (M2L) = "  << counterTime.tacAndElapsed() << " s)\n" );
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

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.moveDown();

        for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
            octreeIterator.moveDown();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);

        const int heightMinusOne = FAbstractAlgorithm::lowerWorkingLevel - 1;
        // for each levels excepted leaf level
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < heightMinusOne ; ++idxLevel ){
            FLOG(FTic counterTimeLevel);
            int numberOfCells = 0;
            // for each cells
            do{
                iterArray[numberOfCells] = octreeIterator;
                ++numberOfCells;
            } while(octreeIterator.moveRight());
            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;

            FLOG(computationCounter.tic());
            const int chunkSize = this->getChunkSize(numberOfCells);
            #pragma omp parallel num_threads(MaxThreads)
            {
                KernelClass * const myThreadkernels = kernels[omp_get_thread_num()];
                #pragma omp for nowait schedule(dynamic, chunkSize)
                for(int idxCell = 0 ; idxCell < numberOfCells ; ++idxCell){
                    myThreadkernels->L2L( iterArray[idxCell].getCurrentCell() , iterArray[idxCell].getCurrentChild(), idxLevel);
                }
            }
            FLOG(computationCounter.tac());
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }

        FLOG( FLog::Controller << "\tFinished (@Downward Pass (L2L) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.cumulated() << " s\n" );
    }



    /////////////////////////////////////////////////////////////////////////////
    // Direct
    /////////////////////////////////////////////////////////////////////////////

    /** Runs the P2P & L2P kernels.
      *
     * \param p2pEnabled Run the P2P kernel.
     * \param l2pEnabled Run the L2P kernel.
     */
    void directPass(const bool p2pEnabled, const bool l2pEnabled){
        FLOG( FLog::Controller.write("\tStart Direct Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);
        FLOG(FTic computationCounterP2P);

        omp_lock_t lockShape[SizeShape];
        for(int idxShape = 0 ; idxShape < SizeShape ; ++idxShape){
            omp_init_lock(&lockShape[idxShape]);
        }

        struct LeafData{
            MortonIndex index;
            CellClass* cell;
            ContainerClass* targets;
            ContainerClass* sources;
        };
        LeafData* const leafsDataArray = new LeafData[this->leafsNumber];

        int startPosAtShape[SizeShape];
        startPosAtShape[0] = 0;
        for(int idxShape = 1 ; idxShape < SizeShape ; ++idxShape){
            startPosAtShape[idxShape] = startPosAtShape[idxShape-1] + this->shapeLeaf[idxShape-1];
        }

        #pragma omp parallel num_threads(MaxThreads)
        {

            const float step = float(this->leafsNumber) / float(omp_get_num_threads());
            const int start = int(FMath::Ceil(step * float(omp_get_thread_num())));
            const int tempEnd = int(FMath::Ceil(step * float(omp_get_thread_num()+1)));
            const int end = (tempEnd > this->leafsNumber ? this->leafsNumber : tempEnd);

            typename OctreeClass::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();

            for(int idxPreLeaf = 0 ; idxPreLeaf < start ; ++idxPreLeaf){
                octreeIterator.moveRight();
            }

            // for each leafs
            for(int idxMyLeafs = start ; idxMyLeafs < end ; ++idxMyLeafs){
                //iterArray[leafs] = octreeIterator;
                //++leafs;
                const FTreeCoordinate& coord = octreeIterator.getCurrentGlobalCoordinate();
                const int shapePosition = P2PExclusionClass::GetShapeIdx(coord);

                omp_set_lock(&lockShape[shapePosition]);
                const int positionToWork = startPosAtShape[shapePosition]++;
                omp_unset_lock(&lockShape[shapePosition]);

                leafsDataArray[positionToWork].index   = octreeIterator.getCurrentGlobalIndex();
                leafsDataArray[positionToWork].cell    = octreeIterator.getCurrentCell();
                leafsDataArray[positionToWork].targets = octreeIterator.getCurrentListTargets();
                leafsDataArray[positionToWork].sources = octreeIterator.getCurrentListSrc();

                octreeIterator.moveRight();
            }

            #pragma omp barrier

            FLOG(if(!omp_get_thread_num()) computationCounter.tic());

            KernelClass& myThreadkernels = (*kernels[omp_get_thread_num()]);
            // There is a maximum of 26 neighbors
            ContainerClass* neighbors[26];
            int neighborPositions[26];
            int previous = 0;

            for(int idxShape = 0 ; idxShape < SizeShape ; ++idxShape){
                const int endAtThisShape = this->shapeLeaf[idxShape] + previous;
                const int chunkSize = this->getChunkSize(endAtThisShape-previous);
                #pragma omp for schedule(dynamic, chunkSize)
                for(int idxLeafs = previous ; idxLeafs < endAtThisShape ; ++idxLeafs){
                    LeafData& currentIter = leafsDataArray[idxLeafs];
                    if(l2pEnabled){
                        myThreadkernels.L2P(currentIter.cell, currentIter.targets);
                    }
                    if(p2pEnabled){
                        // need the current particles and neighbors particles
                        FLOG(if(!omp_get_thread_num()) computationCounterP2P.tic());
                        const int counter = tree->getLeafsNeighbors(neighbors, neighborPositions, currentIter.cell->getCoordinate(), OctreeHeight-1);
                        myThreadkernels.P2P(currentIter.cell->getCoordinate(), currentIter.targets,
                                            currentIter.sources, neighbors, neighborPositions, counter);
                        FLOG(if(!omp_get_thread_num()) computationCounterP2P.tac());
                    }
                }

                previous = endAtThisShape;
            }
        }

        FLOG(computationCounter.tac());

        delete [] leafsDataArray;
        for(int idxShape = 0 ; idxShape < SizeShape ; ++idxShape){
            omp_destroy_lock(&lockShape[idxShape]);
        }


        FLOG( FLog::Controller << "\tFinished (@Direct Pass (L2P + P2P) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation L2P + P2P : " << computationCounter.cumulated()    << " s\n" );
        FLOG( FLog::Controller << "\t\t Computation P2P :       " << computationCounterP2P.cumulated() << " s\n" );

    }

};


#endif //FFMMALGORITHMTHREAD_HPP
