// See LICENCE file at project root
#ifndef FFMMALGORITHMSECTIONTASK_HPP
#define FFMMALGORITHMSECTIONTASK_HPP


#include "../Utils/FGlobal.hpp"
#include "../Utils/FAssert.hpp"
#include "../Utils/FLog.hpp"
#include "Utils/FAlgorithmTimers.hpp"

#include "../Utils/FTic.hpp"
#include "../Utils/FEnv.hpp"

#include "../Containers/FOctree.hpp"
#include "../Containers/FVector.hpp"

#include "FCoreCommon.hpp"
#include "FP2PExclusion.hpp"

#include <omp.h>

/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * @class FFmmAlgorithmSectionTask
 * @brief Parallel FMM algorithm implementation using OpenMP sections.
 * Please read the license
 *
 *
 * This class implements the FMM algorithm. the far field and near field
 * computations are parallelized using OpenMP sections (the L2P operator is done
 * last)
 *
 * \warning Because of the parallelization in this class, the only timers that
 * are meaningfull after a call to #executeCore are the P2M timer, which times the
 * far (P2M,M2M,M2L,L2L) and near (P2P) field computation and the NearTimer (L2P).
 *
 * Upon destruction, this class does not deallocate pointers given to its constructor.
 */
template<class OctreeClass, class CellClass, class ContainerClass, class KernelClass, class LeafClass, class P2PExclusionClass = FP2PMiddleExclusion>
class FFmmAlgorithmSectionTask : public FAbstractAlgorithm, public FAlgorithmTimers {

    OctreeClass* const tree;  ///< The octree to work on
    KernelClass** kernels;    ///< The kernels

    const int MaxThreads;

    const int OctreeHeight;

    const int leafLevelSeparationCriteria;
public:
    /** \brief Class constructor
     *
     * \note An assert checks whether one of the arguments is null.
     * \param inTree the octree to work on.
     * \param inKernels the kernels used for computation.
     */
    FFmmAlgorithmSectionTask(OctreeClass* const inTree,
                             KernelClass* const inKernels,
                             const int inLeafLevelSeparationCriteria = 1) :
        tree(inTree) ,
        kernels(0),
        MaxThreads(FEnv::GetValue("SCALFMM_ALGO_NUM_THREADS",omp_get_max_threads())),
        OctreeHeight(tree->getHeight()),
        leafLevelSeparationCriteria(inLeafLevelSeparationCriteria) {

        FAssertLF(tree, "tree cannot be null");
        FAssertLF(inKernels, "kernels cannot be null");
        FAssertLF(leafLevelSeparationCriteria < 3, "Separation criteria should be < 3");

        this->kernels = new KernelClass*[MaxThreads];
        #pragma omp parallel num_threads(MaxThreads)
        {
            #pragma omp critical (InitFFmmAlgorithmSectionTask)
            {
                this->kernels[omp_get_thread_num()] = new KernelClass(*inKernels);
            }
        }

        FAbstractAlgorithm::setNbLevelsInTree(tree->getHeight());

        FLOG(FLog::Controller << "FFmmAlgorithmSectionTask (Max Thread " << omp_get_max_threads() << ")\n");
    }

    /** Default destructor */
    virtual ~FFmmAlgorithmSectionTask(){
        for(int idxThread = 0 ; idxThread < MaxThreads ; ++idxThread){
            delete this->kernels[idxThread];
        }
        delete [] this->kernels;
    }

protected:
    /**
      * To execute the fmm algorithm
      * Call this function to run the complete algorithm
      */
    void executeCore(const unsigned operationsToProceed) override {
        Timers[P2MTimer].tic();
        #pragma omp parallel num_threads(MaxThreads)
        {
            #pragma omp sections
            {
                #pragma omp section
                {
                    if(operationsToProceed & FFmmP2M) bottomPass();

                    if(operationsToProceed & FFmmM2M) upwardPass();

                    if(operationsToProceed & FFmmM2L) transferPass();

                    if(operationsToProceed & FFmmL2L) downardPass();

                }
                #pragma omp section
                {
                    if( operationsToProceed & FFmmP2P ) directPass();
                }
            }

            #pragma omp single
            {
                if(operationsToProceed & FFmmL2P) L2PPass();
            }
        }
        Timers[P2MTimer].tac();
    }

    /////////////////////////////////////////////////////////////////////////////
    // P2M
    /////////////////////////////////////////////////////////////////////////////

    /** P2M */
    void bottomPass(){
        FLOG( FLog::Controller.write("\tStart Bottom Pass\n").write(FLog::Flush) );
        FLOG(FTic counterTime);

        typename OctreeClass::Iterator octreeIterator(tree);

        // Iterate on leafs
        octreeIterator.gotoBottomLeft();
        do{
            // We need the current cell that represent the leaf
            // and the list of particles
            #pragma omp task firstprivate(octreeIterator)
            {
                kernels[omp_get_thread_num()]->P2M( octreeIterator.getCurrentCell() , octreeIterator.getCurrentListSrc());
            }
        } while(octreeIterator.moveRight());

        #pragma omp taskwait


        FLOG( FLog::Controller << "\tFinished (@Bottom Pass (P2M) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }

    /////////////////////////////////////////////////////////////////////////////
    // Upward
    /////////////////////////////////////////////////////////////////////////////

    /** M2M */
    void upwardPass(){
        FLOG( FLog::Controller.write("\tStart Upward Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);

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
            // for each cells
            do{
                // We need the current cell and the child
                // child is an array (of 8 child) that may be null
                #pragma omp task firstprivate(octreeIterator,idxLevel)
                {
                    kernels[omp_get_thread_num()]->M2M( octreeIterator.getCurrentCell() , octreeIterator.getCurrentChild(), idxLevel);
                }
            } while(octreeIterator.moveRight());

            avoidGotoLeftIterator.moveUp();
            octreeIterator = avoidGotoLeftIterator;// equal octreeIterator.moveUp(); octreeIterator.gotoLeft();

            #pragma omp taskwait
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }


        FLOG( FLog::Controller << "\tFinished (@Upward Pass (M2M) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }

    /////////////////////////////////////////////////////////////////////////////
    // Transfer
    /////////////////////////////////////////////////////////////////////////////

    /** M2L  */
 	/** M2L  */
  void transferPass(){
#ifdef SCALFMM_USE_EZTRACE

    eztrace_start();
#endif
    if(KernelClass::NeedFinishedM2LEvent()){
        this->transferPassWithFinalize() ;
    }
    else{
        this->transferPassWithoutFinalize() ;
    }
#ifdef SCALFMM_USE_EZTRACE
    eztrace_stop();
#endif
  }
   void transferPassWithFinalize(){
        FLOG( FLog::Controller.write("\tStart Downward Pass (M2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);

        const CellClass* neighbors[342];
        int neighborPositions[342];

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.moveDown();

        for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
            octreeIterator.moveDown();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);
        //
        // for each levels
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel ){
            FLOG(FTic counterTimeLevel);
            const int separationCriteria = (idxLevel != FAbstractAlgorithm::lowerWorkingLevel-1 ? 1 : leafLevelSeparationCriteria);
            // for each cells
            do{
                const int counter = tree->getInteractionNeighbors(neighbors, neighborPositions, octreeIterator.getCurrentGlobalCoordinate(), idxLevel, separationCriteria);
                if(counter){
                    #pragma omp task firstprivate(octreeIterator, neighbors, neighborPositions, counter,idxLevel)
                    {
                        kernels[omp_get_thread_num()]->M2L( octreeIterator.getCurrentCell() , neighbors, neighborPositions, counter, idxLevel);
                    }
                }

            } while(octreeIterator.moveRight());

            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;

            #pragma omp taskwait

            for( int idxThread = 0 ; idxThread < omp_get_num_threads() ; ++idxThread){
                #pragma omp task
                {
                    kernels[idxThread]->finishedLevelM2L(idxLevel);
                }
            }

            #pragma omp taskwait
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }

        FLOG( FLog::Controller << "\tFinished (@Downward Pass (M2L) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }

   void transferPassWithoutFinalize(){
        FLOG( FLog::Controller.write("\tStart Downward Pass (M2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);

        const CellClass* neighbors[342];
        int neighborPositions[342];

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.moveDown();

        for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
            octreeIterator.moveDown();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);
        //
        // for each levels
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel ){
            FLOG(FTic counterTimeLevel);
            const int separationCriteria = (idxLevel != FAbstractAlgorithm::lowerWorkingLevel-1 ? 1 : leafLevelSeparationCriteria);
            // for each cells
            do{
                const int counter = tree->getInteractionNeighbors(neighbors, neighborPositions, octreeIterator.getCurrentGlobalCoordinate(), idxLevel, separationCriteria);
                if(counter){
                    #pragma omp task firstprivate(octreeIterator, neighbors, neighborPositions, counter,idxLevel)
                    {
                        kernels[omp_get_thread_num()]->M2L( octreeIterator.getCurrentCell() , neighbors, neighborPositions, counter, idxLevel);
                    }
                }

            } while(octreeIterator.moveRight());

            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;

        }

        #pragma omp taskwait

        FLOG( FLog::Controller << "\tFinished (@Downward Pass (M2L) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }

    /////////////////////////////////////////////////////////////////////////////
    // Downward
    /////////////////////////////////////////////////////////////////////////////

    void downardPass(){ // second L2L
        FLOG( FLog::Controller.write("\tStart Downward Pass (L2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.moveDown();

        for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
            octreeIterator.moveDown();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);

        const int heightMinusOne = FAbstractAlgorithm::lowerWorkingLevel - 1;
        // for each levels exepted leaf level
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < heightMinusOne ; ++idxLevel ){
            FLOG(FTic counterTimeLevel);
            // for each cells
            do{
                #pragma omp task firstprivate(octreeIterator,idxLevel)
                {
                    kernels[omp_get_thread_num()]->L2L( octreeIterator.getCurrentCell() , octreeIterator.getCurrentChild(), idxLevel);
                }

            } while(octreeIterator.moveRight());

            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;

            #pragma omp taskwait
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }

        FLOG( FLog::Controller << "\tFinished (@Downward Pass (L2L) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }


    /////////////////////////////////////////////////////////////////////////////
    // Direct
    /////////////////////////////////////////////////////////////////////////////

    /** P2P */
    void directPass(){
        FLOG( FLog::Controller.write("\tStart Direct Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);

        const int heightMinusOne = OctreeHeight - 1;


        const int SizeShape = P2PExclusionClass::SizeShape;
        FVector<typename OctreeClass::Iterator> shapes[SizeShape];

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();

        // for each leafs
        // Coloring all the cells
        do{
            const FTreeCoordinate& coord = octreeIterator.getCurrentGlobalCoordinate();
            const int shapePosition = P2PExclusionClass::GetShapeIdx(coord);

            shapes[shapePosition].push(octreeIterator);

        } while(octreeIterator.moveRight());

        FLOG( computationCounter.tic() );

        for( int idxShape = 0 ; idxShape < SizeShape ; ++idxShape){
            const FSize nbLeaf = shapes[idxShape].getSize();
            for(int iterLeaf = 0 ; iterLeaf < nbLeaf ; ++iterLeaf ){
                typename OctreeClass::Iterator toWork = shapes[idxShape][iterLeaf];
                // ToDO increase the granularity of the task
                #pragma omp task   firstprivate(toWork)
                {
                    // There is a maximum of 26 neighbors
                    ContainerClass* neighbors[26];
                    int neighborPositions[26];
                    const int counter = tree->getLeafsNeighbors(neighbors, neighborPositions, toWork.getCurrentGlobalCoordinate(),heightMinusOne);
                    kernels[omp_get_thread_num()]->P2P(toWork.getCurrentGlobalCoordinate(), toWork.getCurrentListTargets(),
                                         toWork.getCurrentListSrc(), neighbors, neighborPositions, counter);
                }
            }

            #pragma omp taskwait
        }

        FLOG( computationCounter.tac() );


        FLOG( FLog::Controller << "\tFinished (@Direct Pass (P2P) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation P2P : " << computationCounter.cumulated() << " s\n" );
    }

    void L2PPass(){
        FLOG( FLog::Controller.write("\tStart L2P Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();

        // for each leafs
        do{
            #pragma omp task firstprivate(octreeIterator)
            {
                kernels[omp_get_thread_num()]->L2P(octreeIterator.getCurrentCell(), octreeIterator.getCurrentListTargets());
            }
        } while(octreeIterator.moveRight());

        #pragma omp taskwait

        FLOG( FLog::Controller << "\tFinished (@Direct Pass (L2P) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }

};


#endif //FFMMALGORITHMTASK_HPP
