// See LICENCE file at project root
#ifndef FFMMALGORITHM_HPP
#define FFMMALGORITHM_HPP


#include "../Utils/FGlobal.hpp"
#include "../Utils/FAssert.hpp"
#include "../Utils/FLog.hpp"

#include "../Utils/FTic.hpp"

#include "../Containers/FOctree.hpp"
#include "../Containers/FVector.hpp"
#include "../Utils/FAlgorithmTimers.hpp"

#include "FCoreCommon.hpp"

/**
* \author Berenger Bramas (berenger.bramas@inria.fr)
* \brief Implements a basic FMM algorithm.
*
* Please read the license.
*
* This class runs the FMM algorithm on a tree using the kernels that it was given.
*
* This class does not deallocate pointers given to it constructor.
*/
template<class OctreeClass, class CellClass, class ContainerClass, class KernelClass, class LeafClass>
class FFmmAlgorithm :  public FAbstractAlgorithm, public FAlgorithmTimers {

    OctreeClass* const tree;       ///< The octree to work on.
    KernelClass* const kernels;    ///< The kernels.

    const int OctreeHeight;        ///< The height of the given tree.

    const int leafLevelSeparationCriteria;
public:
    /** Class constructor
     * 
     * The constructor needs the octree and the kernels used for computation.
     * @param inTree the octree to work on.
     * @param inKernels the kernels to call.
     *
     * \except An exception is thrown if one of the arguments is NULL.
     */
    FFmmAlgorithm(OctreeClass* const inTree, KernelClass* const inKernels, const int inLeafLevelSeparationCriteria = 1)
        : tree(inTree) , kernels(inKernels), OctreeHeight(tree->getHeight()), leafLevelSeparationCriteria(inLeafLevelSeparationCriteria) {

        FAssertLF(tree, "tree cannot be null");
        FAssertLF(kernels, "kernels cannot be null");
        FAssertLF(leafLevelSeparationCriteria < 3, "Separation criteria should be < 3");

        FAbstractAlgorithm::setNbLevelsInTree(tree->getHeight());

        FLOG(FLog::Controller << "FFmmAlgorithm\n");
    }

    /** Default destructor */
    virtual ~FFmmAlgorithm(){
    }

protected:
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
        if( (operationsToProceed & FFmmP2P) || (operationsToProceed & FFmmL2P) ) directPass((operationsToProceed & FFmmP2P),(operationsToProceed & FFmmL2P));
        Timers[NearTimer].tac();
    }

    /////////////////////////////////////////////////////////////////////////////
    // P2M
    /////////////////////////////////////////////////////////////////////////////

    /** Runs the P2M kernel. */
    void bottomPass(){
        FLOG( FLog::Controller.write("\tStart Bottom Pass\n").write(FLog::Flush) );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);

        typename OctreeClass::Iterator octreeIterator(tree);

        // Iterate on leafs
        octreeIterator.gotoBottomLeft();
        do{
            // We need the current cell that represents the leaf
            // and the list of particles
            FLOG(computationCounter.tic());
            kernels->P2M( octreeIterator.getCurrentCell() , octreeIterator.getCurrentListSrc());
            FLOG(computationCounter.tac());
        } while(octreeIterator.moveRight());

        FLOG( FLog::Controller << "\tFinished (@Bottom Pass (P2M) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.cumulated() << " s\n" );
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

            // for each cells
            do{
                // We need the current cell and the child
                // child is an array (of 8 child) that may be null
                FLOG(computationCounter.tic());
                kernels->M2M( octreeIterator.getCurrentCell() , octreeIterator.getCurrentChild(), idxLevel);
                FLOG(computationCounter.tac());
            } while(octreeIterator.moveRight());

            avoidGotoLeftIterator.moveUp();
            octreeIterator = avoidGotoLeftIterator;

            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }


        FLOG( FLog::Controller << "\tFinished (@Upward Pass (M2M) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.cumulated() << " s\n" );
    }

    /////////////////////////////////////////////////////////////////////////////
    // Transfer
    /////////////////////////////////////////////////////////////////////////////

    /** Runs the M2L kernel. */
    void transferPass(){
        FLOG( FLog::Controller.write("\tStart Downward Pass (M2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter);

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.moveDown();

        for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
            octreeIterator.moveDown();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);

        const CellClass* neighbors[342];
        int neighborPositions[342];

        // for each levels
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel ){
            FLOG(FTic counterTimeLevel);

            const int separationCriteria = (idxLevel != FAbstractAlgorithm::lowerWorkingLevel-1 ? 1 : leafLevelSeparationCriteria);

            // for each cells
            do{
                const int counter = tree->getInteractionNeighbors(neighbors, neighborPositions, octreeIterator.getCurrentGlobalCoordinate(), idxLevel, separationCriteria);
                FLOG(computationCounter.tic());
                if(counter) kernels->M2L( octreeIterator.getCurrentCell() , neighbors, neighborPositions, counter, idxLevel);
                FLOG(computationCounter.tac());
            } while(octreeIterator.moveRight());

            FLOG(computationCounter.tic());
            kernels->finishedLevelM2L(idxLevel);
            FLOG(computationCounter.tac());

            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;

            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }
        FLOG( FLog::Controller << "\tFinished (@Downward Pass (M2L) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << computationCounter.cumulated() << " s\n" );
    }

    /////////////////////////////////////////////////////////////////////////////
    // Downward
    /////////////////////////////////////////////////////////////////////////////

    /** Runs the L2L kernel .*/
    void downardPass(){
        FLOG( FLog::Controller.write("\tStart Downward Pass (L2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounter );

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
                FLOG(computationCounter.tic());
                kernels->L2L( octreeIterator.getCurrentCell() , octreeIterator.getCurrentChild(), idxLevel);
                FLOG(computationCounter.tac());
            } while(octreeIterator.moveRight());

            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;

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
     * \param p2pEnabled If true, run the P2P kernel.
     * \param l2pEnabled If true, run the L2P kernel.
     */
    void directPass(const bool p2pEnabled, const bool l2pEnabled){
        FLOG( FLog::Controller.write("\tStart Direct Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        FLOG(FTic computationCounterL2P);
        FLOG(FTic computationCounterP2P);

        const int heightMinusOne = OctreeHeight - 1;

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();
        // There is a maximum of 26 neighbors
        ContainerClass* neighbors[26];
        int neighborPositions[26];
        // for each leafs
        do{
            if(l2pEnabled){
                FLOG(computationCounterL2P.tic());
                kernels->L2P(octreeIterator.getCurrentCell(), octreeIterator.getCurrentListTargets());
                FLOG(computationCounterL2P.tac());
            }
            if(p2pEnabled){
                // need the current particles and neighbors particles
                const int counter = tree->getLeafsNeighbors(neighbors, neighborPositions, octreeIterator.getCurrentGlobalCoordinate(),heightMinusOne);
                FLOG(computationCounterP2P.tic());
                kernels->P2P(octreeIterator.getCurrentGlobalCoordinate(),octreeIterator.getCurrentListTargets(),
                             octreeIterator.getCurrentListSrc(), neighbors, neighborPositions, counter);
                FLOG(computationCounterP2P.tac());
            }
        } while(octreeIterator.moveRight());


        FLOG( FLog::Controller << "\tFinished (@Direct Pass (L2P + P2P) = "  << counterTime.tacAndElapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation L2P : " << computationCounterL2P.cumulated() << " s\n" );
        FLOG( FLog::Controller << "\t\t Computation P2P : " << computationCounterP2P.cumulated() << " s\n" );

    }

};


#endif //FFMMALGORITHM_HPP
