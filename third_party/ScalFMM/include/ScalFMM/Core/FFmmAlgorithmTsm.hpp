// See LICENCE file at project root
#ifndef FFMMALGORITHMTSM_HPP
#define FFMMALGORITHMTSM_HPP


#include "../Utils/FAssert.hpp"
#include "../Utils/FLog.hpp"

#include "../Utils/FTic.hpp"

#include "../Containers/FOctree.hpp"
#include "FCoreCommon.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FFmmAlgorithmTsm
* @brief
* Please read the license
*
* This class is a basic FMM algorithm
* It just iterates on a tree and call the kernels with good arguments.
*
* Of course this class does not deallocate pointer given in arguements.
*
* The differences with FmmAlgorithm is that it used target source model.
*/
template<class OctreeClass, class CellClass, class ContainerClass, class KernelClass, class LeafClass>
class FFmmAlgorithmTsm : public FAbstractAlgorithm{

    OctreeClass* const tree;                                                     //< The octree to work on
    KernelClass* const kernels;    //< The kernels

    const int OctreeHeight;

    const int leafLevelSeparationCriteria;

    FLOG(FTic counterTime);                                               //< In case of debug: to count the elapsed time
    FLOG(FTic computationCounter);                                        //< In case of debug: to  count computation time

public:
    /** The constructor need the octree and the kernels used for computation
      * @param inTree the octree to work on
      * @param inKernels the kernels to call
      * An assert is launched if one of the arguments is null
      */
    FFmmAlgorithmTsm(OctreeClass* const inTree, KernelClass* const inKernels, const int inLeafLevelSeperationCriteria = 1)
        : tree(inTree) , kernels(inKernels) , OctreeHeight(tree->getHeight()), leafLevelSeparationCriteria(inLeafLevelSeperationCriteria){

        FAssertLF(tree, "tree cannot be null");
        FAssertLF(kernels, "kernels cannot be null");
        FAssertLF(leafLevelSeparationCriteria < 3, "Separation criteria should be < 3");

        FAbstractAlgorithm::setNbLevelsInTree(tree->getHeight());

        FLOG(FLog::Controller << "FFmmAlgorithmTsm\n");
    }

    /** Default destructor */
    virtual ~FFmmAlgorithmTsm(){
    }

protected:
    /**
      * To execute the fmm algorithm
      * Call this function to run the complete algorithm
      */
    void executeCore(const unsigned operationsToProceed) override {

        if(operationsToProceed & FFmmP2M) bottomPass();

        if(operationsToProceed & FFmmM2M) upwardPass();

        if(operationsToProceed & FFmmM2L) transferPass();

        if(operationsToProceed & FFmmL2L) downardPass();

        if((operationsToProceed & FFmmP2P) || (operationsToProceed & FFmmL2P)) directPass((operationsToProceed & FFmmP2P),(operationsToProceed & FFmmL2P));
    }

    /** P2M */
    void bottomPass(){
        FLOG( FLog::Controller.write("\tStart Bottom Pass\n").write(FLog::Flush) );
        FLOG( counterTime.tic() );
        FLOG( double totalComputation = 0 );

        typename OctreeClass::Iterator octreeIterator(tree);

        // Iterate on leafs
        octreeIterator.gotoBottomLeft();
        do{
            // We need the current cell that represent the leaf
            // and the list of particles
            FLOG(computationCounter.tic());
            ContainerClass* const sources = octreeIterator.getCurrentListSrc();
            if(sources->getNbParticles()){
                octreeIterator.getCurrentCell()->setSrcChildTrue();
                kernels->P2M( octreeIterator.getCurrentCell() , sources);
            }
            if(octreeIterator.getCurrentListTargets()->getNbParticles()){
                octreeIterator.getCurrentCell()->setTargetsChildTrue();
            }
            FLOG(computationCounter.tac());
            FLOG(totalComputation += computationCounter.elapsed());
        } while(octreeIterator.moveRight());

        FLOG( counterTime.tac() );
        FLOG( FLog::Controller << "\tFinished (@Bottom Pass (P2M) = "  << counterTime.elapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << totalComputation << " s\n" );

    }

    /** M2M */
    void upwardPass(){
        FLOG( FLog::Controller.write("\tStart Upward Pass\n").write(FLog::Flush); );
        FLOG( counterTime.tic() );
        FLOG( double totalComputation = 0 );

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

                CellClass* potentialChild[8];
                CellClass** const realChild = octreeIterator.getCurrentChild();
                CellClass* const currentCell = octreeIterator.getCurrentCell();
                for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
                    potentialChild[idxChild] = nullptr;
                    if(realChild[idxChild]){
                        if(realChild[idxChild]->hasSrcChild()){
                            currentCell->setSrcChildTrue();
                            potentialChild[idxChild] = realChild[idxChild];
                        }
                        if(realChild[idxChild]->hasTargetsChild()){
                            currentCell->setTargetsChildTrue();
                        }
                    }
                }
                kernels->M2M( currentCell , potentialChild, idxLevel);

                FLOG(computationCounter.tac());
                FLOG(totalComputation += computationCounter.elapsed());
            } while(octreeIterator.moveRight());

            avoidGotoLeftIterator.moveUp();
            octreeIterator = avoidGotoLeftIterator;// equal octreeIterator.moveUp(); octreeIterator.gotoLeft();
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }

        FLOG( counterTime.tac() );
        FLOG( FLog::Controller << "\tFinished (@Upward Pass (M2M) = "  << counterTime.elapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << totalComputation << " s\n" );

    }

    /** M2L */
    void transferPass(){
        FLOG( FLog::Controller.write("\tStart Downward Pass (M2L)\n").write(FLog::Flush); );
        FLOG( counterTime.tic() );
        FLOG( double totalComputation = 0 );

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
                FLOG(computationCounter.tic());
                CellClass* const currentCell = octreeIterator.getCurrentCell();

                if(currentCell->hasTargetsChild()){
                    const int counter = tree->getInteractionNeighbors(neighbors, neighborPositions, octreeIterator.getCurrentGlobalCoordinate(),idxLevel, separationCriteria);
                    if( counter ){
                        int counterWithSrc = 0;
                        for(int idxRealNeighbors = 0 ; idxRealNeighbors < counter ; ++idxRealNeighbors ){
                            if(neighbors[idxRealNeighbors]->hasSrcChild()){
                                neighbors[counterWithSrc] = neighbors[idxRealNeighbors];
                                neighborPositions[counterWithSrc] = neighborPositions[idxRealNeighbors];
                                ++counterWithSrc;
                            }
                        }
                        if(counterWithSrc){
                            kernels->M2L( currentCell , neighbors, neighborPositions, counterWithSrc, idxLevel);
                        }
                    }
                }
                FLOG(computationCounter.tac());
                FLOG(totalComputation += computationCounter.elapsed());
            } while(octreeIterator.moveRight());

            FLOG(computationCounter.tic());
            kernels->finishedLevelM2L(idxLevel);
            FLOG(computationCounter.tac());

            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }

        FLOG( counterTime.tac() );
        FLOG( FLog::Controller << "\tFinished (@Downward Pass (M2L) = "  << counterTime.elapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << totalComputation << " s\n" );
    }

    /** L2L */
    void downardPass(){
        FLOG( FLog::Controller.write("\tStart Downward Pass (L2L)\n").write(FLog::Flush); );
        FLOG( counterTime.tic() );
        FLOG( double totalComputation = 0 );

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
                if( octreeIterator.getCurrentCell()->hasTargetsChild() ){
                    FLOG(computationCounter.tic());
                    CellClass* potentialChild[8];
                    CellClass** const realChild = octreeIterator.getCurrentChild();
                    CellClass* const currentCell = octreeIterator.getCurrentCell();
                    for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
                        if(realChild[idxChild] && realChild[idxChild]->hasTargetsChild()){
                            potentialChild[idxChild] = realChild[idxChild];
                        }
                        else{
                            potentialChild[idxChild] = nullptr;
                        }
                    }
                    kernels->L2L( currentCell , potentialChild, idxLevel);
                    FLOG(computationCounter.tac());
                    FLOG(totalComputation += computationCounter.elapsed());
                }
            } while(octreeIterator.moveRight());

            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;
            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }

        FLOG( counterTime.tac() );
        FLOG( FLog::Controller << "\tFinished (@Downward Pass (L2L) = "  << counterTime.elapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation : " << totalComputation << " s\n" );
    }



    /** P2P */
    void directPass(const bool p2pEnabled, const bool l2pEnabled){
        FLOG( FLog::Controller.write("\tStart Direct Pass\n").write(FLog::Flush); );
        FLOG( counterTime.tic() );
        FLOG( double totalComputation = 0 );

        const int heightMinusOne = OctreeHeight - 1;

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();
        // There is a maximum of 26 neighbors
        ContainerClass* neighbors[26];
        int neighborPositions[26];
        // for each leafs
        do{
            if( octreeIterator.getCurrentCell()->hasTargetsChild() ){
                FLOG(computationCounter.tic());
                if(l2pEnabled){
                    kernels->L2P(octreeIterator.getCurrentCell(), octreeIterator.getCurrentListTargets());
                }
                if(p2pEnabled){
                    if(octreeIterator.getCurrentCell()->hasSrcChild()){
                        kernels->P2P( octreeIterator.getCurrentGlobalCoordinate(), octreeIterator.getCurrentListTargets(),
                                      octreeIterator.getCurrentListSrc() , neighbors, neighborPositions, 0);
                    }
                    // need the current particles and neighbors particles
                    const int counter = tree->getLeafsNeighbors(neighbors, neighborPositions, octreeIterator.getCurrentGlobalCoordinate(), heightMinusOne);
                    kernels->P2PRemote( octreeIterator.getCurrentGlobalCoordinate(), octreeIterator.getCurrentListTargets(),
                              octreeIterator.getCurrentListSrc() , neighbors, neighborPositions, counter);
                }
                FLOG(computationCounter.tac());
                FLOG(totalComputation += computationCounter.elapsed());
            }
        } while(octreeIterator.moveRight());

        FLOG( counterTime.tac() );
        FLOG( FLog::Controller << "\tFinished (@Direct Pass (L2P + P2P) = "  << counterTime.elapsed() << " s)\n" );
        FLOG( FLog::Controller << "\t\t Computation L2P + P2P : " << totalComputation << " s\n" );

    }

};


#endif //FFMMALGORITHMTSM_HPP


