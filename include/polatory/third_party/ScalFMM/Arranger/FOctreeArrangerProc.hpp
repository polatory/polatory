// See LICENCE file at project root
#ifndef FOCTREEARRANGERPROC_HPP
#define FOCTREEARRANGERPROC_HPP

#include "../Utils/FGlobal.hpp"
#include "../Containers/FVector.hpp"
#include "../Utils/FAssert.hpp"
#include "../Utils/FMpi.hpp"

#include "../Utils/FGlobalPeriodic.hpp"

/**
* This example show how to use the FOctreeArrangerProc.
* @example testOctreeRearrangeProc.cpp
*/

/** @brief This class is an arranger, it move the particles that need to be hosted in a different leaf. This is the parallel version that use MPI.
  *
  * For example, if a simulation has been executed and the position
  * of the particles have been changed, then it may be better
  * to move the particles in the tree instead of building a new
  * tree.
  */
template <class FReal, class OctreeClass, class ContainerClass, class ParticleClass, class ConverterClass >
class FOctreeArrangerProc  {
    /** Interval is the min/max morton index
      * for a proc
      */
    struct Interval{
        MortonIndex min;
        MortonIndex max;
    };


    /** Find the interval that contains mindex */
    int getInterval(const MortonIndex mindex, const int size, const Interval intervals[]) const{
        for(int idxProc = 0 ; idxProc < size ; ++idxProc){
            // does it contains the index?
            if( intervals[idxProc].min <= mindex && mindex < intervals[idxProc].max){
                return idxProc;
            }
        }
        // if no interval found return the lastest one
        return size - 1;
    }

    OctreeClass* const tree;


public:
    /** Basic constructor */
    FOctreeArrangerProc(OctreeClass* const inTree) : tree(inTree) {
        FAssertLF(tree, "Tree cannot be null");
    }

    /** return false if the tree is empty after processing */
    bool rearrange(const FMpi::FComm& comm, const int isPeriodic = DirNone){
        // interval of each procs
        Interval*const intervals = new Interval[comm.processCount()];
        memset(intervals, 0, sizeof(Interval) * comm.processCount());

        {   // We need to exchange interval of each process, this interval
            // will be based on the current morton min max
            Interval myLastInterval;

            // take fist index
            typename OctreeClass::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();
            myLastInterval.min = octreeIterator.getCurrentGlobalIndex();
            // take last index
            octreeIterator.gotoRight();
            myLastInterval.max = octreeIterator.getCurrentGlobalIndex();

            // We get the min/max indexes from each procs
            FMpi::MpiAssert( MPI_Allgather( &myLastInterval, sizeof(Interval), MPI_BYTE, intervals, sizeof(Interval), MPI_BYTE, comm.getComm()),  __LINE__ );

            // increase interval in the empty morton index
            intervals[0].min = 0;
            for(int idxProc = 1 ; idxProc < comm.processCount() ; ++idxProc){
                intervals[idxProc].min = ((intervals[idxProc].min - intervals[idxProc-1].max)/2) + intervals[idxProc-1].max;
                intervals[idxProc-1].max = intervals[idxProc].min;
            }

            intervals[comm.processCount() - 1].max = ((1 << (3*(tree->getHeight()-1))) - 1);
        }

        // Particles that move
        FVector<ParticleClass>*const toMove = new FVector<ParticleClass>[comm.processCount()];

        { // iterate on the leafs and found particle to remove or to send
            // For periodic
            const FReal boxWidth = tree->getBoxWidth();
            const FPoint<FReal> min(tree->getBoxCenter(),-boxWidth/2);
            const FPoint<FReal> max(tree->getBoxCenter(),boxWidth/2);

            FVector<FSize> indexesToExtract;

            typename OctreeClass::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();
            do{
                const MortonIndex currentIndex = octreeIterator.getCurrentGlobalIndex();
                ContainerClass* particles = octreeIterator.getCurrentLeaf()->getSrc();
                //IdxPart is incremented at the end of the loop
                for(FSize idxPart = 0 ; idxPart < particles->getNbParticles(); /*++idxPart*/){
                    FPoint<FReal> partPos( particles->getPositions()[0][idxPart],
                            particles->getPositions()[1][idxPart],
                            particles->getPositions()[2][idxPart] );
                    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                    if( TestPeriodicCondition(isPeriodic, DirPlusX) ){
                        while(partPos.getX() >= max.getX()){
                            partPos.incX(-boxWidth);
                        }
                    }
                    else if(partPos.getX() >= max.getX()){
                        printf("Error, particle out of Box in +X, index %lld\n", currentIndex);
                        printf("Application is exiting...\n");
                    }
                    if( TestPeriodicCondition(isPeriodic, DirMinusX) ){
                        while(partPos.getX() < min.getX()){
                            partPos.incX(boxWidth);
                        }
                    }
                    else if(partPos.getX() < min.getX()){
                        printf("Error, particle out of Box in -X, index %lld\n", currentIndex);
                        printf("Application is exiting...\n");
                    }
                    // YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
                    if( TestPeriodicCondition(isPeriodic, DirPlusY) ){
                        while(partPos.getY() >= max.getY()){
                            partPos.incY(-boxWidth);
                        }
                    }
                    else if(partPos.getY() >= max.getY()){
                        printf("Error, particle out of Box in +Y, index %lld\n", currentIndex);
                        printf("Application is exiting...\n");
                    }
                    if( TestPeriodicCondition(isPeriodic, DirMinusY) ){
                        while(partPos.getY() < min.getY()){
                            partPos.incY(boxWidth);
                        }
                    }
                    else if(partPos.getY() < min.getY()){
                        printf("Error, particle out of Box in -Y, index %lld\n", currentIndex);
                        printf("Application is exiting...\n");
                    }
                    // ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
                    if( TestPeriodicCondition(isPeriodic, DirPlusX) ){
                        while(partPos.getZ() >= max.getZ()){
                            partPos.incZ(-boxWidth);
                        }
                    }
                    else if(partPos.getZ() >= max.getZ()){
                        printf("Error, particle out of Box in +Z, index %lld\n", currentIndex);
                        printf("Application is exiting...\n");
                    }
                    if( TestPeriodicCondition(isPeriodic, DirMinusX) ){
                        while(partPos.getZ() < min.getZ()){
                            partPos.incZ(boxWidth);
                        }
                    }
                    else if(partPos.getZ() < min.getZ()){
                        printf("Error, particle out of Box in -Z, index %lld\n", currentIndex);
                        printf("Application is exiting...\n");
                    }
                    // set pos
                    particles->getWPositions()[0][idxPart] = partPos.getX();
                    particles->getWPositions()[1][idxPart] = partPos.getY();
                    particles->getWPositions()[2][idxPart] = partPos.getZ();

                    const MortonIndex particuleIndex = tree->getMortonFromPosition(partPos);
                    // is this particle need to be changed from its leaf
                    if(particuleIndex != currentIndex){
                        // find the right interval
                        const int procConcerned = getInterval( particuleIndex, comm.processCount(), intervals);
                        toMove[procConcerned].push(ConverterClass::GetParticleAndRemove(particles,idxPart));
                        indexesToExtract.push(idxPart);
                        //No need to increment idxPart, since the array has been staggered
                    }
                    else{
                        idxPart++;
                    }
                }

                particles->removeParticles(indexesToExtract.data(), indexesToExtract.getSize());
                indexesToExtract.clear();

            } while(octreeIterator.moveRight());
        }

        // To send and recv
        ParticleClass* toReceive = nullptr;
        MPI_Request*const requests = new MPI_Request[comm.processCount()*2];
        memset(requests, 0, sizeof(MPI_Request) * comm.processCount() * 2);
        FSize*const indexToReceive = new FSize[comm.processCount() + 1];
        memset(indexToReceive, 0, sizeof(FSize) * comm.processCount() + 1);

        int iterRequests = 0;
        int limitRecvSend = 0;
        int hasToRecvFrom = 0;

        { // gather what to send to who + isend data
            FSize*const counter = new FSize[comm.processCount()];
            memset(counter, 0, sizeof(FSize) * comm.processCount());

            for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
                counter[idxProc] = toMove[idxProc].getSize();
            }
            // say who send to who
            FSize*const allcounter = new FSize[comm.processCount()*comm.processCount()];
            FMpi::MpiAssert( MPI_Allgather( counter, comm.processCount(), FMpi::GetType(*counter), allcounter, comm.processCount(),
                                            FMpi::GetType(*counter), comm.getComm()),  __LINE__ );

            // prepare buffer to receive
            FSize sumToRecv = 0;
            indexToReceive[0] = 0;
            for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
                if( idxProc != comm.processId()){
                    sumToRecv += allcounter[idxProc * comm.processCount() + comm.processId()];
                }
                indexToReceive[idxProc + 1] = sumToRecv;
            }
            toReceive = new ParticleClass[sumToRecv];

            // send
            for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
                if(idxProc != comm.processId() && allcounter[idxProc * comm.processCount() + comm.processId()]){
                    FAssertLF( allcounter[idxProc * comm.processCount() + comm.processId()] * sizeof(ParticleClass) < std::numeric_limits<int>::max());
                    FMpi::MpiAssert( MPI_Irecv(&toReceive[indexToReceive[idxProc]], int(allcounter[idxProc * comm.processCount() + comm.processId()] * sizeof(ParticleClass)), MPI_BYTE,
                              idxProc, 0, comm.getComm(), &requests[iterRequests++]),  __LINE__ );
                    hasToRecvFrom += 1;
                }
            }

            limitRecvSend = iterRequests;

            // recv
            for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
                if(idxProc != comm.processId() && toMove[idxProc].getSize()){
                    FAssertLF( toMove[idxProc].getSize() * sizeof(ParticleClass) < std::numeric_limits<int>::max());
                    FMpi::MpiAssert( MPI_Isend(toMove[idxProc].data(), int(toMove[idxProc].getSize() * sizeof(ParticleClass)), MPI_BYTE,
                              idxProc, 0, comm.getComm(), &requests[iterRequests++]),  __LINE__ );
                }
            }

            delete[] allcounter;
            delete[] counter;
        }

        { // insert particles that moved
            for(FSize idxPart = 0 ; idxPart < toMove[comm.processId()].getSize() ; ++idxPart){
                ConverterClass::Insert( tree , toMove[comm.processId()][idxPart]);
            }
        }

        {   // wait any recv or send
            // if it is a recv then insert particles
            MPI_Status status;
            while( hasToRecvFrom ){
                int done = 0;
                FMpi::MpiAssert( MPI_Waitany( iterRequests, requests, &done, &status ),  __LINE__ );
                if( done < limitRecvSend ){
                    const int source = status.MPI_SOURCE;
                    for(FSize idxPart = indexToReceive[source] ; idxPart < indexToReceive[source+1] ; ++idxPart){
                        ConverterClass::Insert( tree , toReceive[idxPart]);
                    }
                    hasToRecvFrom -= 1;
                }
            }
        }

        int counterLeavesAlive = 0;
        { // Remove empty leaves
            typename OctreeClass::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();
            bool workOnNext = true;

            do{
                // Empty leaf
                if( octreeIterator.getCurrentListTargets()->getNbParticles() == 0 ){
                    const MortonIndex currentIndex = octreeIterator.getCurrentGlobalIndex();
                    workOnNext = octreeIterator.moveRight();
                    tree->removeLeaf( currentIndex );
                }
                // Not empty, just continue
                else {
                    workOnNext = octreeIterator.moveRight();
                    counterLeavesAlive += 1;
                }
            } while( workOnNext );
        }

        // wait all send
        FMpi::MpiAssert( MPI_Waitall( iterRequests, requests, MPI_STATUSES_IGNORE),  __LINE__ );

        delete[] intervals;
        delete[] toMove;
        delete[] requests;
        delete[] toReceive;
        delete[] indexToReceive;

        // return false if tree is empty
        return counterLeavesAlive != 0;
    }

};

#endif // FOCTREEARRANGERPROC_HPP
