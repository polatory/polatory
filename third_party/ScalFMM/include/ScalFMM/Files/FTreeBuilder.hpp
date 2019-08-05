// See LICENCE file at project root
#ifndef FTREEBUILDER_H
#define FTREEBUILDER_H

#include "../Utils/FGlobal.hpp"

#include "../Utils/FLog.hpp"
#include "../Utils/FQuickSort.hpp"
#include "../Utils/FTic.hpp"
#include "../Utils/FAssert.hpp"
#include "../Containers/FOctree.hpp"
#include "../Containers/FTreeCoordinate.hpp"

#include "../Components/FBasicParticleContainer.hpp"

#include <omp.h>

#include <memory>

/**
* @author Cyrille Piacibello, Berenger Bramas
* @class FTreeBuilder
* @brief
* Please read the license
*
* This class provides a way to insert efficiently large amount of particles inside a tree.
*
* This is a static class. It's useless to instance it.  This class use
* the Threaded QuickSort or the output of FMpiTreeBuilder in order to
* sort the parts and insert them.
*
*/

template<class FReal, class OctreeClass, class LeafClass>
class FTreeBuilder{
private:
    /** This class is the relation between particles and their morton idx */
    struct IndexedParticle{
        MortonIndex mindex;
        FSize particlePositionInArray;

        // To sort according to the mindex
        bool operator<=(const IndexedParticle& rhs) const {
            return this->mindex <= rhs.mindex;
        }
    };

    /** In order to keep all the created leaves */
    struct LeafDescriptor{
        LeafClass* leafPtr;
        FSize offsetInArray;
        FSize nbParticlesInLeaf;
    };

public:

    /** Should be used to insert a FBasicParticleContainer class */
    template < unsigned NbAttributes, class AttributeClass>
    static void BuildTreeFromArray(OctreeClass*const tree, const FBasicParticleContainer<FReal, NbAttributes, AttributeClass>& particlesContainers,
                                   bool isAlreadySorted=false){
        const FSize numberOfParticle = particlesContainers.getNbParticles();
        // If the parts are already sorted, no need to sort again
        FLOG(FTic enumTimer, leavesPtr, leavesOffset );
        FLOG(FTic insertTimer, copyTimer);

        // General values needed
        const int NbLevels       = tree->getHeight();
        const FPoint<FReal> centerOfBox = tree->getBoxCenter();
        const FReal boxWidth     = tree->getBoxWidth();
        const FReal boxWidthAtLeafLevel = boxWidth/FReal(1 << (NbLevels - 1));
        const FPoint<FReal> boxCorner   = centerOfBox - boxWidth/2;

        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        /// We need to sort it in order to insert efficiently
        ////////////////////////////////////////////////////////////////

        // First, copy datas into an array that will be sorted and
        // set Morton index for each particle

        // Temporary FTreeCoordinate
        std::unique_ptr<IndexedParticle[]> particleIndexes(new IndexedParticle[numberOfParticle]);

        FLOG(copyTimer.tic());
        #pragma omp parallel for schedule(static)
        for(FSize idxParts=0; idxParts<numberOfParticle ; ++idxParts ){
            // Get the Morton Index
            const FTreeCoordinate host(
                     FCoordinateComputer::GetTreeCoordinate<FReal>(particlesContainers.getPositions()[0][idxParts] - boxCorner.getX(),
                                         boxWidth,boxWidthAtLeafLevel,  NbLevels),
                     FCoordinateComputer::GetTreeCoordinate<FReal>(particlesContainers.getPositions()[1][idxParts] - boxCorner.getY(),
                                          boxWidth,boxWidthAtLeafLevel, NbLevels ),
                     FCoordinateComputer::GetTreeCoordinate<FReal>(particlesContainers.getPositions()[2][idxParts] - boxCorner.getZ(),
                                          boxWidth,boxWidthAtLeafLevel, NbLevels )
            );
            // Store morton index and original idx
            particleIndexes[idxParts].mindex = host.getMortonIndex();
            particleIndexes[idxParts].particlePositionInArray = idxParts;
        }

        FLOG(copyTimer.tac());
        FLOG(FLog::Controller<<"Time needed for copying "<< numberOfParticle<<" particles : "<<copyTimer.elapsed() << " secondes !\n");

        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        /// Sort it needed
        ////////////////////////////////////////////////////////////////

        if(!isAlreadySorted){
            //Sort dat array
            FLOG(FTic sortTimer);
            FQuickSort<IndexedParticle,FSize>::QsOmp( particleIndexes.get(), numberOfParticle);
            FLOG(sortTimer.tac());
            FLOG(FLog::Controller << "Time needed for sorting the particles : "<< sortTimer.elapsed() << " secondes !\n");
        }

        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        /// Get the number of leaves
        ////////////////////////////////////////////////////////////////

        //Enumerate the different leaves AND copy the positions
        unsigned int numberOfLeaves = 0;
        FLOG(enumTimer.tic());
        {
            MortonIndex previousIndex = -1;

            for(FSize idxParts = 0 ; idxParts < numberOfParticle ; ++idxParts){
                // If not the same leaf, inc the counter
                if(particleIndexes[idxParts].mindex != previousIndex){
                    previousIndex   = particleIndexes[idxParts].mindex;
                    numberOfLeaves += 1;
                }
            }
        }

        FLOG(enumTimer.tac());
        FLOG(FLog::Controller << "Time needed for enumerate the leaves : "<< enumTimer.elapsed() << " secondes !\n");
        FLOG(FLog::Controller << "Found " << numberOfLeaves << " leaves differents. \n");

        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        /// Count the number of particles per leaf
        ////////////////////////////////////////////////////////////////

        FLOG(leavesOffset.tic());

        // Store the size of each leaves
        std::unique_ptr<LeafDescriptor[]> leavesDescriptor(new LeafDescriptor[numberOfLeaves]);
        memset(leavesDescriptor.get(), 0, sizeof(LeafDescriptor)*(numberOfLeaves));
        {
            //Init
            int currentLeafIndex = -1;
            MortonIndex currentMortonIndex = -1;

            for(FSize idxParts = 0 ; idxParts < numberOfParticle ; ++idxParts){
                // If not the same leaf
                if(particleIndexes[idxParts].mindex != currentMortonIndex){
                    FAssertLF(FSize(currentLeafIndex) < numberOfLeaves);
                    // Move to next descriptor
                    currentLeafIndex  += 1;
                    currentMortonIndex = particleIndexes[idxParts].mindex;
                    // Fill the descriptor
                    leavesDescriptor[currentLeafIndex].offsetInArray = idxParts;
                    leavesDescriptor[currentLeafIndex].leafPtr = tree->createLeaf(currentMortonIndex);
                    leavesDescriptor[currentLeafIndex].nbParticlesInLeaf = 0;
                }
                // Inc the number of particles in the current leaf
                leavesDescriptor[currentLeafIndex].nbParticlesInLeaf += 1;
            }
        }

        FLOG(leavesOffset.tac());
        FLOG(FLog::Controller << "Time needed for setting the offset of each leaves : "<< leavesOffset.elapsed() << " secondes !\n");
        //Then, we create the leaves inside the tree

        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        /// Insert multiple particles inside their corresponding leaf
        ////////////////////////////////////////////////////////////////

        FLOG(insertTimer.tic());

        // Copy each parts into corresponding Leaf
        #pragma omp parallel
        {
            std::array<AttributeClass, NbAttributes> particleAttr;

            const FReal*const partX = particlesContainers.getPositions()[0];
            const FReal*const partY = particlesContainers.getPositions()[1];
            const FReal*const partZ = particlesContainers.getPositions()[2];

            #pragma omp for schedule(static)
            for(FSize idxLeaf = 0 ; idxLeaf < numberOfLeaves ; ++idxLeaf ){
                const FSize nbParticlesAlreadyInLeaf = leavesDescriptor[idxLeaf].leafPtr->getSrc()->getNbParticles();
                // Reserve the space needed for the new particles
                leavesDescriptor[idxLeaf].leafPtr->getSrc()->reserve(nbParticlesAlreadyInLeaf + leavesDescriptor[idxLeaf].nbParticlesInLeaf);

                // For all particles
                for(FSize idxPart = 0 ; idxPart < leavesDescriptor[idxLeaf].nbParticlesInLeaf ; ++idxPart){
                    // Get position in the original container
                    const FSize particleOriginalPos = particleIndexes[leavesDescriptor[idxLeaf].offsetInArray + idxPart].particlePositionInArray;
                    // Get the original position
                    FPoint<FReal> particlePos( partX[particleOriginalPos],
                                        partY[particleOriginalPos],
                                        partZ[particleOriginalPos]);
                    // Copy the attributes
                    for(unsigned idxAttr = 0 ; idxAttr < NbAttributes; ++idxAttr){
                        particleAttr[idxAttr] = particlesContainers.getAttribute(idxAttr)[particleOriginalPos];
                    }
                    // Push the particle in the array
                    leavesDescriptor[idxLeaf].leafPtr->push(particlePos, particleAttr);
                }
            }
        }

        FLOG(insertTimer.tac());
        FLOG(FLog::Controller << "Time needed for inserting the parts into the leaves : "<< insertTimer.elapsed() << " secondes !\n");
    }
};




#endif //FTREEBUILDER_H
