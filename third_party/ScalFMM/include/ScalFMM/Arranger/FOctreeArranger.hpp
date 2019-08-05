// See LICENCE file at project root
#ifndef FOCTREEARRANGER_HPP
#define FOCTREEARRANGER_HPP

#include "../Utils/FGlobal.hpp"
#include "../Utils/FPoint.hpp"
#include "../Containers/FVector.hpp"
#include "../Utils/FAssert.hpp"

#include "../Utils/FGlobalPeriodic.hpp"
#include "../Utils/FAssert.hpp"
#include "../Components/FParticleType.hpp"
/**
* This example show how to use the FOctreeArranger.
* @example testOctreeRearrange.cpp
*/


/**
* @brief This class is an arranger, it moves the particles that need to be hosted in a different leaf.
*
* For example, if a simulation has been executed and the position
* of the particles have been changed, then it may be better
* to move the particles in the tree instead of building a new
* tree.
*/
template <class FReal, class OctreeClass, class ContainerClass, class MoverClass >
class FOctreeArranger {
    OctreeClass* const tree; //< The tree to work on

public:
    FReal boxWidth;
    FPoint<FReal> MinBox;
    FPoint<FReal> MaxBox;
    MoverClass* interface;

public:
    /** Basic constructor */
    explicit FOctreeArranger(OctreeClass* const inTree) : tree(inTree), boxWidth(tree->getBoxWidth()),
                                                 MinBox(tree->getBoxCenter(),-tree->getBoxWidth()/2),
                                                 MaxBox(tree->getBoxCenter(),tree->getBoxWidth()/2),
                                                 interface(nullptr){
        FAssertLF(tree, "Tree cannot be null" );
        interface = new MoverClass;
    }

    virtual ~FOctreeArranger(){
        delete interface;
    }

    virtual void checkPosition(FPoint<FReal>& particlePos){
        // Assert
        FAssertLF(   MinBox.getX() < particlePos.getX() && MaxBox.getX() > particlePos.getX()
                  && MinBox.getY() < particlePos.getY() && MaxBox.getY() > particlePos.getY()
                  && MinBox.getZ() < particlePos.getZ() && MaxBox.getZ() > particlePos.getZ());
    }


    void rearrange(){
        {
            typename OctreeClass::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();
            do{
                const MortonIndex currentMortonIndex = octreeIterator.getCurrentGlobalIndex();
                //First we test sources
                ContainerClass * particles = octreeIterator.getCurrentLeaf()->getSrc();
                for(FSize idxPart = 0 ; idxPart < particles->getNbParticles(); /*++idxPart*/){
                    FPoint<FReal> currentPart;
                    interface->getParticlePosition(particles,idxPart,&currentPart);
                    checkPosition(currentPart);
                    const MortonIndex particuleIndex = tree->getMortonFromPosition(currentPart);
                    if(particuleIndex != currentMortonIndex){
                        //Need to move this one
                        interface->removeFromLeafAndKeep(particles,currentPart,idxPart,FParticleType::FParticleTypeSource);
                    }
                    else{
                        //Need to increment idx;
                        ++idxPart;
                    }
                }
                //Then we test targets
                if(octreeIterator.getCurrentLeaf()->getTargets() != particles){ //Leaf is TypedLeaf
                    ContainerClass * particleTargets = octreeIterator.getCurrentLeaf()->getTargets();
                    for(FSize idxPart = 0 ; idxPart < particleTargets->getNbParticles(); /*++idxPart*/){
                        FPoint<FReal> currentPart;
                        interface->getParticlePosition(particleTargets,idxPart,&currentPart);
                        checkPosition(currentPart);
                        const MortonIndex particuleIndex = tree->getMortonFromPosition(currentPart);
                        if(particuleIndex != currentMortonIndex){
                            //Need to move this one
                            interface->removeFromLeafAndKeep(particleTargets,currentPart,idxPart, FParticleType::FParticleTypeTarget);
                        }
                        else{
                            //Need to increment idx;
                            ++idxPart;
                        }
                    }
                }
            }while(octreeIterator.moveRight());
        }
        printf("Insert back particles\n");
        //Insert back the parts that have been removed
        interface->insertAllParticles(tree);

        //Then, remove the empty leaves
        { // Remove empty leaves
            typename OctreeClass::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();
            bool workOnNext = true;
            do{
                // Empty leaf
                if( octreeIterator.getCurrentListTargets()->getNbParticles() == 0 &&
                    octreeIterator.getCurrentListSrc()->getNbParticles() == 0 ){
                    const MortonIndex currentIndex = octreeIterator.getCurrentGlobalIndex();
                    workOnNext = octreeIterator.moveRight();
                    tree->removeLeaf( currentIndex );
                }
                // Not empty, just continue
                else {
                    workOnNext = octreeIterator.moveRight();
                }
            } while( workOnNext );
        }
    }
};

#endif // FOCTREEARRANGER_HPP
