#ifndef FBASICPARTICLECONTAINERINDEXEDMOVER_HPP
#define FBASICPARTICLECONTAINERINDEXEDMOVER_HPP

#include "FAbstractMover.hpp"
#include "../Containers/FVector.hpp"
/**
 * This class should be use with the octree arrange to move particles
 * that are stored in a FBasicParticleContainer
 */
template<class FReal, class OctreeClass, class ContainerClass >
class FBasicParticleContainerIndexedMover : public FAbstractMover<FReal,OctreeClass, ContainerClass>{
private:
    ContainerClass toStoreRemovedParts;

public:
    FBasicParticleContainerIndexedMover() {
    }

    virtual ~FBasicParticleContainerIndexedMover(){
    }

    /** To get the position of the particle at idx idxPart in leaf lf */
    void getParticlePosition(ContainerClass* lf, const FSize idxPart, FPoint<FReal>* particlePos){
        (*particlePos) = FPoint<FReal>(lf->getPositions()[0][idxPart],lf->getPositions()[1][idxPart],lf->getPositions()[2][idxPart]);
    }

    /** Remove a particle but keep it to reinsert it later*/
    void removeFromLeafAndKeep(ContainerClass* lf, const FPoint<FReal>& particlePos, const FSize idxPart, FParticleType /*type*/){
        std::array<typename ContainerClass::AttributesClass, ContainerClass::NbAttributes> particleValues;
        for(int idxAttr = 0 ; idxAttr < ContainerClass::NbAttributes ; ++idxAttr){
            particleValues[idxAttr] = lf->getAttribute(idxAttr)[idxPart];
        }

        toStoreRemovedParts.push(particlePos,lf->getIndexes()[idxPart],particleValues);

        lf->removeParticles(&idxPart,1);
    }

    /** Reinsert the previously saved particles */
    void insertAllParticles(OctreeClass* tree){
        std::array<typename ContainerClass::AttributesClass, ContainerClass::NbAttributes> particleValues;

        for(FSize idxToInsert = 0; idxToInsert<toStoreRemovedParts.getNbParticles() ; ++idxToInsert){
            for(int idxAttr = 0 ; idxAttr < ContainerClass::NbAttributes ; ++idxAttr){
                particleValues[idxAttr] = toStoreRemovedParts.getAttribute(idxAttr)[idxToInsert];
            }
            const FPoint<FReal> particlePos(toStoreRemovedParts.getPositions()[0][idxToInsert],
                                     toStoreRemovedParts.getPositions()[1][idxToInsert],
                                     toStoreRemovedParts.getPositions()[2][idxToInsert]);

            tree->insert(particlePos, toStoreRemovedParts.getIndexes()[idxToInsert], particleValues);
        }

        toStoreRemovedParts.clear();
    }
};

#endif // FBASICPARTICLECONTAINERINDEXEDMOVER_HPP
