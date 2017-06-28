#ifndef FPARTICULETYPEDINDEXEDMOVER_HPP
#define FPARTICULETYPEDINDEXEDMOVER_HPP

#include "FAbstractMover.hpp"
#include "../Containers/FVector.hpp"

/**
 * This class should be use with the octree arrange to move particles
 * that are typed (src/tgt) and stored in a FBasicParticleContainer
 */
template<class FReal, class OctreeClass, class ContainerClass >
class FParticleTypedIndexedMover : public FAbstractMover<FReal, OctreeClass, ContainerClass>{
private:
    ContainerClass toStoreRemovedSourceParts;
    ContainerClass toStoreRemovedTargetParts;
public:

    FParticleTypedIndexedMover(){
    }


    virtual ~FParticleTypedIndexedMover(){
    }

/** To get the position of the particle at idx idxPart in leaf lf */
    void getParticlePosition(ContainerClass* lf, const FSize idxPart, FPoint<FReal>* particlePos){
        (*particlePos) = FPoint<FReal>(lf->getPositions()[0][idxPart],lf->getPositions()[1][idxPart],lf->getPositions()[2][idxPart]);
    }

    /** Remove a particle but keep it to reinsert it later*/
    void removeFromLeafAndKeep(ContainerClass* lf, const FPoint<FReal>& particlePos, const FSize idxPart, FParticleType type){
        std::array<typename ContainerClass::AttributesClass, ContainerClass::NbAttributes> particleValues;
        for(int idxAttr = 0 ; idxAttr < ContainerClass::NbAttributes ; ++idxAttr){
            particleValues[idxAttr] = lf->getAttribute(idxAttr)[idxPart];
        }
        if(type == FParticleType::FParticleTypeTarget){
            toStoreRemovedTargetParts.push(particlePos,FParticleType::FParticleTypeTarget,lf->getIndexes()[idxPart],particleValues);
        }
        else{
            toStoreRemovedSourceParts.push(particlePos,FParticleType::FParticleTypeSource,lf->getIndexes()[idxPart],particleValues);
        }
        lf->removeParticles(&idxPart,1);
    }

    /** Reinsert the previously saved particles */
    void insertAllParticles(OctreeClass* tree){
        std::array<typename ContainerClass::AttributesClass, ContainerClass::NbAttributes> particleValues;

        for(FSize idxToInsert = 0; idxToInsert<toStoreRemovedSourceParts.getNbParticles() ; ++idxToInsert){
            for(int idxAttr = 0 ; idxAttr < ContainerClass::NbAttributes ; ++idxAttr){
                particleValues[idxAttr] = toStoreRemovedSourceParts.getAttribute(idxAttr)[idxToInsert];
            }
            const FPoint<FReal> particlePos(toStoreRemovedSourceParts.getPositions()[0][idxToInsert],
                                     toStoreRemovedSourceParts.getPositions()[1][idxToInsert],
                                     toStoreRemovedSourceParts.getPositions()[2][idxToInsert]);
            tree->insert(particlePos, FParticleType::FParticleTypeSource, toStoreRemovedSourceParts.getIndexes()[idxToInsert], particleValues);
        }

        for(FSize idxToInsert = 0; idxToInsert<toStoreRemovedTargetParts.getNbParticles() ; ++idxToInsert){
            for(int idxAttr = 0 ; idxAttr < ContainerClass::NbAttributes ; ++idxAttr){
                particleValues[idxAttr] = toStoreRemovedTargetParts.getAttribute(idxAttr)[idxToInsert];
            }
            const FPoint<FReal> particlePos(toStoreRemovedTargetParts.getPositions()[0][idxToInsert],
                                     toStoreRemovedTargetParts.getPositions()[1][idxToInsert],
                                     toStoreRemovedTargetParts.getPositions()[2][idxToInsert]);

            tree->insert(particlePos, FParticleType::FParticleTypeTarget, toStoreRemovedTargetParts.getIndexes()[idxToInsert], particleValues);
        }

        toStoreRemovedSourceParts.clear();
        toStoreRemovedTargetParts.clear();
    }


};

#endif //FPARTICULETYPEDINDEXEDMOVER_HPP
