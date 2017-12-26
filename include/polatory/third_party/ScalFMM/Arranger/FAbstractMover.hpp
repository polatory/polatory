// See LICENCE file at project root

#ifndef FABSTRACTLEAFINTERFACE_HPP
#define FABSTRACTLEAFINTERFACE_HPP

template<class FReal,class OctreeClass,class ParticleClass>
class FAbstractMover{
public:
    virtual void getParticlePosition(ParticleClass* lf, const FSize idxPart, FPoint<FReal>* particlePos) = 0;
    virtual void removeFromLeafAndKeep(ParticleClass* lf, const FPoint<FReal>& particlePos, const FSize idxPart, FParticleType type) = 0;
    virtual void insertAllParticles(OctreeClass* tree) = 0;
};





#endif //FABSTRACTLEAFINTERFACE_HPP
