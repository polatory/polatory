
// Keep in private GIT
#ifndef FP2PGROUPPARTICLECONTAINER_HPP
#define FP2PGROUPPARTICLECONTAINER_HPP

#include "FGroupAttachedLeaf.hpp"

template<class FReal, int NRHS = 1, int NLHS = 1, int NVALS = 1>
class FP2PGroupParticleContainer : public FGroupAttachedLeaf<FReal, NVALS*NRHS, NVALS*4*NLHS, FReal> {
    typedef FGroupAttachedLeaf<FReal, NVALS*NRHS, NVALS*4*NLHS, FReal> Parent;

public:
    FP2PGroupParticleContainer(){}
    FP2PGroupParticleContainer(const FSize inNbParticles, FReal* inPositionBuffer, const size_t inLeadingPosition,
                       FReal* inAttributesBuffer, const size_t inLeadingAttributes)
        : Parent(inNbParticles, inPositionBuffer, inLeadingPosition, inAttributesBuffer, inLeadingAttributes) {

    }

    FReal* getPhysicalValues(const int idxVals = 0, const int idxRhs = 0){
      return Parent::getAttribute((0+idxRhs)*NVALS+idxVals);
    }

    const FReal* getPhysicalValues(const int idxVals = 0, const int idxRhs = 0) const {
        return Parent::getAttribute((0+idxRhs)*NVALS+idxVals);
    }

    FReal* getPotentials(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getAttribute((NRHS+idxLhs)*NVALS+idxVals);
    }

    const FReal* getPotentials(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getAttribute((NRHS+idxLhs)*NVALS+idxVals);
    }

    FReal* getForcesX(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getAttribute((NRHS+NLHS+idxLhs)*NVALS+idxVals);
    }

    const FReal* getForcesX(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getAttribute((NRHS+NLHS+idxLhs)*NVALS+idxVals);
    }

    FReal* getForcesY(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getAttribute((NRHS+2*NLHS+idxLhs)*NVALS+idxVals);
    }

    const FReal* getForcesY(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getAttribute((NRHS+2*NLHS+idxLhs)*NVALS+idxVals);
    }

    FReal* getForcesZ(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getAttribute((NRHS+3*NLHS+idxLhs)*NVALS+idxVals);
    }

    const FReal* getForcesZ(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getAttribute((NRHS+3*NLHS+idxLhs)*NVALS+idxVals);
    }

    void resetForcesAndPotential(){
        for(int idx = 0 ; idx < 4*NLHS*NVALS ; ++idx){
            Parent::resetToInitialState(idx + NRHS*NVALS);
        }
    }

    int getNVALS() const {
        return NVALS;
    }

};

#endif // FP2PGROUPPARTICLECONTAINER_HPP
