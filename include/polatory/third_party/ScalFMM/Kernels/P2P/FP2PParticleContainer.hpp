// See LICENCE file at project root
#ifndef FP2PPARTICLECONTAINER_HPP
#define FP2PPARTICLECONTAINER_HPP

#include "../../Components/FBasicParticleContainer.hpp"

template<class FReal, int NRHS = 1, int NLHS = 1, int NVALS = 1>
class FP2PParticleContainer : public FBasicParticleContainer<FReal, NVALS*(NRHS+4*NLHS), FReal> {
    typedef FBasicParticleContainer<FReal, NVALS*(NRHS+4*NLHS), FReal> Parent;

public:
    static const int NbAttributes = NVALS*(NRHS+4*NLHS);
    typedef FReal AttributesClass;

    FReal* getPhysicalValues(const int idxVals = 0, const int idxRhs = 0){
      return Parent::getAttribute((0+idxRhs)*NVALS+idxVals);
    }

    const FReal* getPhysicalValues(const int idxVals = 0, const int idxRhs = 0) const {
        return Parent::getAttribute((0+idxRhs)*NVALS+idxVals);
    }

    FReal* getPhysicalValuesArray(const int idxVals = 0, const int idxRhs = 0){
        return Parent::getRawData() + ((0+idxRhs)*NVALS+idxVals)*Parent::getLeadingRawData();
    }

    const FReal* getPhysicalValuesArray(const int idxVals = 0, const int idxRhs = 0) const {
        return Parent::getRawData() + ((0+idxRhs)*NVALS+idxVals)*Parent::getLeadingRawData();
    }

    FSize getLeadingDimension() const {
        return Parent::getLeadingRawData();
    }

    FReal* getPotentials(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getAttribute((NRHS+idxLhs)*NVALS+idxVals);
    }

    const FReal* getPotentials(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getAttribute((NRHS+idxLhs)*NVALS+idxVals);
    }

    FReal* getPotentialsArray(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getRawData() + ((NRHS+idxLhs)*NVALS+idxVals)*Parent::getLeadingRawData();
    }

    const FReal* getPotentialsArray(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getRawData() + ((NRHS+idxLhs)*NVALS+idxVals)*Parent::getLeadingRawData();
    }

    FReal* getForcesX(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getAttribute((NRHS+NLHS+idxLhs)*NVALS+idxVals);
    }

    const FReal* getForcesX(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getAttribute((NRHS+NLHS+idxLhs)*NVALS+idxVals);
    }

    FReal* getForcesXArray(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getRawData() + ((NRHS+NLHS+idxLhs)*NVALS+idxVals)*Parent::getLeadingRawData();
    }

    const FReal* getForcesXArray(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getRawData() + ((NRHS+NLHS+idxLhs)*NVALS+idxVals)*Parent::getLeadingRawData();
    }

    FReal* getForcesY(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getAttribute((NRHS+2*NLHS+idxLhs)*NVALS+idxVals);
    }

    const FReal* getForcesY(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getAttribute((NRHS+2*NLHS+idxLhs)*NVALS+idxVals);
    }

    FReal* getForcesYArray(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getRawData() + ((NRHS+2*NLHS+idxLhs)*NVALS+idxVals)*Parent::getLeadingRawData();
    }

    const FReal* getForcesYArray(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getRawData() + ((NRHS+2*NLHS+idxLhs)*NVALS+idxVals)*Parent::getLeadingRawData();
    }

    FReal* getForcesZ(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getAttribute((NRHS+3*NLHS+idxLhs)*NVALS+idxVals);
    }

    const FReal* getForcesZ(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getAttribute((NRHS+3*NLHS+idxLhs)*NVALS+idxVals);
    }

    FReal* getForcesZArray(const int idxVals = 0, const int idxLhs = 0){
        return Parent::getRawData() + ((NRHS+3*NLHS+idxLhs)*NVALS+idxVals)*Parent::getLeadingRawData();
    }

    const FReal* getForcesZArray(const int idxVals = 0, const int idxLhs = 0) const {
        return Parent::getRawData() + ((NRHS+3*NLHS+idxLhs)*NVALS+idxVals)*Parent::getLeadingRawData();
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

#endif // FP2PPARTICLECONTAINER_HPP
