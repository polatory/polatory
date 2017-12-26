// See LICENCE file at project root
#ifndef FBASICKERNELS_HPP
#define FBASICKERNELS_HPP


#include "FAbstractKernels.hpp"


/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class AbstractKernels
* @brief This kernels is empty and does nothing.
*
* It can be inherited to define only what you want.
*/
template< class CellClass, class ContainerClass>
class FBasicKernels : public FAbstractKernels<CellClass,ContainerClass> {
public:
    /** Default destructor */
    virtual ~FBasicKernels(){
    }

    /** Do nothing */
    virtual void P2M(CellClass* const /*targetCell*/, const ContainerClass* const /*sourceParticles*/) override {

    }

    /** Do nothing */
    virtual void M2M(CellClass* const FRestrict /*parentCell*/, const CellClass*const FRestrict *const FRestrict /*children*/, const int /*level*/) override {

    }

    /** Do nothing */
    virtual void M2L(CellClass* const FRestrict /*targetLocal*/, const CellClass* /*sourceMultipoles*/[],
                     const int /*relativePostions*/[], const int /*nbInteractions*/, const int /*level*/) override {

    }

    /** Do nothing */
    virtual void L2L(const CellClass* const FRestrict /*parentCell*/, CellClass* FRestrict *const FRestrict  /*children*/, const int /*level*/) override {

    }

    /** Do nothing */
    virtual void L2P(const CellClass* const /*sourceCell*/, ContainerClass* const /*targetPaticles*/) override {

    }


    /** Do nothing */
    virtual void P2P(const FTreeCoordinate& /*treeCoord*/,
                     ContainerClass* const FRestrict /*targetParticles*/, const ContainerClass* const FRestrict /*sourceParticles*/,
                     ContainerClass* const /*neigbhorsParticles*/[], const int /*neighborPositions*/[], const int /*nbNeighbors*/) override {

    }

    /** Do nothing */
    virtual void P2POuter(const FTreeCoordinate& /*inLeafPosition*/,
                          ContainerClass* const FRestrict /*targets*/,
                          ContainerClass* const /*directNeighborsParticles*/[], const int /*neighborPositions*/[],
                          const int /*size*/) override {

    }

    /** Do nothing */
    virtual void P2PRemote(const FTreeCoordinate& /*treeCoord*/,
                           ContainerClass* const FRestrict /*targetParticles*/, const ContainerClass* const FRestrict /*sourceParticles*/,
                           const ContainerClass* const /*neigbhorsParticles*/[], const int /*neighborPositions*/[], const int /*nbNeighbors*/) override {

    }

};


#endif //FBASICKERNELS_HPP


