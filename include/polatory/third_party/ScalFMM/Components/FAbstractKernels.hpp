// See LICENCE file at project root
#ifndef FABSTRACTKERNELS_HPP
#define FABSTRACTKERNELS_HPP


#include "../Utils/FGlobal.hpp"
#include "../Utils/FLog.hpp"
#include "../Containers/FTreeCoordinate.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FAbstractKernels
* @brief This class defines what any kernel has to implement.
*
* Please notice that P2PRemote is optional and should be implemented in case of
* MPI usage and certain algorithm.
* It is better to inherit from this class even if it is not obligatory thanks to
* the templates. But inheriting will force your class to have the correct parameters
* especially of you use the override keyword.
*
* You can find an example of implementation in FBasicKernels.
*/
template< class CellClass, class ContainerClass >
class FAbstractKernels{
public:
    /** Default destructor */
    virtual ~FAbstractKernels(){
    }

    /**
    * P2M
    * particles to multipole
    * @param pole the multipole to fill using the particles
    * @param particles the particles from the same spacial boxe
    */
    virtual void P2M(CellClass* const pole, const ContainerClass* const particles) = 0;

    /**
    * M2M
    * Multipole to multipole
    * @param pole the father (the boxe that contains other ones)
    * @param child the boxe to take values from
    * @param the current computation level
    * the child array has a size of 8 elements (address if exists or 0 otherwise).
    * You must test if a pointer is 0 to know if an element exists inside this array.
    * The order of the M2M are driven by the Morton indexing (the children appear in the
    * array in increasing Morton index).
    * In binary: xyz, so 100 means 1 in x, 0 in y, 0 in z.
    */
    virtual void M2M(CellClass* const FRestrict pole, const CellClass*const FRestrict *const FRestrict child, const int inLevel) = 0;

    /**
    * M2L
    * Multipole to local
    * @param local the element to fill using distant neighbors
    * @param distantNeighbors is an array containing fathers's direct neighbors's child - direct neigbors
    *        (max 189 in normal M2L and 189+26 with extended separation criteria)
    * @param neighborPositions the relative position of the neighbor (between O and 342)
    * @param size the number of neighbors
    * @param inLevel the current level of the computation
    *
    * The relative position is given by the formula:
    * (((xdiff+3) * 7) + (ydiff+3)) * 7 + zdiff + 3
    * with xdiff, ydiff and zdiff from -3 to 2 or from -2 to 3.
    *
    * The way of accessing the interaction list has been changed in a previous ScalFMM:
    * @code // the code was:
    * @code  for(int idxNeigh = 0 ; idxNeigh < 343 ; ++idxNeigh){
    * @code      // Test if a cell exists
    * @code      if(distantNeighbors[idxNeigh]){
    * @code           ...
    * @code      }
    * @code  }
    * @code  // Now it must be :
    * @code  for(int idxExistingNeigh = 0 ; idxExistingNeigh < size ; ++idxExistingNeigh){
    * @code      const int idxNeigh = neighborPositions[idxExistingNeigh]
    * @code      distantNeighbors[idxExistingNeigh]...
    * @code  }
    * It may have negligable extra cost in dense FMM but is clearly benefits for Sparse FMM.
        */
    virtual void M2L(CellClass* const FRestrict local, const CellClass* distantNeighbors[],
                     const int neighborPositions[],
                     const int size, const int inLevel) = 0;


    /** This method is used to bypass needFinishedM2LEvent method a each level
    *  during the transferPass.  If you have to use the  finishedLevelM2L then you
    *  have to  inherit it and return true rather than false.
    * But it will imply extra dependencies and even not be supported by some algorithms.
    *
    * @return false
    */
    constexpr static bool NeedFinishedM2LEvent(){
        return false;
    }

    /** This method can be optionally inherited
      * It is called at the end of each computation level during the M2L pass
      * @param level the ending level
      */
    virtual void finishedLevelM2L(const int /*level*/){
    }

    /**
        * L2L
        * Local to local
        * @param local the father to take value from
        * @param child the child to downward values (child may have already been impacted by M2L)
        * @param inLevel the current level of computation
        * the child array has a size of 8 elements (address if exists or 0 otherwise).
        * You must test if a pointer is 0 to know if an element exists inside this array.
        * The order of the M2M are driven by the Morton indexing (the children appear in the
        * array in increasing Morton index).
        * In binary: xyz, so 100 means 1 in x, 0 in y, 0 in z.
        */
    virtual void L2L(const CellClass* const FRestrict local, CellClass* FRestrict * const FRestrict child, const int inLevel) = 0;

    /**
        * L2P
        * Local to particles
        * @param local the leaf element (smaller boxe local element)
        * @param particles the list of particles inside this boxe
        */
    virtual void L2P(const CellClass* const local, ContainerClass* const particles) = 0;

    /**
        * P2P
        * Particles to particles
        * This functions should compute the inner interactions inside the target leaf,
        * and the interaction between the target and the list of neighbors.
        * This neighbor computation should be the same as P2POuter and thus,
        * internally this function may call P2POuter directly.
        *
        * The neighborPositions contains a value from 0 to 26 included which represent the relative
        * position of the neighbor given by:
        * (((idxX + 1) * 3) + (idxY +1)) * 3 + idxZ + 1
        * with idxX, idxY, and idxZ from -1 to 1.
        *
        *
        * @param inLeafPosition tree coordinate of the leaf (number of boxes in x, y and z)
        * @param targets current boxe targets particles
        * @param sources current boxe sources particles (can be == to targets)
        * @param directNeighborsParticles the particles from direct neighbors (max length = 26)
        * @param neighborPositions the relative position of the neighbors (between O and 25, max length = 26)
        * @param size the number of direct neighbors
        */
    virtual void P2P(const FTreeCoordinate& inLeafPosition,
                     ContainerClass* const FRestrict targets, const ContainerClass* const FRestrict sources,
                     ContainerClass* const directNeighborsParticles[], const int neighborPositions[],
                     const int size) = 0;

    /**
        * P2P
        * Particles to particles
        * This functions should compute the interaction between the target and the list of neighbors.
        * This neighbor computation should be the same as P2P and thus,
        * this function may be called by P2P.
        *
        * The neighborPositions contains a value from 0 to 26 included which represent the relative
        * position of the neighbor given by:
        * (((idxX + 1) * 3) + (idxY +1)) * 3 + idxZ + 1
        * with idxX, idxY, and idxZ from -1 to 1.
        *
        *
        * @param inLeafPosition tree coordinate of the leaf (number of boxes in x, y and z)
        * @param targets current boxe targets particles
        * @param directNeighborsParticles the particles from direct neighbors (max length = 26)
        * @param neighborPositions the relative position of the neighbors (between O and 25, max length = 26)
        * @param size the number of direct neighbors
        */
    virtual void P2POuter(const FTreeCoordinate& /*inLeafPosition*/,
                          ContainerClass* const FRestrict /*targets*/,
                          ContainerClass* const /*directNeighborsParticles*/[], const int /*neighborPositions*/[],
                          const int /*size*/)  = 0;

    /**
    * P2P
    * Particles to particles
    * This functions should compute the interaction between the target and the list of neighbors.
    *
    * There are no interest to compute mutual interaction involve in this function.
    * It is mainly called by the MPI algorithm with leaves from other hosts and thus
    * modifying the neighbors do not modify the original data:
    * directNeighborsParticles will be destroyed once all P2P remote have been performed.
    *
    * The neighborPositions contains a value from 0 to 26 included which represent the relative
    * position of the neighbor given by:
    * (((idxX + 1) * 3) + (idxY +1)) * 3 + idxZ + 1
    * with idxX, idxY, and idxZ from -1 to 1.
    *
    * The sources is given but the inner P2P must not be computed here.
    * It is given just in case the P2PRemote need it, but the interaction between
    * targets and sources should be done in the P2P function.
    *
    * @param inLeafPosition tree coordinate of the leaf (number of boxes in x, y and z)
    * @param targets current boxe targets particles
    * @param sources current boxe sources particles (can be == to targets)
    * @param directNeighborsParticles the particles from direct neighbors (max length = 26)
    * @param neighborPositions the relative position of the neighbors (between O and 25, max length = 26)
    * @param size the number of direct neighbors
    *
    */
    virtual void P2PRemote(const FTreeCoordinate& /*inLeafPosition*/,
                           ContainerClass* const FRestrict /*targets*/, const ContainerClass* const FRestrict /*sources*/,
                           const ContainerClass* const /*directNeighborsParticles*/[],
                           const int /*neighborPositions*/[], const int /*size*/) {
        FLOG( FLog::Controller.write("Warning, P2P remote is used but not implemented!").write(FLog::Flush) );
    }

};


#endif //FABSTRACTKERNELS_HPP


