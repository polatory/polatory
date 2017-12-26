// See LICENCE file at project root
#ifndef FSIMPLELEAF_HPP
#define FSIMPLELEAF_HPP


#include "FAbstractLeaf.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FSimpleLeaf
* @brief This class is used as a leaf in simple system (source AND target).
* Here there only one container stores all particles.
*/
template<class FReal, class ContainerClass>
class FSimpleLeaf : public FAbstractLeaf<FReal, ContainerClass> {
    ContainerClass particles; //! The container to store all the particles

public:
    /** Default destructor */
    virtual ~FSimpleLeaf(){
    }

    /**
    * To add a new particle in the leaf
    * @param inParticlePosition the new particle position to store in the current leaf
    * and the other parameters given by the user
    */
    template<typename... Args>
    void push(const FPoint<FReal>& inParticlePosition, Args ...  args){
	// We pass every thing to the container and let it manage
	this->particles.push(inParticlePosition, args...);
    }

    /**
    * To add several new particles
    * @param inParticlePosition array of the positions of the parts to be stored in that leaf
    * and the other parameters given by the user
    */
    template<typename... Args>
    void pushArray(const FPoint<FReal>* inParticlePosition, FSize numberOfParts, Args ...  args){
	// We pass every thing to the container and let it manage
	this->particles.pushArray(inParticlePosition,numberOfParts, args...);
    }


    /**
    * To get all the sources in a leaf
    * @return a pointer to the list of particles that are sources
    */
    ContainerClass* getSrc() {
	return &this->particles;
    }

    /**
    * To get all the target in a leaf
    * @return a pointer to the list of particles that are targets
    */
    ContainerClass* getTargets() {
	return &this->particles;
    }
};


#endif //FSIMPLELEAF_HPP
