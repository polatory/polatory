// See LICENCE file at project root
#ifndef FABSTRACTPARTICLECONTAINER_HPP
#define FABSTRACTPARTICLECONTAINER_HPP

#include "../Utils/FGlobal.hpp"
#include "../Utils/FLog.hpp"
#include "../Utils/FPoint.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @brief
* Please read the license
*
* This class define the method that every particle container
* has to implement.
*
* @warning Inherit from this class when implement a specific particle type
*/
template <class FReal>
class FAbstractParticleContainer {
public:
    /** Default destructor */
    virtual ~FAbstractParticleContainer(){
    }

    /**
     * This method should be inherited (or your leaf will do nothing)
     * the point is coming from the tree and is followed by what let the leaf
     * pass through its push method.
     */
    template<typename... Args>
    void push(const FPoint<FReal>& /*inParticlePosition*/, Args ... /*args*/){
        FLOG( FLog::Controller.write("Warning, push is not implemented!").write(FLog::Flush) );
    }
};


#endif //FABSTRACTPARTICLECONTAINER_HPP


