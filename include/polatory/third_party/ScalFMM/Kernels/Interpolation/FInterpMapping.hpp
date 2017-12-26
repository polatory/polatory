// See LICENCE file at project root
#ifndef FINTERPMAPPING_HPP
#define FINTERPMAPPING_HPP

#include <iostream>
#include <limits>

#include "../../Utils/FNoCopyable.hpp"
#include "../../Utils/FPoint.hpp"

/**
 * @author Matthias Messner (matthias.matthias@inria.fr)
 * Please read the license
 */

/**
 * @class FInterpMapping
 *
 * The class @p FInterpMapping is the base class for the affine mapping
 * \f$\Phi:[-1,1]\rightarrow[a,b] \f$ and the inverse affine mapping
 * \f$\Phi^{-1}:[a,b]\rightarrow[-1,1]\f$.
 */
template <class FReal>
class FInterpMapping : FNoCopyable
{
protected:
    FPoint<FReal> a;
    FPoint<FReal> b;

    explicit FInterpMapping(const FPoint<FReal>& center,
                            const FReal width)
        : a(center.getX() - width / FReal(2.),
            center.getY() - width / FReal(2.),
            center.getZ() - width / FReal(2.)),
          b(center.getX() + width / FReal(2.),
            center.getY() + width / FReal(2.),
            center.getZ() + width / FReal(2.)) {}

    virtual void operator()(const FPoint<FReal>&, FPoint<FReal>&) const = 0;

public:
    /**
     * Checks wheter @p position is within cluster, ie within @p a and @p b, or
     * not.
     *
     * @param[in] position position (eg of a particle)
     * @return @p true if position is in cluster else @p false
     */
    bool is_in_cluster(const FPoint<FReal>& position) const
    {
        // Set numerical limit
        const FReal epsilon = FReal(10.) * std::numeric_limits<FReal>::epsilon();
        // Return false if x-coordinate is not in cluster
        if (a.getX()-position.getX()>epsilon ||	position.getX()-b.getX()>epsilon) {
            std::cout << a.getX()-position.getX() << "\t"
                      << position.getX()-b.getX() << "\t"	<< epsilon << std::endl;
            return false;
        }
        // Return false if x-coordinate is not in cluster
        if (a.getY()-position.getY()>epsilon ||	position.getY()-b.getY()>epsilon) {
            std::cout << a.getY()-position.getY() << "\t"
                      << position.getY()-b.getY() << "\t"	<< epsilon << std::endl;
            return false;
        }
        // Return false if x-coordinate is not in cluster
        if (a.getZ()-position.getZ()>epsilon ||	position.getZ()-b.getZ()>epsilon) {
            std::cout << a.getZ()-position.getZ() << "\t"
                      << position.getZ()-b.getZ() << "\t"	<< epsilon << std::endl;
            return false;
        }
        // Position is in cluster, return true
        return true;
    }

};


/**
 * @class map_glob_loc
 *
 * This class defines the inverse affine mapping
 * \f$\Phi^{-1}:[a,b]\rightarrow[-1,1]\f$. It maps from global coordinates to
 * local ones.
 */
template <class FReal>
class map_glob_loc : public FInterpMapping<FReal>
{
    using FInterpMapping<FReal>::a;
    using FInterpMapping<FReal>::b;
public:
    explicit map_glob_loc(const FPoint<FReal>& center, const FReal width)
        : FInterpMapping<FReal>(center, width) {}

    /**
     * Maps from a global position to its local position: \f$\Phi^{-1}(x) =
     * \frac{2x-b-a}{b-a}\f$.
     *
     * @param[in] globPos global position
     * @param[out] loclPos local position
     */
    void operator()(const FPoint<FReal>& globPos, FPoint<FReal>& loclPos) const
    {
        loclPos.setX((FReal(2.)*globPos.getX()-b.getX()-a.getX()) / (b.getX()-a.getX())); // 5 flops
        loclPos.setY((FReal(2.)*globPos.getY()-b.getY()-a.getY()) / (b.getY()-a.getY())); // 5 flops
        loclPos.setZ((FReal(2.)*globPos.getZ()-b.getZ()-a.getZ()) / (b.getZ()-a.getZ())); // 5 flops
    }

    // jacobian = 2 / (b - a);
    void computeJacobian(FPoint<FReal>& jacobian) const
    {
        jacobian.setX(FReal(2.) / (b.getX() - a.getX())); // 2 flops
        jacobian.setY(FReal(2.) / (b.getY() - a.getY()));	// 2 flops
        jacobian.setZ(FReal(2.) / (b.getZ() - a.getZ())); // 2 flops
    }
};


/**
 * @class map_loc_glob
 *
 * This class defines the affine mapping \f$\Phi:[-1,1]\rightarrow[a,b]\f$. It
 * maps from local coordinates to global ones.
 */
template <class FReal>
class map_loc_glob : public FInterpMapping<FReal>
{
    using FInterpMapping<FReal>::a;
    using FInterpMapping<FReal>::b;
public:
    explicit map_loc_glob(const FPoint<FReal>& center, const FReal width)
        : FInterpMapping<FReal>(center, width) {}

    // globPos = (a + b) / 2 + (b - a) * loclPos / 2;
    /**
     * Maps from a local position to its global position: \f$\Phi(\xi) =
     * \frac{1}{2}(a+b)+ \frac{1}{2}(b-a)\xi\f$.
     *
     * @param[in] loclPos local position
     * @param[out] globPos global position
     */
    void operator()(const FPoint<FReal>& loclPos, FPoint<FReal>& globPos) const
    {
        globPos.setX((a.getX()+b.getX())/FReal(2.)+
                     (b.getX()-a.getX())*loclPos.getX()/FReal(2.));
        globPos.setY((a.getY()+b.getY())/FReal(2.)+
                     (b.getY()-a.getY())*loclPos.getY()/FReal(2.));
        globPos.setZ((a.getZ()+b.getZ())/FReal(2.)+
                     (b.getZ()-a.getZ())*loclPos.getZ()/FReal(2.));
    }
};



#endif /*FUNIFTENSOR_HPP*/
