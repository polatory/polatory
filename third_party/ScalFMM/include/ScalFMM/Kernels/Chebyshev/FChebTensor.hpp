// See LICENCE file at project root
#ifndef FCHEBTENSOR_HPP
#define FCHEBTENSOR_HPP

#include "../../Utils/FMath.hpp"

#include "./FChebRoots.hpp"
#include "./../Interpolation/FInterpTensor.hpp"


/**
 * @author Matthias Messner (matthias.matthias@inria.fr)
 * Please read the license
 */



/**
 * @class FChebTensor
 *
 * The class FChebTensor provides function considering the tensor product
 * interpolation.
 *
 * @tparam ORDER interpolation order \f$\ell\f$
 */
template <class FReal, int ORDER>
class FChebTensor : public FInterpTensor<FReal, ORDER,FChebRoots<FReal, ORDER>>
{
    enum {nnodes = TensorTraits<ORDER>::nnodes};
    typedef FChebRoots<FReal, ORDER> BasisType;
    typedef FInterpTensor<FReal, ORDER,BasisType> ParentTensor;

public:

    /**
   * Sets the roots of the Chebyshev quadrature weights defined as \f$w_i =
   * \frac{\pi}{\ell}\sqrt{1-\bar x_i^2}\f$ with the Chebyshev roots \f$\bar
   * x\f$.
   *
   * @param weights[out] the root of the weights \f$\sqrt{w_i}\f$
   */
    static
    void setRootOfWeights(FReal weights[nnodes])
    {
        // weights in 1d
        FReal weights_1d[ORDER];
        for (unsigned int o=0; o<ORDER; ++o)
            weights_1d[o] = FMath::FPi<FReal>()/ORDER * FMath::Sqrt(FReal(1.)-FReal(BasisType::roots[o])*FReal(BasisType::roots[o]));
        // weights in 3d (tensor structure)
        unsigned int node_ids[nnodes][3];
        ParentTensor::setNodeIds(node_ids);
        for (unsigned int n=0; n<nnodes; ++n) {
            weights[n] = FMath::Sqrt(weights_1d[node_ids[n][0]]*weights_1d[node_ids[n][1]]*weights_1d[node_ids[n][2]]);
        }
    }

};


#endif
