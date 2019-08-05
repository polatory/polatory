// See LICENCE file at project root
#ifndef FCHEBROOTS_HPP
#define FCHEBROOTS_HPP

#include <cmath>
#include <limits>
#include <cassert>
#include <array>

#include "../../Utils/FNoCopyable.hpp"


/**
 * @author Matthias Messner (matthias.matthias@inria.fr)
 * Please read the license
 */

/**
 * @class FChebRoots
 *
 * The class @p FChebRoots provides the Chebyshev roots of order \f$\ell\f$
 * and the Chebyshev polynomials of first kind \f$T_n(x)\f$ and second kind
 * \f$U_{n-1}(x)\f$.
 *
 * @tparam ORDER interpolation order \f$\ell\f$
 */
template <class FReal, int ORDER>
struct FChebRoots : FNoCopyable
{
    enum {order = ORDER}; //!< interpolation order

    /**
     * Chebyshev roots in [-1,1] computed as \f$\bar x_n =
     * \cos\left(\frac{\pi}{2}\frac{2n-1}{\ell}\right)\f$ for
     * \f$n=1,\dots,\ell\f$
     */
    const static std::array<FReal, ORDER> roots;

    /**
   * Chebyshev polynomials of first kind \f$ T_n(x) = \cos(n \arccos(x)) \f$
     *
     * @param[in] n index
     * @param[in] x coordinate in [-1,1]
     * @return function value
   */
    static FReal T(const unsigned int n, FReal x)
    {
        //std::cout << x << std::endl;
        assert(std::abs(x)-1.<10.*std::numeric_limits<FReal>::epsilon());
        if (std::abs(x)>1.) {
            x = (x > FReal( 1.) ? FReal( 1.) : x);
            x = (x < FReal(-1.) ? FReal(-1.) : x);
        }

        return FReal(cos(n * acos(x)));
    }


    /**
     * For the derivation of the Chebyshev polynomials of first kind \f$
   * \frac{\mathrm{d} T_n(x)}{\mathrm{d}x} = n U_{n-1}(x) \f$ the Chebyshev
   * polynomials of second kind \f$ U_{n-1}(x) = \frac{\sin(n
   * \arccos(x))}{\sqrt{1-x^2}} \f$ are needed.
     *
     * @param[in] n index
     * @param[in] x coordinate in [-1,1]
     * @return function value
   */
    static FReal U(const unsigned int n, FReal x)
    {
        assert(std::abs(x)-1.<10.*std::numeric_limits<FReal>::epsilon());
        if (std::abs(x)>1.) {
            x = (x > FReal( 1.) ? FReal( 1.) : x);
            x = (x < FReal(-1.) ? FReal(-1.) : x);
        }

        return FReal(n * (sin(n * acos(x))) / sqrt(1 - x*x));
    }
};

template<int ORDER>
struct FChebRootsCore{};

template<class FReal, int ORDER>
const std::array<FReal,ORDER> FChebRoots<FReal,ORDER>::roots = FChebRootsCore<ORDER>::template Build<FReal>();

// We declare the roots here only once Please look to .cpp for definitions


// order 2
template<>
struct FChebRootsCore<2>{
    template <class FReal>
    static std::array<FReal,2> Build(){
        return { { FReal(-0.707106781186548),
                   FReal( 0.707106781186547)} };
    }
};

// order 3
template<>
struct FChebRootsCore<3>{
    template <class FReal>
    static std::array<FReal,3> Build(){
        return { {FReal(-8.66025403784439e-01),
                  FReal( 0.0                 ),
                  FReal( 8.66025403784438e-01)} };
    }
};

// order 4
template<>
struct FChebRootsCore<4>{
    template <class FReal>
    static std::array<FReal,4> Build(){
        return { {FReal(-0.923879532511287),
                  FReal(-0.382683432365090),
                  FReal( 0.382683432365090),
                  FReal( 0.923879532511287) } };
    }
};

// order 5
template<>
struct FChebRootsCore<5>{
    template <class FReal>
    static std::array<FReal,5> Build(){
        return { {FReal(-9.51056516295154e-01),
                  FReal(-5.87785252292473e-01),
                  FReal( 0.0                 ),
                  FReal( 5.87785252292473e-01),
                  FReal( 9.51056516295154e-01) } };
    }
};

// order 6
template<>
struct FChebRootsCore<6>{
    template <class FReal>
    static std::array<FReal,6> Build(){
        return { {FReal(-0.965925826289068),
                  FReal(-0.707106781186548),
                  FReal(-0.258819045102521),
                  FReal( 0.258819045102521),
                  FReal( 0.707106781186547),
                  FReal( 0.965925826289068)} };
    }
};

// order 7
template<>
struct FChebRootsCore<7>{
    template <class FReal>
    static std::array<FReal,7> Build(){
        return {{FReal(-9.74927912181824e-01),
                 FReal(-7.81831482468030e-01),
                 FReal(-4.33883739117558e-01),
                 FReal( 0.0                 ),
                 FReal( 4.33883739117558e-01),
                 FReal( 7.81831482468030e-01),
                 FReal( 9.74927912181824e-01) }};
    }
};

// order 8
template<>
struct FChebRootsCore<8>{
    template <class FReal>
    static std::array<FReal,8> Build(){
        return { {FReal(-0.980785280403230),
                  FReal(-0.831469612302545),
                  FReal(-0.555570233019602),
                  FReal(-0.195090322016128),
                  FReal( 0.195090322016128),
                  FReal( 0.555570233019602),
                  FReal( 0.831469612302545),
                  FReal( 0.980785280403230) } };
    }
};

// order 9
template<>
struct FChebRootsCore<9>{
    template <class FReal>
    static std::array<FReal,9> Build(){
        return { {FReal(-9.84807753012208e-01),
                  FReal(-8.66025403784439e-01),
                  FReal(-6.42787609686539e-01),
                  FReal(-3.42020143325669e-01),
                  FReal( 0.0                 ),
                  FReal( 3.42020143325669e-01),
                  FReal( 6.42787609686539e-01),
                  FReal( 8.66025403784438e-01),
                  FReal( 9.84807753012208e-01)} };
    }
};

// order 10
template<>
struct FChebRootsCore<10>{
    template <class FReal>
    static std::array<FReal,10> Build(){
        return { {FReal(-0.987688340595138),
                  FReal(-0.891006524188368),
                  FReal(-0.707106781186548),
                  FReal(-0.453990499739547),
                  FReal(-0.156434465040231),
                  FReal( 0.156434465040231),
                  FReal( 0.453990499739547),
                  FReal( 0.707106781186547),
                  FReal( 0.891006524188368),
                  FReal( 0.987688340595138)} };
    }
};

// order 11
template<>
struct FChebRootsCore<11>{
    template <class FReal>
    static std::array<FReal,11> Build(){
        return { {FReal(-9.89821441880933e-01),
                  FReal(-9.09631995354518e-01),
                  FReal(-7.55749574354258e-01),
                  FReal(-5.40640817455598e-01),
                  FReal(-2.81732556841430e-01),
                  FReal( 0.0                 ),
                  FReal( 2.81732556841430e-01),
                  FReal( 5.40640817455597e-01),
                  FReal( 7.55749574354258e-01),
                  FReal( 9.09631995354518e-01),
                  FReal( 9.89821441880933e-01)} };
    }
};

// order 12
template<>
struct FChebRootsCore<12>{
    template <class FReal>
    static std::array<FReal,12> Build(){
            return { {FReal(-0.991444861373810),
                      FReal(-0.923879532511287),
                      FReal(-0.793353340291235),
                      FReal(-0.608761429008721),
                      FReal(-0.382683432365090),
                      FReal(-0.130526192220052),
                      FReal( 0.130526192220052),
                      FReal( 0.382683432365090),
                      FReal( 0.608761429008721),
                      FReal( 0.793353340291235),
                      FReal( 0.923879532511287),
                      FReal( 0.991444861373810)} };
    }
};


// order 13
template<>
struct FChebRootsCore<13>{
    template <class FReal>
    static std::array<FReal,13> Build(){
        return { {FReal(-9.92708874098054e-01),
                  FReal(-9.35016242685415e-01),
                  FReal(-8.22983865893656e-01),
                  FReal(-6.63122658240795e-01),
                  FReal(-4.64723172043769e-01),
                  FReal(-2.39315664287558e-01),
                  FReal( 0.0                 ),
                  FReal( 2.39315664287557e-01),
                  FReal( 4.64723172043769e-01),
                  FReal( 6.63122658240795e-01),
                  FReal( 8.22983865893656e-01),
                  FReal( 9.35016242685415e-01),
                  FReal( 9.92708874098054e-01)} };
    }
};

// order 14
template<>
struct FChebRootsCore<14>{
    template <class FReal>
    static std::array<FReal,14> Build(){
        return { {FReal(-0.99371220989324258353),
                  FReal(-0.94388333030836756290),
                  FReal(-0.84672419922828416835),
                  FReal(-0.70710678118654752440),
                  FReal(-0.53203207651533656356),
                  FReal(-0.33027906195516708177),
                  FReal(-0.11196447610330785847),
                  FReal( 0.11196447610330785847),
                  FReal( 0.33027906195516708177),
                  FReal( 0.53203207651533656356),
                  FReal( 0.70710678118654752440),
                  FReal( 0.84672419922828416835),
                  FReal( 0.94388333030836756290),
                  FReal( 0.99371220989324258353)} };
    }
};


#endif
