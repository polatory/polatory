// ===================================================================================
// Copyright ScalFmm 2016 INRIA, Olivier Coulaud, Bérenger Bramas,
// Matthias Messner olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the
// FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
// An extension to the license is given to allow static linking of scalfmm
// inside a proprietary application (no matter its license).
// See the main license file for more details.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================
#ifndef FGENERATEDISTRIBUTION_HPP
#define FGENERATEDISTRIBUTION_HPP

/**
 * \file
 * \brief Distribution generation implementations
 * \author O. Coulaud
 */


#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>
//
#include "Utils/FMath.hpp"
#include "Utils/FParameters.hpp"

/**
 * \brief Seed the random number generator using current time
 */
void initRandom() {
    srand48(static_cast<long int>(time(nullptr)));
}

/**
 * \brief Generate a random number
 * \tparam FReal Floating point type
 * \return A random number in [0,1]
 */
template <class FReal>
FReal getRandom() {
    return static_cast<FReal>(drand48());
}

/**
 * \brief Generate points uniformly inside a cuboid
 *
 * \tparam FReal Floating point type
 *
 * \param N the number of points uniformly randomly sample on the unit cube
 * \param Lx the the X-length of the cuboid
 * \param Ly the the Y-length of the cuboid
 * \param Lz the the Z-length of the cuboid
 * \param points array of size 4*N and stores data as follow x,y,z,0,x,y,z,0...
 */
template <class FReal>
void unifRandomPointsInCube(const FSize N, const FReal& Lx, const FReal& Ly,
                            const FReal& Lz, FReal* points)
{
    initRandom();
    for(FSize i = 0, j = 0 ; i< N ; ++i, j+=4)  {
        points[j]   = getRandom<FReal>() * Lx;
        points[j+1] = getRandom<FReal>() * Ly;
        points[j+2] = getRandom<FReal>() * Lz;
    }
}

/**
 * \brief Generate points uniformly inside a ball
 *
 * \tparam FReal Floating point type
 *
 * \param R the ball radius
 * \param points array of size 4*N and stores data as follow x,y,z,0,x,y,z,0...
 */
template<class FReal>
void unifRandomPointsInBall(const FSize N, const FReal R, FReal* points) {
    initRandom();

    auto is_in_sphere = [&R](FReal* p) {
        return p[0]*p[0] + p[1]*p[1] + p[2]*p[2] < R*R;
    };

    for(FSize i = 0, j = 0 ; i< N ; ++i, j+=4)  {
        do {
            points[j]   = (getRandom<FReal>() - 0.5) * 2 * R;
            points[j+1] = (getRandom<FReal>() - 0.5) * 2 * R;
            points[j+2] = (getRandom<FReal>() - 0.5) * 2 * R;
        } while(! is_in_sphere(points + j));
    }
}

/**
 * \brief Generate N points non uniformly distributed on the ellipsoid of aspect ratio a:b:c
 *
 * \tparam FReal Floating point type
 *
 * \param N the number of points
 * \param a  the x semi-axe length
 * \param b  the y semi-axe length
 * \param c  the z semi-axe length
 * \param points array of size 4*N and stores data as follow x,y,z,0,x,y,z,0....
 */
template <class FReal>
void nonunifRandomPointsOnElipsoid(const FSize N, const FReal& a, const FReal& b,
                                   const FReal& c, FReal* points)
{
    FReal u, v, cosu;
    for (FSize i = 0, j = 0 ; i< N ; ++i, j+=4)  {
        u = FMath::FPi<FReal>() * getRandom<FReal>() - FMath::FPiDiv2<FReal>();
        v = FMath::FTwoPi<FReal>() * getRandom<FReal>() - FMath::FPi<FReal>();
        cosu = FMath::Cos(u);
        points[j]   = a * cosu * FMath::Cos(v);
        points[j+1] = b * cosu * FMath::Sin(v);
        points[j+2] = c * FMath::Sin(u);
    }
}


/**
 * \brief Generate N points uniformly distributed on the ellipsoid of aspect ratio a:a:c
 *
 * \tparam FReal Floating point type
 *
 * \param N the number of points
 * \param a  the x  semi-axe length
 * \param c  the z  semi-axe length
 * \param points array of size 4*N and stores data as follow x,y,z,0,x,y,z,0....
*/
template <class FReal>
void unifRandomPointsOnProlate(const FSize N, const FReal& a, const FReal& c,
                               FReal* points)
{
    FReal u, w, v, ksi;
    FReal e = (a*a*a*a)/(c*c*c*c);
    bool isgood = false;
    FSize cpt = 0;

    for (FSize i = 0, j = 0 ; i< N ; ++i, j+=4)  {
        // Select a random point on the prolate
        do {
            ++cpt;
            u = 2.0 * getRandom<FReal>() - 1.0;
            v = FMath::FTwoPi<FReal>() * getRandom<FReal>();
            w = FMath::Sqrt(1 - u*u);
            points[j]	= a * w * FMath::Cos(v);
            points[j+1] = a * w * FMath::Sin(v);
            points[j+2] = c * u;
            // Accept the position ?
            ksi = a * getRandom<FReal>();
            isgood = (points[j]*points[j]
                      + points[j+1]*points[j+1]
                      + e*points[j+2]*points[j+2]) < ksi*ksi;
        } while(isgood);
    }
    std::cout.precision(4);
    std::cout << "Total tested points: " << cpt
              << " % of rejected points: "
              << 100 * static_cast<FReal>(cpt-N) / static_cast<FReal>(cpt) << " %"
              << std::endl;
}


/**
 * \brief  Generate N points uniformly distributed on the hyperbolic paraboloid of  aspect ratio a:b:c
 *
 * \tparam FReal Floating point type
 *
 * \param N the number of points
 * \param a  the x  semi-axe length
 * \param b  the y  semi-axe length
 * \param c  the z  semi-axe length
 * \param points array of size 4*N and stores data as follow x,y,z,0,x,y,z,0...
 */
template <class FReal>
void unifRandomPointsOnHyperPara(const FSize N, const FReal &a, const FReal &b,
                                 const FReal &c, FReal * points)
{
    FReal u, v;
    for (FSize i = 0, j = 0 ; i< N ; ++i, j+=4)  {
        u = 2.0 * getRandom<FReal>() - 1.0;
        v = 2.0 * getRandom<FReal>() - 1.0;
        points[j]   = a * u;
        points[j+1] = b * v;
        points[j+2] = c * (u*u - v*v);
    }
};


/**
 * \brief Generate N points uniformly distributed on the sphere of radius R
 *
 * \tparam FReal Floating point type
 *
 * \param N the number of points uniformly randomly sample on the sphere
 * \param R the radius of the sphere
 * \param points array of size 4*N and stores data as follow x,y,z,0,x,y,z,0...
 */
template <class FReal>
void unifRandomPointsOnSphere(const FSize N, const FReal R, FReal* points) {
    initRandom();
    FReal u, v, theta, phi, sinPhi;
    for (FSize i = 0, j = 0 ; i< N ; ++i, j+=4)  {
        u = getRandom<FReal>();
        v = getRandom<FReal>();
        theta  = FMath::FTwoPi<FReal>() * u;
        phi    = FMath::ACos(2*v - 1);
        sinPhi = FMath::Sin(phi);

        points[j]   = FMath::Cos(theta) * sinPhi * R;
        points[j+1] = FMath::Sin(theta) * sinPhi * R;
        points[j+2] = (2*v - 1) * R;
    }
};


/**
 * \brief Radial Plummer distribution
 *
 * \tparam FReal Floating point type
 *
 * \param cpt counter to know how many random selections we need to obtain a radius less than R
 * \param R   radius of the sphere that contains the particles
 * \return The radius according to the Plummer distribution
 */
template <class FReal>
FReal plummerDist(FSize& cpt, const FReal &R) {
    FReal radius, u;
    while(true) {
        u = FMath::pow(getRandom<FReal>(), 2.0/3.0);
        radius = FMath::Sqrt(u/(1.0-u));
        ++cpt;
        if(radius <= R) {
            return static_cast<FReal>(radius);
        }
    }
}


/**
 * \brief Build N points following the Plummer distribution
 *
 * First we construct N points uniformly distributed on the unit sphere. Then
 * the radius in construct according to the Plummer distribution for
 * a  constant mass of 1/N
 *
 * \tparam FReal Floating point type
 *
 * \param N the number of points following the Plummer distribution
 * \param R the radius of the sphere that contains all the points
 * \param points array of size 4*N and stores data as follow x,y,z,0,x,y,z,0....
 */
template <class FReal>
void unifRandomPlummer(const FSize N, const FReal R, FReal * points) {
  //Pb of large box
    constexpr const FReal rand_max = 0.8;
    const FReal r_max = std::sqrt(1.0/(std::pow(rand_max, -2.0/3.0) - 1.0));
    //
    unifRandomPointsOnSphere<FReal>(N, 1, points);
    FReal mc = 1.0/static_cast<FReal>(N);
    for (FSize i = 0, j = 0 ; i< N ; ++i, j+=4)  {
    	FReal m = getRandom<FReal>();
	//    	FReal r = FMath::Sqrt( 1.0/(FMath::pow(m, -2.0/3.0) - 1.0)) ;
	FReal r = FMath::Sqrt(1.0/(FMath::pow(m, -2.0/3.0) - 1.0)) / r_max * R;
	points[j]    *= r;
        points[j+1]  *= r;
        points[j+2]  *= r;
        points[j+3]   = mc;  // the mass
    }
}
// template <class FReal>
// void unifRandomPlummer(const FSize N, const FReal R, FReal * points) {
//     constexpr const FReal rand_max = 0.8;
//     constexpr const FReal r_max = std::sqrt(1.0/(std::pow(rand_max, -2.0/3.0) - 1.0));

//     unifRandomPointsOnSphere<FReal>(N, 1, points);
//     FReal mc = 1.0/static_cast<FReal>(N);
//     for (FSize i = 0, j = 0 ; i< N ; ++i, j+=4)  {
//         FReal m = getRandom<FReal>();
//         while(m > rand_max) {
//             m = getRandom<FReal>();
//         }
//         FReal r = FMath::Sqrt(1.0/(FMath::pow(m, -2.0/3.0) - 1.0)) / r_max * R;
//         points[j]    *= r;
//         points[j+1]  *= r;
//         points[j+2]  *= r;
//         points[j+3]   = mc;  // the mass
//     }
// }
/**
 * \brief Build N points following the Plummer like distribution
 *
 * First we construct N points uniformly distributed on the unit sphere. Then
 * the radius in construct according to the Plummer like distribution.
 *
 * \tparam FReal Floating point type
 *
 * \param N the number of points following the Plummer distribution
 * \param R the radius of the sphere that contains all the points
 * \param points array of size 4*N and stores data as follow x,y,z,0,x,y,z,0....
 */
template <class FReal>
void unifRandomPlummerLike(const FSize N, const FReal R, FReal * points) {
	FReal a = 1.0 ;
    unifRandomPointsOnSphere<FReal>(N, 1, points);
    FReal r;
    FSize cpt = 0;
    for (FSize i = 0, j = 0 ; i< N ; ++i, j+=4)  {
        r = plummerDist(cpt,R);
	//        rm = std::max(rm, r);
        points[j]    *= r;
        points[j+1]  *= r;
        points[j+2]  *= r;
    }

    std::cout << "Total tested points: " << cpt << " % of rejected points: "
              << 100 * static_cast<FReal>(cpt-N) / static_cast<FReal>(cpt)
              << " %"
              << std::endl;
}
//
#endif
