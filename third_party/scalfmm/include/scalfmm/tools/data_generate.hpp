// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tools/data_generate.hpp
// --------------------------------

#ifndef SCALFMM_TOOLS_DATA_GENERATE_HPP
#define SCALFMM_TOOLS_DATA_GENERATE_HPP

#include <array>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <numbers>
#include <random>
#include <vector>

namespace scalfmm::tools
{
    /**
     * @brief Generate points uniformly inside a cuboid [0,width[0]]x..x[0,width[d]]
     *
     *  the number of particle is data.size() / stride
     *
     * @tparam ContainerType the container of particle. method size should exist
     * @tparam ValueType Floating point type
     * @param dimension (int) the space dimension
     * @param stride the stride between two points in the container
     * @param data array of size stride*N and stores data as follow x,y,z,0-0,x,y,z,0-0...
     * @param width a vector of size dimension containig the length of the cuboid
     */
    template<class ContainerType, typename ValueType>
    auto uniform_points_in_cuboid(const int dimension, const int stride, ContainerType& data,
                                  const std::vector<ValueType> width) -> void
    {
        std::cout << " call uniform_points_in_cuboid " << std::endl;
        const auto seed{33};
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<> dist(0, 1);
        for(std::size_t i = 0; i < data.size(); i += stride)
        {
            for(int j = 0; j < dimension; ++j)
            {
                data[i + j] = dist(gen) * width[j];
            }
        }
    }

    /**
     * @brief Generate points uniformly inside a ball of radius R
     *
     * The Rejection Method is used to generate uniform points unsid a ball. The
     * method is independent of tehe dimension.
     *
     * @tparam ContainerType array for the data
     * @tparam ValueType Floating point type
     * @param dimension (int) the space dimension
     * @param stride the stride between two points in the container
     * @param data array of size stride*N and stores data as follow x,y,z,0-0,x,y,z,0-0...
     * @param R (ValueType) the ball radius
     */
    template<class ContainerType, typename ValueType>
    auto uniform_points_ball(const int dimension, const int stride, ContainerType& data, const ValueType R) -> void
    {
        std::cout << " call unifRandomPointsInBall with the R= " << R << std::endl;
        using point_type = typename ContainerType::value_type;
        const auto seed{33};
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<ValueType> dist(-R, R);

        auto is_in_sphere = [&R, &dimension](point_type* p)
        {
            ValueType norm = 0.0;
            for(int i = 0; i < dimension; ++i)
            {
                norm += p[i] * p[i];
            }
            return norm <= R * R;
        };

        for(std::size_t i = 0; i < data.size(); i += stride)
        {
            do
            {
                for(int k = 0; k < dimension; ++k)
                {
                    data[i + k] = dist(gen);
                }
            } while(!is_in_sphere(&(data[i])));
        }
    }

    /**
     * @brief Generate N points uniformly distributed on the cercle of radius R
     *
     * We use the polar method do biild the distribution. The number of points is data.size()/stride
     *
     * @tparam ContainerType array for the data
     * @tparam ValueType Floating point type
     * @param stride the stride between two points in the container
     * @param data array of size stride*N and stores data as follow x,y,z,0-0,x,y,z,0-0...
     * @param R the radius of the sphere
     */
    template<class ContainerType, typename ValueType>
    auto uniform_points_on_cercle(const int stride, ContainerType& data, const ValueType R) -> void
    {
        std::cout << " call unifRandomPointsOnSphere with the R= " << R << std::endl;

        const auto seed{33};
        std::mt19937_64 gen(seed);
        ValueType u{}, theta{}, twoPi{std::numbers::pi_v<ValueType> * 2.0};
        std::uniform_real_distribution<ValueType> dist(0.0, twoPi);
        for(std::size_t i = 0; i < data.size(); i += stride)
        {
            u = dist(gen);
            theta = twoPi * u;
            data[i] = std::cos(theta) * R;
            data[i + 1] = std::sin(theta) * R;
        }
    }

    /**
     * @brief Generate N points uniformly distributed on the sphere of radius R
     *
     * We use the polar method do biild the distribution. The number of points is data.size()/stride
     *
     * @tparam ContainerType array for the data
     * @tparam ValueType Floating point type
     * @param stride the stride between two points in the container
     * @param data array of size stride*N and stores data as follow x,y,z,0-0,x,y,z,0-0...
     * @param R the radius of the sphere
     */
    template<class ContainerType, typename ValueType>
    auto uniform_points_on_sphere(const int stride, ContainerType& data, const ValueType R) -> void
    {
        std::cout << " call unifRandomPointsOnSphere with the R= " << R << std::endl;

        const auto seed{33};
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<ValueType> dist(0.0, 1.0);
        ValueType u, v, theta, phi, sinPhi, twoPi = std::numbers::pi_v<ValueType> * 2.0;
        for(std::size_t i = 0; i < data.size(); i += stride)
        {
            u = dist(gen);
            v = dist(gen);
            theta = twoPi * u;
            phi = std::acos(2.0 * v - 1.0);
            sinPhi = std::sin(phi);

            data[i] = std::cos(theta) * sinPhi * R;
            data[i + 1] = std::sin(theta) * sinPhi * R;
            data[i + 2] = (2.0 * v - 1.0) * R;
        }
    }

    /**
     * @brief Generate N points uniformly distributed on the sphere of radius R
     *
     * @tparam ContainerType array for the data
     * @tparam ValueType Floating point type
     * @param dimension
     * @param stride the stride between two points in the container
     * @param data array of size stride*N and stores data as follow x,y,z,-1-0,x,y,z,0-0...
     * @param R the radius of the sphere
     */
    template<class ContainerType, typename ValueType>
    auto uniform_points_on_d_sphere(const int dimension, const int stride, ContainerType& data,
                                    const ValueType R) -> void
    {
        if(dimension == 2)
        {
            uniform_points_on_cercle(stride, data, R);
        }
        else if(dimension == 3)
        {
            uniform_points_on_sphere(stride, data, R);
        }
        else
        {
            std::cerr << "To generate a d-sphere the dimension should be 2 or 3" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    /**
     * @brief Generate N points uniformly distributed on the ellipsoid of aspect ratio a:a:c
     *
     * @tparam ContainerType array for the data
     * @tparam ValueType Floating point type
     * @param stride the stride between two points in the container
     * @param radius (a,c)  the x  semi-axe length and the z  semi-axe length
     * @param points of data of size stride*N and stores data as follow
     * points(dim), values(nb) ....
     * stride = dim + nb
     */
    template<class ContainerType, typename ValueType>
    auto uniform_points_on_prolate(const int& stride, std::array<ValueType, 2>& radius, ContainerType& points) -> void
    {
        // Number of particles
        const auto N = points.size() / stride;
        auto a = radius[0];
        auto c = radius[1];
        //
        ValueType u, w, v;
        ValueType e = (a * a * a * a) / (c * c * c * c);
        std::size_t cpt = 0;
        const int NN = 20;
        std::vector<int> bin(NN, 0);
        ValueType h = 2 * c / NN;
        ValueType twoPi = std::numbers::pi_v<ValueType> * 2.0;
        ValueType pi = std::numbers::pi_v<ValueType>;
        std::cout << " call unifRandomPointsOnProlate with the a= " << a << " and c= " << c << std::endl;
        //
        const auto seed{33};
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<ValueType> dist(0.0, 1.0);
        //
        bool isgood = false;
        for(std::size_t i = 0, j = 0; i < N; ++i, j += stride)
        {
            // Select a random point on the prolate

            do
            {
                ++cpt;
                u = 2.0 * dist(gen) - 1.0;
                v = twoPi * dist(gen);
                w = std::sqrt(1.0 - u * u);
                points[j] = a * w * std::cos(v);
                points[j + 1] = a * w * std::sin(v);
                points[j + 2] = c * u;
                // Accept the position ? if x*x +y*y +e *z*z > a^2 kxi ( see hen and
                // Glotzer)
                ValueType ksi = dist(gen) / a;
                isgood = (points[j] * points[j] + points[j + 1] * points[j + 1] + e * points[j + 2] * points[j + 2]) <
                         ksi * ksi;
            } while(isgood);
            unsigned int k1 = static_cast<unsigned int>((c + points[j + 2]) / h);
            //    std::cout <<  points[j+2] << " k " << k << " h " << h << "   " <<
            //    c+ points[j+2]/h  << std::endl;
            //      if (k < NN){
            bin[k1] += 1;
            //          }
            //      else {
            //              std::cout << "  ERROR ERROR  ERROR ERROR " <<std::endl;
            //          }
        }
        std::cout.precision(4);
        std::cout << "Total tested points: " << cpt
                  << " % of rejected points: " << 100 * static_cast<ValueType>(cpt - N) / static_cast<ValueType>(cpt)
                  << " %" << std::endl;
        std::cout << " h " << h << std::endl;
        //    std::cout << " [ " ;
        //    for ( int k = 0 ; k < bin.size()-1; ++k) {
        //            std::cout  << bin[k]<< " , " ;
        //        }
        //    std::cout  << bin[bin.size() -1]<< " ] "<<  std::endl;
        ValueType x1, x2, surf;
        // We approximate the arc of the ellipsoide by as straight line (Conical
        // Frustum) see http://mathworld.wolfram.com/ConicalFrustum.html
        std::cout << " z-density - bins: [ ";
        for(unsigned int k = 0; k < bin.size(); ++k)
        {
            x1 = -c + k * h;
            x2 = x1 + h;   // point position
            x1 = a * std::sqrt(1 - x1 * x1 / (c * c));
            x2 = a * std::sqrt(1 - x2 * x2 / (c * c));                          // xm = a*FMath::Sqrt(1 -
                                                                                // xm*xm/(c*c) ); // radius position
            surf = pi * (x1 + x2) * std::sqrt((x1 - x2) * (x1 - x2) + h * h);   // Conical Frustum
            //   std::cout  << "  (" <<bin[k]<< " , " << x1 <<", "<< x2 <<", "<<surf
            //   <<", " << bin[k]/surf << " ) ";
            std::cout << bin[k] / surf << "  ";
        }
        std::cout << " ] " << std::endl;
    }

    /**
     * @brief Generate N points uniformly distributed on the ellipsoid of aspect ratio a:b:c
     *
     * See Chen, T., & Glotzer, S. C. (2007). Simulation studies of a phenomenological
     *      model for elongated virus capsid formation. Physical Review. E,
     *      Statistical, Nonlinear, and Soft Matter Physics, 75, 1–25.
     *      http://www.biomedsearch.com/nih/Simulation-studies-phenomenological-model-elongated/17677070.html
     *
     * @tparam ContainerType array for the data
     * @tparam ValueType Floating point type
     * @param stride the stride between two points in the container
     * @param radius (a, b, c)  the x  semi-axe length, y and the z  semi-axe length
     * @param points of data of size stride*N and stores data as follow
     * points(dim), values(nb) ....
     * stride = dim + nb
     */
    template<class ContainerType, typename ValueType>
    auto uniform_points_on_ellipsoid(const int& stride, std::array<ValueType, 3>& radius, ContainerType& points) -> void
    {
        // Number of particles
        const auto N = points.size() / stride;
        auto a = radius[0];
        auto b = radius[1];
        auto c = radius[2];
        //
        ValueType u, w, v;
        ValueType e1 = (a * a * a * a) / (b * b * b * b);
        ValueType e2 = (a * a * a * a) / (c * c * c * c);
        std::size_t cpt = 0;
        const int NN = 20;
        std::vector<int> bin(NN, 0);
        ValueType h = 2 * c / NN;
        ValueType twoPi = std::numbers::pi_v<ValueType> * 2.0;
        ValueType pi = std::numbers::pi_v<ValueType>;
        std::cout << " call unifRandomPointsOnProlate with the a= " << a << " and c= " << c << std::endl;
        //
        const auto seed{33};
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<ValueType> dist(0.0, 1.0);
        //
        bool isgood = false;
        for(std::size_t i = 0, j = 0; i < N; ++i, j += stride)
        {
            // Select a random point on the prolate

            do
            {
                ++cpt;
                u = 2.0 * dist(gen) - 1.0;
                v = twoPi * dist(gen);
                w = std::sqrt(1.0 - u * u);
                points[j] = a * w * std::cos(v);
                points[j + 1] = b * w * std::sin(v);
                points[j + 2] = c * u;
                // Accept the position ? if x*x + e2*y*y +e2 *z*z > a^2 kxi ( see hen and
                // Glotzer)
                ValueType ksi = dist(gen) / a;
                isgood = (points[j] * points[j] + e1 * points[j + 1] * points[j + 1] +
                          e2 * points[j + 2] * points[j + 2]) < ksi * ksi;
            } while(isgood);
            unsigned int k1 = static_cast<unsigned int>((c + points[j + 2]) / h);
            //    std::cout <<  points[j+2] << " k " << k << " h " << h << "   " <<
            //    c+ points[j+2]/h  << std::endl;
            //      if (k < NN){
            bin[k1] += 1;
            //          }
            //      else {
            //              std::cout << "  ERROR ERROR  ERROR ERROR " <<std::endl;
            //          }
        }
        std::cout.precision(4);
        std::cout << "Total tested points: " << cpt
                  << " % of rejected points: " << 100 * static_cast<ValueType>(cpt - N) / static_cast<ValueType>(cpt)
                  << " %" << std::endl;
        std::cout << " h " << h << std::endl;
        //    std::cout << " [ " ;
        //    for ( int k = 0 ; k < bin.size()-1; ++k) {
        //            std::cout  << bin[k]<< " , " ;
        //        }
        //    std::cout  << bin[bin.size() -1]<< " ] "<<  std::endl;
        ValueType x1, x2, surf;
        // We approximate the arc of the ellipsoide by as straight line (Conical
        // Frustum) see http://mathworld.wolfram.com/ConicalFrustum.html
        std::cout << " z-density - bins: [ ";
        for(unsigned int k = 0; k < bin.size(); ++k)
        {
            x1 = -c + k * h;
            x2 = x1 + h;   // point position
            x1 = a * std::sqrt(1 - x1 * x1 / (c * c));
            x2 = a * std::sqrt(1 - x2 * x2 / (c * c));                          // xm = a*FMath::Sqrt(1 -
                                                                                // xm*xm/(c*c) ); // radius position
            surf = pi * (x1 + x2) * std::sqrt((x1 - x2) * (x1 - x2) + h * h);   // Conical Frustum
            //   std::cout  << "  (" <<bin[k]<< " , " << x1 <<", "<< x2 <<", "<<surf
            //   <<", " << bin[k]/surf << " ) ";
            std::cout << bin[k] / surf << "  ";
        }
        std::cout << " ] " << std::endl;
    }

    /**
     * @brief Generate N points non uniformly distributed on the ellipsoid of
     * aspect ratio a:a:c
     *
     *   f(x,y,z) = (x^2+y^2)/a^2 + z^2/c^2
     *
     * @tparam ContainerType array for the data
     * @tparam ValueType Floating point type
     * @param stride the stride between two points in the container
     * @param radius (a,b,c)  the x,y and  semi-axe length
     * @param density
     * @param points of data of size stride*N with N the number of points.
     */
    template<class ContainerType, class ValueType>
    auto nonuniform_point_on_prolate(const int stride, std::array<ValueType, 2>& radius, const double& density,
                                     ContainerType& points) -> void
    {
        ValueType twoPi = std::numbers::pi_v<ValueType> * 2.0;
        ValueType pi = std::numbers::pi_v<ValueType>;
        const auto N = points.size() / stride;
        auto a = radius[0];
        auto b = radius[1];
        //  auto c = radius[2];

        ValueType rotationMatrix[3][3];
        ValueType alpha = pi / 8.0;
        ValueType omega = pi / 4.0;

        ValueType yrotation[3][3];
        yrotation[0][0] = std::cos(alpha);
        yrotation[0][1] = 0.0;
        yrotation[0][2] = std::sin(alpha);
        yrotation[1][0] = 0.0;
        yrotation[1][1] = 1.0;
        yrotation[1][2] = 0.0;
        yrotation[2][0] = -std::sin(alpha);
        yrotation[2][1] = 0.0;
        yrotation[2][2] = std::cos(alpha);

        ValueType zrotation[3][3];
        zrotation[0][0] = std::cos(omega);
        zrotation[0][1] = -std::sin(omega);
        zrotation[0][2] = 0.0;
        zrotation[1][0] = std::sin(omega);
        zrotation[1][1] = std::cos(omega);
        zrotation[1][2] = 0.0;
        zrotation[2][0] = 0.0;
        zrotation[2][1] = 0.0;
        zrotation[2][2] = 1.0;

        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                ValueType sum = 0.0;
                for(int k = 0; k < 3; ++k)
                {
                    sum += zrotation[i][k] * yrotation[k][j];
                }
                rotationMatrix[i][j] = sum;
            }
        }
        const auto seed{33};
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<ValueType> dist(0.0, 1.0);

        const ValueType MaxDensity = density;
        std::cout << "MaxDensity: " << MaxDensity << std::endl;

        for(std::size_t i = 0, j = 0; i < N; ++i, j += stride)
        {
            const ValueType maxPerimeter = twoPi * a;

            ValueType px = 0;
            // rayon du cercle pour ce x
            ValueType subr = 0;
            ValueType coef = 1.0;

            //  the ellipsoid is generated by the rotation of an ellipse around one of its axes
            // px^2/a^2 + z^2/b^2 = 1
            do
            {
                // px   = ( ((getRandom()*8.0+getRandom())/9.0) * a * 2) - a;
                px = (dist(gen) * a * 2.0) - a;
                coef = std::abs(px) * MaxDensity / a + 1.0;
                subr = std::sqrt((1.0 - ((px * px) / (a * a))) * (b * b));
            } while((dist(gen) * maxPerimeter) > subr * coef);

            // on genere un angle for the rotation
            omega = dist(gen) * twoPi;
            // on recupere py et pz sur le cercle
            const ValueType py = std::cos(omega) * subr;
            const ValueType pz = std::sin(omega) * subr;
            //std::cout  << j << "  " << px*px +py*py + pz*pz
            // inParticle.setPosition(px,py,pz);
            points[j] = px * rotationMatrix[0][0] + py * rotationMatrix[0][1] + pz * rotationMatrix[0][2];
            points[j + 1] = px * rotationMatrix[1][0] + py * rotationMatrix[1][1] + pz * rotationMatrix[1][2];
            points[j + 2] = px * rotationMatrix[2][0] + py * rotationMatrix[2][1] + pz * rotationMatrix[2][2];
        }
    }

    /**
     * @brief Build N points following the Plummer like distribution
     *
     * First we construct N points uniformly distributed on the unit sphere.
     * Then the radius in construct according to the Plummer like distribution.
     *
     * @tparam ContainerType type of container containig the points (typically std::vector)
     * @tparam ValueType Floating point type
     * @param stride the number of values between two point positions
     * @param radius_max the maximal radius of the sphere that contains all the points
     * @param points array of size stride*N and stores data as follow
     *         x,y,z,0-0,x,y,z,0-0....... The size of the array is N*stride
     */
    template<class ContainerType, class ValueType>
    auto plummer_distrib(const int stride, const ValueType radius_max, ContainerType& points) -> void
    {
        ValueType r = 1.0;
        uniform_points_on_sphere(stride, points, r);

        const auto seed{44};
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<ValueType> dist(0.0, 1.0);

        std::size_t cpt = 0;
        for(std::size_t j = 0; j < points.size(); j += stride)
        {
            do
            {
                //
                r = dist(gen);
                ValueType u = std::pow(r, 2.0 / 3.0);
                r = std::sqrt(u / (1.0 - u));
            } while(r >= radius_max);

            //        rm = std::max(rm, r);
            points[j] *= r;
            points[j + 1] *= r;
            points[j + 2] *= r;
        }
        auto N = points.size() / stride;
        std::cout << "Total tested points: " << cpt
                  << " % of rejected points: " << 100 * static_cast<ValueType>(cpt - N) / static_cast<ValueType>(cpt)
                  << " %" << std::endl;
    }

    /**
    * @brief generate_input_values generate random input values for particles
     *
     * @tparam ContainerType
     * @tparam ValueType
     * @param data array contains the positions of the particles and the input values
     * @param nb_input_values number of physical values to generate for each particle
     * @param stride  the stride between two particle position = dim + nb_input_values
     * @param interval
     * @param zeromean boolean to center each physical values
     */
    template<class ContainerType, typename ValueType>
    auto generate_input_values(ContainerType& data, const int nb_input_values, const int stride,
                               std::vector<std::array<ValueType, 2>>& interval, const bool zeromean) -> void
    {
        // using ValueType = typename ContainerType::value_type;
        const auto seed{33};
        std::mt19937_64 gen(seed);

        ValueType* mean = new ValueType[nb_input_values]{};
        std::uniform_real_distribution<ValueType> dist(0.0, 1.0);
        const int start = stride - nb_input_values;
        std::size_t pos = 0;
        for(std::size_t i = 0; i < data.size(); i += stride)
        {
            pos = i + start;
            for(int j = 0; j < nb_input_values; ++j)
            {
                data[pos + j] = (interval[j][1] - interval[j][0]) * dist(gen) + interval[j][0];
                mean[j] += data[pos + j];
            }
        }
        const auto nb_particles = data.size() / stride;
        double cor = 1.0 / static_cast<ValueType>(nb_particles);
        if(zeromean)
        {
            for(int j = 0; j < nb_input_values; ++j)
            {
                mean[j] /= static_cast<ValueType>(nb_particles);
                std::cout << " Mean for variables " << j << " is " << mean[j] << std::endl;
            }
            for(std::size_t i = 0; i < data.size(); i += stride)
            {
                pos = i + start;
                for(int j = 0; j < nb_input_values; ++j)
                {
                    data[pos + j] -= mean[j];
                }
            }
            cor = 0.0;
        }
        // Means
        for(int j = 0; j < nb_input_values; ++j)
        {
            std::cout << " Mean for variables " << j << " is " << mean[j] * cor << std::endl;
        }
    }

}   // namespace scalfmm::tools

#endif
