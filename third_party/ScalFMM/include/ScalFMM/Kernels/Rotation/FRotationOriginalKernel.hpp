// See LICENCE file at project root
#ifndef FROTATIONORIGINALKERNEL_HPP
#define FROTATIONORIGINALKERNEL_HPP

#include "../../Components/FAbstractKernels.hpp"
#include "../../Utils/FSmartPointer.hpp"
#include "../../Utils/FComplex.hpp"
#include "../../Utils/FMemUtils.hpp"
#include "../../Utils/FSpherical.hpp"

#include "../P2P/FP2PR.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FRotationOriginalKernel
* @brief
*
* This kernels is a complete rotation based kernel with spherical
* harmonic.
*
* This kernel is still not optimized for a pedagogic purpose.
* Please with refer to the optimized version to run real simulation.
*
* Here there is no real precomputation no optimization.
*
* The fallowing code has to be insert inside this class to print
* the details of a vector during the computation if needed.
* @code
* for(int l = 0 ; l <= P ; ++l ){
*    for(int m = 0 ; m <= l ; ++m ){
*        printf("3 - [%d][%d] %f i%f\t", l, m, target_u[atLm(l,m)].getReal(), target_u[atLm(l,m)].getImag());
*    }
*    printf("\n");
* }
* @endcode
*/
template< class FReal, class CellClass, class ContainerClass, int P>
class FRotationOriginalKernel : public FAbstractKernels<CellClass,ContainerClass> {
    //< Size of the data array computed using a suite relation
    static const int SizeArray = ((P+2)*(P+1))/2;

    ///////////////////////////////////////////////////////
    // Object attributes
    ///////////////////////////////////////////////////////

    const FReal boxWidth;               //< the box width at leaf level
    const int   treeHeight;             //< The height of the tree
    const FReal widthAtLeafLevel;       //< width of box at leaf level
    const FReal widthAtLeafLevelDiv2;   //< width of box at leaf leve div 2
    const FPoint<FReal> boxCorner;             //< position of the box corner


    FSmartPointer<FSpherical<FReal>> childrenPosition;     //< The transfers between level
    FSmartPointer<FSpherical<FReal>> interactionsPosition; //< The transfers at a same level

    ///////////////////////////////////////////////////////
    // d_lmk
    ///////////////////////////////////////////////////////

    /** Return position in the array of the l/m couple */
    int atLm(const int l, const int m){
        // summation series over l + m => (l*(l+1))/2 + m
        return ((l*(l+1))>>1) + m;
    }

    /** Return the sign of a value */
    template < class T >
    T Sgn(const T a){
        if(a < 0) return T(-1);
        else if(a > 0) return T(1);
        return T(0);
    }

    /** Return the factorial of a number */
    FReal fact(const int a){
        if(a<0) printf("Error factorial negative!! a=%d\n",a);
        FReal result = 1;
        for(int i = 1 ; i <= a ; ++i){
            result *= FReal(i);
        }
        return result;
    }

    /** Return the combine of a paire of number */    
    FReal combin(const int& a, const int& b){
        if(a-b<0) printf("Error combi negative!! a=%d b=%d\n",a,b);
        return fact(a) / (fact(b)*fact(a-b));
    }

    /** To access the rotation matrix value with an analytical computation on the fly
      * These analytical formula has been taken from two papers:
      * Fast and accurate determination of the wigner rotation matrices in the fast multipole method
      * Formulas 13, 14, 15, 16
      * Parallelization of the Fast Multipole Method
      * Formula 19
      *  \f[
      * P^{l}_{m,k} = \frac{1}{2} \sqrt{ \frac{(l-m)!(l+m)!}{(l-k)!(l+k)!} } (1+sgn(k)cos( \theta))^{|k|} sin( \theta )^{m-|k|}
      *              \sum_{n=max(-(m+k),0)}^{min(l-m,l-k)}{ (-1)^{l-m-n} {{l-k} \choose {n}} {{l+k} \choose {l-m-n}}
      *              (1+cos( \theta ))^n (1+cos( \theta ))^{l-m-n} } , l \geq 0, -l \leq k \leq l, |k| \leq m \leq l
      * P^{l}_{m,k} = (-1)^{m+k} P^{l}_{k,m} , l \ge 0, -l \leq m \le 0, |m| \leq k \leq l
      * P^{l}_{m,k} = (-1)^{m+k} P^{l}_{k,m} , l \ge 0, 0 \leq m \le l, m \le k \leq l
      * \f]
      */
    FReal d_lmk_analytical(const FReal cosTheta, const FReal sinTheta, const int l , const int m, const int k){
        if( l >= 0 && -l <= k && k <= l && FMath::Abs(k) <= m && m <= l ){
            FReal sum = 0;
            for(int n = FMath::Max(-(m+k),0) ; n <= FMath::Min(l-m,l-k) ; ++n){
	      sum += FMath::pow(FReal(-1.0), FMath::Abs(l-m-n)) * combin(l-k, n) * combin(l+k, l-m-n) * FMath::pow(FReal(1.0)+cosTheta,FMath::Abs(n)) * FMath::pow(FReal(1.0)-cosTheta, FMath::Abs(l-m-n));
            }
            return (FReal(1.0)/FMath::pow(FReal(2.0),FMath::Abs(l))) * FMath::Sqrt((fact(l-m)*fact(l+m))/(fact(l-k)*fact(l+k)))
	      * FMath::pow(FReal(1.0) + FReal(Sgn(k))*cosTheta, FMath::Abs(k)) * FMath::pow(sinTheta, FMath::Abs(m - FMath::Abs(k))) * sum;
        }
        // P^{l}_{m,k} = (-1)^{m+k} P^{l}_{k,m} , l \ge 0, -l \leq m \le 0, |m| \leq k \leq l
        else if( (l > 0 && -l <= m && m < 0 && FMath::Abs(m) <= k && k <= l)
                 ||  (l > 0 && 0 <= m && m < l && m < k && k <= l )) {
	  return FMath::pow(FReal(-1.0), FMath::Abs(m+k)) * d_lmk_analytical(cosTheta,sinTheta, l, k ,m);
        }
        // P^{l}_{m,k} = (-1)^{m+k} P^{l}_{k,m} , l \ge 0, 0 \leq m \le l, m \le k \leq l
        else if( l > 0 && -l <= m && m < l && -l <= k && k < -m ){
	  return FMath::pow(FReal(-1.0), FMath::Abs(m+k)) * d_lmk_analytical(cosTheta,sinTheta, l, -m, -k);
        }
        else{
            printf("Error that should not be possible!\n");
            return FReal(0.0);
        }
    }

    ///////////////////////////////////////////////////////
    // legendre
    ///////////////////////////////////////////////////////

    /** Compute the legendre polynomial from {0,0} to {P,P}
      * the computation is made by recurence (P cannot be equal to 0)
      * @todo use pointer
      *
      * The formula has been taken from:
      * Fast and accurate determination of the wigner rotation matrices in the fast multipole method
      * Formula number (22)
      * \f[
      * P_{0,0} = 1
      * P_{l,l} = (2l-1) sin( \theta ) P_{l-1,l-1} ,l \ge 0
      * P_{l,l-1} = (2l-1) cos( \theta ) P_{l-1,l-1} ,l \ge 0
      * P_{l,m} = \frac{(2l-1) cos( \theta ) P_{l-1,m} - (l+m-1) P_{l-2,m}x}{(l-k)} ,l \ge 1, 0 \leq m \le l-1
      * \f]
      */
    void computeLegendre(FReal legendre[], const FReal inCosTheta, const FReal inSinTheta){
        const FReal invSinTheta = -inSinTheta;
        legendre[atLm(0,0)] = 1.0;             // P_0,0(1) = 1

        legendre[atLm(1,0)] = inCosTheta;      // P_1,0 = cos(theta)
        legendre[atLm(1,1)] = invSinTheta;     // P_1,1 = -sin(theta)

        // Compute using recurrence
        for(int l = 2; l <= P ; ++l ){
            for( int m = 0; m < l - 1 ; ++m ){
                // P_{l,m} = \frac{(2l-1) cos( \theta ) P_{l-1,m} - (l+m-1) P_{l-2,m}x}{(l-k)}
                legendre[atLm(l,m)] = (FReal(2*l-1) * inCosTheta * legendre[atLm(l-1,m)] - FReal( l + m - 1 ) * legendre[atLm(l-2,m)] )
                                                                    / FReal( l - m );
            }
            // P_{l,l-1} = (2l-1) cos( \theta ) P_{l-1,l-1}
            legendre[atLm(l,l-1)] = FReal(2*l-1) * inCosTheta * legendre[atLm(l-1,l-1)];
            // P_{l,l} = (2l-1) sin( \theta ) P_{l-1,l-1}
            legendre[atLm(l,l)] = FReal(2*l-1) * invSinTheta * legendre[atLm(l-1,l-1)];
        }
    }

    ///////////////////////////////////////////////////////
    // Real rotation
    ///////////////////////////////////////////////////////

    /** This function rotate a multipole vector by an angle azimuth phi
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1
      * \f[
      * O_{l,m}( \alpha, \beta + \phi ) = e^{-i \phi m} O_{l,m}( \alpha, \beta )
      * \f]
      * The computation is simply a multiplication per a complex number \f$ e^{-i \phi m} \f$
      * Phi should be in [0,2pi]
      */
    void rotateMultipoleAroundZ(FComplex<FReal> vec[], const FReal phi){
        FComplex<FReal> cell_rotate[SizeArray];
        for(int l = 0 ; l <= P ; ++l){
            for(int m = 0 ; m <= l ; ++m ){
                // O_{l,m}( \alpha, \beta + \phi ) = e^{-i \phi m} O_{l,m}( \alpha, \beta )
                const FComplex<FReal> exp_minus_imphi(FMath::Cos(-phi * FReal(m)), FMath::Sin(-phi * FReal(m)));
                cell_rotate[atLm(l,m)].equalMul(exp_minus_imphi , vec[atLm(l,m)]);
            }
        }
        FMemUtils::copyall(vec,cell_rotate,SizeArray);
    }

    /** This function rotate a local vector by an angle azimuth phi
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1
      * \f[
      * M_{l,m}( \alpha, \beta + \phi ) = e^{i \phi m} M_{l,m}( \alpha, \beta )
      * \f]
      * The computation is simply a multiplication per a complex number \f$ e^{i \phi m} \f$
      * Phi should be in [0,2pi]
      */
    void rotateTaylorAroundZ(FComplex<FReal> vec[], const FReal phi){
        FComplex<FReal> cell_rotate[SizeArray];
        for(int l = 0 ; l <= P ; ++l){
            for(int m = 0 ; m <= l ; ++m ){
                // M_{l,m}( \alpha, \beta + \phi ) = e^{i \phi m} M_{l,m}( \alpha, \beta )
                const FComplex<FReal> exp_imphi(FMath::Cos(phi * FReal(m)), FMath::Sin(phi * FReal(m)));
                cell_rotate[atLm(l,m)].equalMul(exp_imphi , vec[atLm(l,m)]);
            }
        }
        FMemUtils::copyall(vec,cell_rotate,SizeArray);
    }

    /** This function rotate a multipole vector by an angle inclination \theta
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1
      * \f[
      * O_{l,m}( \alpha + \theta, \beta ) = \sum_{k=-l}^l{ \sqrt{ \frac{(l-k)!(l+k)!}{(l-|m|)!(l+|m|)!} }
      *                                     d^l_{km}( \theta ) O_{l,k}( \alpha, \beta ) }
      * \f]
      * Because we store only P_lm for l >= 0 and m >= 0 we use the relation of symetrie as:
      * \f$ O_{l,-m} = \bar{ O_{l,m} } (-1)^m \f$
      * Theta should be in [0,pi]
      */
    void rotateMultipoleAroundY(FComplex<FReal> vec[], const FReal theta){
        FComplex<FReal> cell_rotate[SizeArray];
        for(int l = 0 ; l <= P ; ++l){
            for(int m = 0 ; m <= l ; ++m ){
                FReal w_lkm_real = 0.0;
                FReal w_lkm_imag = 0.0;

                for(int k = -l ; k < 0 ; ++k){
                    // O_{l,-m} = \bar{ O_{l,m} } (-1)^m
                    const FReal d_lmk = d_lmk_analytical(FMath::Cos(theta),FMath::Sin(theta),l,m,k);
                    // \sqrt{ \frac{(l-k)!(l+k)!}{(l-|m|)!(l+|m|)!} }
                    const FReal factor = FMath::Sqrt((fact(l-k)*fact(l+k))/(fact(l-abs(m))*fact(l+abs(m))));
                    w_lkm_real += FMath::pow(FReal(-1.0),-k) * factor * d_lmk * vec[atLm(l,-k)].getReal(); // k<0 => Conjugate * -1^k
                    w_lkm_imag -= FMath::pow(FReal(-1.0),-k) * factor * d_lmk * vec[atLm(l,-k)].getImag(); // k<0 => Conjugate * -1^k
                }
                for(int k = 0 ; k <= l ; ++k){
                    const FReal d_lmk = d_lmk_analytical(FMath::Cos(theta),FMath::Sin(theta),l,m,k);
                    // \sqrt{ \frac{(l-k)!(l+k)!}{(l-|m|)!(l+|m|)!} }
                    const FReal factor = FMath::Sqrt((fact(l-k)*fact(l+k))/(fact(l-abs(m))*fact(l+abs(m))));
                    w_lkm_real += factor * d_lmk * vec[atLm(l,k)].getReal();
                    w_lkm_imag += factor * d_lmk * vec[atLm(l,k)].getImag();
                }
                cell_rotate[atLm(l,m)].setRealImag(w_lkm_real, w_lkm_imag);
            }
        }
        FMemUtils::copyall(vec,cell_rotate,SizeArray);
    }

    /** This function rotate a local vector by an angle inclination \theta
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1
      * \f[
      * M_{l,m}( \alpha + \theta, \beta ) = \sum_{k=-l}^l{ \sqrt{ \frac{(l-|m|)!(l+|m|)!}{(l-k)!(l+k)!} }
      *                                     d^l_{km}( \theta ) M_{l,k}( \alpha, \beta ) }
      * \f]
      * Because we store only P_lm for l >= 0 and m >= 0 we use the relation of symetrie as:
      * \f$ M_{l,-m} = \bar{ M_{l,m} } (-1)^m \f$
      * Theta should be in [0,pi]
      */
    void rotateTaylorAroundY(FComplex<FReal> vec[], const FReal theta){
        FComplex<FReal> cell_rotate[SizeArray];
        for(int l = 0 ; l <= P ; ++l){
            for(int m = 0 ; m <= l ; ++m ){
                FReal w_lkm_real = 0.0;
                FReal w_lkm_imag = 0.0;

                for(int k = -l ; k < 0 ; ++k){
                    // M_{l,-m} = \bar{ M_{l,m} } (-1)^m
                    const FReal d_lmk = d_lmk_analytical(FMath::Cos(theta),FMath::Sin(theta),l,m,k);
                    // \sqrt{ \frac{(l-|m|)!(l+|m|)!}{(l-k)!(l+k)!} }
                    const FReal factor = FMath::Sqrt((fact(l-abs(m))*fact(l+abs(m)))/(fact(l-k)*fact(l+k)));
                    w_lkm_real += FMath::pow(FReal(-1.0),-k) * factor * d_lmk * vec[atLm(l,-k)].getReal(); // k<0 => Conjugate * -1^k
                    w_lkm_imag -= FMath::pow(FReal(-1.0),-k) * factor * d_lmk * vec[atLm(l,-k)].getImag(); // k<0 => Conjugate * -1^k
                }
                for(int k = 0 ; k <= l ; ++k){
                    const FReal d_lmk = d_lmk_analytical(FMath::Cos(theta),FMath::Sin(theta),l,m,k);
                    // \sqrt{ \frac{(l-|m|)!(l+|m|)!}{(l-k)!(l+k)!} }
                    const FReal factor = FMath::Sqrt((fact(l-abs(m))*fact(l+abs(m)))/(fact(l-k)*fact(l+k)));
                    w_lkm_real += factor * d_lmk * vec[atLm(l,k)].getReal();
                    w_lkm_imag += factor * d_lmk * vec[atLm(l,k)].getImag();
                }
                cell_rotate[atLm(l,m)].setRealImag(w_lkm_real, w_lkm_imag);
            }
        }
        FMemUtils::copyall(vec,cell_rotate,SizeArray);
    }

    /** This function rotate a multipole vector by a angles inclination & azimuth
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1 as the forward rotation
      *
      * Rotation are not commutative so we have to do it in the right order
      */
    void rotateMultipole(FComplex<FReal> vec[], const FReal azimuth, const FReal inclination){
        rotateMultipoleAroundZ(vec,(FMath::FPiDiv2<FReal>() + azimuth));
        rotateMultipoleAroundY(vec,inclination);
    }
    /** This function rotate a multipole vector by a angles inclination & azimuth
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1 as the forward rotation
      *
      * Rotation are not commutative so we have to do it in the right order
      */
    void deRotateMultipole(FComplex<FReal> vec[], const FReal azimuth, const FReal inclination){
        rotateMultipoleAroundY(vec,-inclination);
        rotateMultipoleAroundZ(vec,-(FMath::FPiDiv2<FReal>() + azimuth));
    }

    /** This function rotate a local vector by a angles inclination & azimuth
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1 as the forward rotation
      *
      * Rotation are not commutative so we have to do it in the right order
      */
    void rotateTaylor(FComplex<FReal> vec[], const FReal azimuth, const FReal inclination){
        rotateTaylorAroundZ(vec,(FMath::FPiDiv2<FReal>() + azimuth));
        rotateTaylorAroundY(vec,inclination);
    }
    /** This function rotate a local vector by a angles inclination & azimuth
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1 as the forward rotation
      *
      * Rotation are not commutative so we have to do it in the right order
      */
    void deRotateTaylor(FComplex<FReal> vec[], const FReal azimuth, const FReal inclination){
        rotateTaylorAroundY(vec,-inclination);
        rotateTaylorAroundZ(vec,-(FMath::FPiDiv2<FReal>() + azimuth));
    }

    ///////////////////////////////////////////////////////
    // Position precomputation
    ///////////////////////////////////////////////////////

    /** This function precompute the position between cells */
    void preComputePosition(){
        // Compute the parent/children relation for M2M L2L
        childrenPosition = new FSpherical<FReal>[8 * (treeHeight-1)];
        {
            FReal subBoxWidth = widthAtLeafLevelDiv2;
            // For each
            for(int idxLevel = treeHeight-2 ; idxLevel > 0 ; --idxLevel){
                for(int idxChild = 0 ; idxChild < 8 ; ++idxChild ){
                    // coord from child to parent
                    const FReal x = FReal((idxChild&4)? -subBoxWidth : subBoxWidth);
                    const FReal y = FReal((idxChild&2)? -subBoxWidth : subBoxWidth);
                    const FReal z = FReal((idxChild&1)? -subBoxWidth : subBoxWidth);
                    const FPoint<FReal> relativePosition( x , y , z );

                    childrenPosition[(idxLevel-1)*8 + idxChild] = FSpherical<FReal>(relativePosition);
                }
                subBoxWidth *= FReal(2.0);
            }
        }
        // Compute the interaction relations for M2L
        interactionsPosition = new FSpherical<FReal>[343 * (treeHeight-1)];
        {
            FReal boxWidthAtLevel = widthAtLeafLevel;
            for(int idxLevel = treeHeight-1 ; idxLevel > 0 ; --idxLevel){
                for(int idxX = -3 ; idxX <= 3 ; ++idxX ){
                    for(int idxY = -3 ; idxY <= 3 ; ++idxY ){
                        for(int idxZ = -3 ; idxZ <= 3 ; ++idxZ ){
                            if( idxX != 0 || idxY != 0 || idxZ != 0 ){
                                const FPoint<FReal> relativePosition( -FReal(idxX)*boxWidthAtLevel,
                                                               -FReal(idxY)*boxWidthAtLevel,
                                                               -FReal(idxZ)*boxWidthAtLevel);
                                const int position = ((( (idxX+3) * 7) + (idxY+3))) * 7 + idxZ + 3;
                                interactionsPosition[(idxLevel-1)*343 + position] = FSpherical<FReal>(relativePosition);
                            }
                        }
                    }
                }
                boxWidthAtLevel *= FReal(2.0);
            }
        }
    }

    /** To get the right spherical object from level and child position */
    FSpherical<FReal> getSphericalChild(const int idxLevel, const int position) const {
        return childrenPosition[(idxLevel-1)*8 + position];
    }

    /** To get the right spherical object from level and interaction position */
    FSpherical<FReal> getSphericalInteraction(const int idxLevel, const int position) const {
        return interactionsPosition[(idxLevel-1)*343 + position];
    }

    /** Return the position of a leaf from its tree coordinate */
    FPoint<FReal> getLeafCenter(const FTreeCoordinate coordinate) const {
        return FPoint<FReal>(
                    FReal(coordinate.getX()) * widthAtLeafLevel + widthAtLeafLevelDiv2 + boxCorner.getX(),
                    FReal(coordinate.getY()) * widthAtLeafLevel + widthAtLeafLevelDiv2 + boxCorner.getY(),
                    FReal(coordinate.getZ()) * widthAtLeafLevel + widthAtLeafLevelDiv2 + boxCorner.getZ());
    }

public:

    /** Constructor, needs system information */
    FRotationOriginalKernel( const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter) :
        boxWidth(inBoxWidth),
        treeHeight(inTreeHeight),
        widthAtLeafLevel(inBoxWidth/FReal(1 << (inTreeHeight-1))),
        widthAtLeafLevelDiv2(widthAtLeafLevel/2),
        boxCorner(inBoxCenter.getX()-(inBoxWidth/2),inBoxCenter.getY()-(inBoxWidth/2),inBoxCenter.getZ()-(inBoxWidth/2))
        {
        // simply does the precomputation
        preComputePosition();
    }

    /** Default destructor */
    virtual ~FRotationOriginalKernel(){
    }

    /** P2M
      * The computation is based on the paper :
      * Parallelization of the fast multipole method
      * Formula number 10, page 3
      * \f[
      * \omega (q,a) = q \frac{a^{l}}{(l+|m|)!} P_{lm}(cos( \alpha ) )e^{-im \beta}
      * \f]
      */
    void P2M(CellClass* const inPole, const ContainerClass* const inParticles )override {
        const FReal i_pow_m[4] = {0, FMath::FPiDiv2<FReal>(), FMath::FPi<FReal>(), -FMath::FPiDiv2<FReal>()};
        // w is the multipole moment
        FComplex<FReal>* FRestrict const w = inPole->getMultipole();

        // Copying the position is faster than using cell position
        const FPoint<FReal> cellPosition = getLeafCenter(inPole->getCoordinate());

        // We need a legendre array
        FReal legendre[SizeArray];

        // For all particles in the leaf box
        const FReal*const physicalValues = inParticles->getPhysicalValues();
        const FReal*const positionsX = inParticles->getPositions()[0];
        const FReal*const positionsY = inParticles->getPositions()[1];
        const FReal*const positionsZ = inParticles->getPositions()[2];

        for(FSize idxPart = 0 ; idxPart < inParticles->getNbParticles() ; ++ idxPart){
            // P2M
            const FPoint<FReal> position(positionsX[idxPart],positionsY[idxPart],positionsZ[idxPart]);
            const FSpherical<FReal> sph(position - cellPosition);

            // The physical value (charge, mass)
            const FReal q = physicalValues[idxPart];
            // The distance between the SH and the particle
            const FReal a = sph.getR();

            // Compute the legendre polynomial
            computeLegendre(legendre, sph.getCosTheta(), sph.getSinTheta());

            // w{l,m}(q,a) = q a^l/(l+|m|)! P{l,m}(cos(alpha)) exp(-i m Beta)
            for(int l = 0 ; l <= P ; ++l ){
                for(int m = 0 ; m <= l ; ++m ){
                    const FReal magnitude = q * (FMath::pow( a , l )/fact(l+m))
                            * legendre[atLm(l,m)];
                    w[atLm(l,m)].incReal(magnitude * FMath::Cos(FReal(m) * sph.getPhi() + i_pow_m[m & 0x3]));
                    w[atLm(l,m)].incImag(magnitude * FMath::Sin(FReal(m) * sph.getPhi() + i_pow_m[m & 0x3]));
                }
            }
        }
    }

    /** M2M
      * The operator A has been taken from :
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1 as the operator A
      * \f[
      * O_{l,m}(a+b') = \sum_{j=|m|}^l{ \frac{ b^{l-j} }{ (l-j)! } O_{j,m}(a) }
      * \f]
      * As describe in the paper, when need first to rotate the SH
      * then transfer using the formula
      * and finaly rotate back.
      */
    void M2M(CellClass* const FRestrict inPole, const CellClass*const FRestrict *const FRestrict inChildren, const int inLevel)override {
        // A buffer to copy the source w allocated once
        FComplex<FReal> source_w[SizeArray];
        // For all children
        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            // if child exists
            if(inChildren[idxChild]){
                // Copy the source
                FMemUtils::copyall(source_w, inChildren[idxChild]->getMultipole(), SizeArray);

                // rotate it forward
                const FSpherical<FReal> sph = getSphericalChild(inLevel, idxChild);
                rotateMultipole(source_w, sph.getPhiZero2Pi(), sph.getInclination());

                const FReal b = -sph.getR();
                // Translate it
                FComplex<FReal> target_w[SizeArray];
                for(int l = 0 ; l <= P ; ++l ){
                    for(int m = 0 ; m <= l ; ++m ){
                        // w{l,m}(a+b) = sum(j=m:l, b^(l-j)/(l-j)! w{j,m}(a)
                        FReal w_lm_real = 0.0;
                        FReal w_lm_imag = 0.0;
                        for(int j = m ; j <= l ; ++j ){
                            const FReal coef = FMath::pow(b,l-j) / fact(l-j);
                            w_lm_real += coef * source_w[atLm(j,m)].getReal();
                            w_lm_imag += coef * source_w[atLm(j,m)].getImag();
                        }
                        target_w[atLm(l,m)].setRealImag(w_lm_real,w_lm_imag);
                    }
                }

                // Rotate it back
                deRotateMultipole(target_w, sph.getPhiZero2Pi(), sph.getInclination());

                // Sum the result
                FMemUtils::addall( inPole->getMultipole(), target_w, SizeArray);
            }
        }
    }

    /** M2L
      * The operator B has been taken from :
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1 as the operator B
      * \f[
      * M_{l,m}(a-b') = \sum_{j=|m|}^{\infty}{ \frac{ (j+l)! } { b^{j+l+1} } O_{j,-m}(a) } , \textrm{j bounded by P-l}
      * \f]
      * As describe in the paper, when need first to rotate the SH
      * then transfer using the formula
      * and finaly rotate back.
      */
    void M2L(CellClass* const FRestrict inLocal, const CellClass* inInteractions[],
             const int neighborPositions[], const int inSize, const int inLevel)  override {
        // To copy the multipole data allocated once
        FComplex<FReal> source_w[SizeArray];
        // For all children
        for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
            const int idxNeigh = neighborPositions[idxExistingNeigh];
            // Copy multipole data into buffer
            FMemUtils::copyall(source_w, inInteractions[idxExistingNeigh]->getMultipole(), SizeArray);

            // Rotate
            const FSpherical<FReal> sph = getSphericalInteraction(inLevel, idxNeigh);
            rotateMultipole(source_w, sph.getPhiZero2Pi(), sph.getInclination());

            const FReal b = sph.getR();
            // Transfer to u
            FComplex<FReal> target_u[SizeArray];
            for(int l = 0 ; l <= P ; ++l ){
                for(int m = 0 ; m <= l ; ++m ){
                    // u{l,m}(a-b) = sum(j=|m|:P-l, (j+l)!/b^(j+l+1) w{j,-m}(a)
                    FReal u_lm_real = 0.0;
                    FReal u_lm_imag = 0.0;
                    for(int j = m ; j <= P-l ; ++j ){
                        const FReal coef = FMath::pow(FReal(-1.0),m) * ( fact(j+l) / FMath::pow(b,j+l+1));
                        // because {l,-m} => {l,m} conjugate -1^m with -i
                        u_lm_real += coef * source_w[atLm(j,m)].getReal();
                        u_lm_imag -= coef * source_w[atLm(j,m)].getImag();
                    }
                    target_u[atLm(l,m)].setRealImag(u_lm_real,u_lm_imag);
                }
            }

            // Rotate it back
            deRotateTaylor(target_u, sph.getPhiZero2Pi(), sph.getInclination());

            // Sum
            FMemUtils::addall(inLocal->getLocal(), target_u, SizeArray);
        }
    }

    /** L2L
      * The operator C has been taken from :
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1 as the operator C
      * \f[
      * M_{l,m}(a-b') = \sum_{j=l}^{\infty}{ \frac{ b^{j-l} }{ (j-l)! } M_{j,m}(a) } , \textrm{j bounded by P}
      * \f]
      * As describe in the paper, when need first to rotate the SH
      * then transfer using the formula
      * and finaly rotate back.
      */
    void L2L(const CellClass* const FRestrict inLocal, CellClass* FRestrict *const FRestrict  inChildren, const int inLevel)override {
        // To copy the source local allocated once
        FComplex<FReal> source_u[SizeArray];
        // For all children
        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            // if child exists
            if(inChildren[idxChild]){
                // Copy the local data into the buffer
                FMemUtils::copyall(source_u, inLocal->getLocal(), SizeArray);

                // Rotate
                const FSpherical<FReal> sph = getSphericalChild(inLevel, idxChild);
                rotateTaylor(source_u, sph.getPhiZero2Pi(), sph.getInclination());

                const FReal b = sph.getR();
                // Translate
                FComplex<FReal> target_u[SizeArray];
                for(int l = 0 ; l <= P ; ++l ){
                    for(int m = 0 ; m <= l ; ++m ){
                        // u{l,m}(r-b) = sum(j=0:P, b^(j-l)/(j-l)! u{j,m}(r);
                        FReal u_lm_real = 0.0;
                        FReal u_lm_imag = 0.0;
                        for(int j = l ; j <= P ; ++j ){
                            const FReal coef = FMath::pow(b,j-l) / fact(j-l);
                            u_lm_real += coef * source_u[atLm(j,m)].getReal();
                            u_lm_imag += coef * source_u[atLm(j,m)].getImag();
                        }
                        target_u[atLm(l,m)].setRealImag(u_lm_real,u_lm_imag);
                    }
                }

                // Rotate
                deRotateTaylor(target_u, sph.getPhiZero2Pi(), sph.getInclination());

                // Sum in child
                FMemUtils::addall(inChildren[idxChild]->getLocal(), target_u, SizeArray);
            }
        }
    }

    /** L2P
      * Equation are coming from the PhD report of Pierre Fortin.
      * We have two different computations, one for the potential (end of function)
      * the other for the forces.
      *
      * The potential use the fallowing formula, page 36, formula 2.14 + 1:
      * \f[
      *  \Phi = \sum_{j=0}^P{\left( u_{j,0} I_{j,0}(r, \theta, \phi) + \sum_{k=1}^j{2 Re(u_{j,k} I_{j,k}(r, \theta, \phi))} \right)},
      *  \textrm{since } u_{l,-m} = (-1)^m \overline{ u_{l,m} }
      * \f]
      *
      * The forces are coming form the formulas, page 37, formulas 2.14 + 3:
      * \f[
      * F_r = -\frac{1}{r} \left( \sum_{j=1}^P{j u_{j,0} I_{j,0}(r, \theta, \phi) } + \sum_{k=1}^j{2 j Re(u_{j,k} I_{j,k}(r, \theta, \phi))} \right)
      * F_{ \theta } = -\frac{1}{r} \left( \sum_{j=0}^P{j u_{j,0} \frac{ \partial I_{j,0}(r, \theta, \phi) }{ \partial \theta } } + \sum_{k=1}^j{2 Re(u_{j,k} \frac{ \partial I_{j,k}(r, \theta, \phi) }{ \partial \theta })} \right)
      * F_{ \phi } = -\frac{1}{r sin \phi} \sum_{j=0}^P \sum_{k=1}^j{(-2k) Im(u_{j,k} I_{j,k}(r, \theta, \phi)) }
      * \f]
      */
    void L2P(const CellClass* const inLocal, ContainerClass* const inParticles)override{
        const FReal i_pow_m[4] = {0, FMath::FPiDiv2<FReal>(), FMath::FPi<FReal>(), -FMath::FPiDiv2<FReal>()};
        // Take the local value from the cell
        const FComplex<FReal>* FRestrict const u = inLocal->getLocal();

        // Copying the position is faster than using cell position
        const FPoint<FReal> cellPosition = getLeafCenter(inLocal->getCoordinate());

        // For all particles in the leaf box
        const FReal*const physicalValues = inParticles->getPhysicalValues();
        const FReal*const positionsX = inParticles->getPositions()[0];
        const FReal*const positionsY = inParticles->getPositions()[1];
        const FReal*const positionsZ = inParticles->getPositions()[2];
        FReal*const forcesX = inParticles->getForcesX();
        FReal*const forcesY = inParticles->getForcesY();
        FReal*const forcesZ = inParticles->getForcesZ();
        FReal*const potentials = inParticles->getPotentials();

        for(FSize idxPart = 0 ; idxPart < inParticles->getNbParticles() ; ++ idxPart){
            // L2P
            const FPoint<FReal> position(positionsX[idxPart],positionsY[idxPart],positionsZ[idxPart]);
            const FSpherical<FReal> sph(position - cellPosition);

            // The distance between the SH and the particle
            const FReal r = sph.getR();

            // Compute the legendre polynomial
            FReal legendre[SizeArray];
            computeLegendre(legendre, sph.getCosTheta(), sph.getSinTheta());
            {
                FReal Fr = 0;
                FReal FO = 0;
                FReal Fp = 0;

                for(int l = 1 ; l <= P ; ++l){
                    {
                        const FReal coef = FMath::pow(FReal(-1.0),l) * FMath::pow(r,l) * legendre[atLm(l,0)] / fact(l+0);
                        const FReal I_real = coef * FMath::Cos(FReal(0) * sph.getPhi() + i_pow_m[0]);
                        const FReal I_imag = coef * FMath::Sin(FReal(0) * sph.getPhi() + i_pow_m[0]);

                        Fr += FReal(l) * (u[atLm(l,0)].getReal() * I_real - u[atLm(l,0)].getImag() * I_imag);
                    }
                    {
                        const FReal coef = (FMath::pow(FReal(-1.0),l)/fact(l+0)) * FMath::pow(r,l) * ((FReal(l) * sph.getCosTheta()*legendre[atLm(l,0)]
                                                  - FReal(l) * legendre[atLm(l-1,0)]) / sph.getSinTheta());
                        const FReal dI_real = coef * FMath::Cos(FReal(0) * sph.getPhi() + i_pow_m[0 & 0x3]);
                        const FReal dI_imag = coef * FMath::Sin(FReal(0) * sph.getPhi() + i_pow_m[0 & 0x3]);
                        // F(O) += 2 * Real(L dI/dO)
                        FO += u[atLm(l,0)].getReal() * dI_real - u[atLm(l,0)].getImag() * dI_imag;
                    }

                    for(int m = 1 ; m <= l ; ++m){
                        {
                            const FReal coef = FMath::pow(FReal(-1.0),l) * FMath::pow(r,l) * legendre[atLm(l,m)] / fact(l+m);
                            const FReal I_real = coef * FMath::Cos(FReal(m) * sph.getPhi() + i_pow_m[m & 0x3]);
                            const FReal I_imag = coef * FMath::Sin(FReal(m) * sph.getPhi() + i_pow_m[m & 0x3]);
                            // F(r) += 2 x l x Real(LI)
                            Fr += 2 * FReal(l) * (u[atLm(l,m)].getReal() * I_real - u[atLm(l,m)].getImag() * I_imag);
                            // F(p) += -2 x m x Imag(LI)
                            Fp -= 2 * FReal(m) * (u[atLm(l,m)].getReal() * I_imag + u[atLm(l,m)].getImag() * I_real);
                        }
                        {
                            const FReal legendre_l_minus_1 = (m == l) ? 0 : FReal(l+m)*legendre[atLm(l-1,m)];
                            const FReal coef = (FMath::pow(FReal(-1.0),l)/fact(l+m)) * FMath::pow(r,l) * ((FReal(l) * sph.getCosTheta()*legendre[atLm(l,m)]
                                                      - legendre_l_minus_1) / sph.getSinTheta());
                            const FReal dI_real = coef * FMath::Cos(FReal(m) * sph.getPhi() + i_pow_m[m & 0x3]);
                            const FReal dI_imag = coef * FMath::Sin(FReal(m) * sph.getPhi() + i_pow_m[m & 0x3]);
                            // F(O) += 2 * Real(L dI/dO)
                            FO += FReal(2.0) * (u[atLm(l,m)].getReal() * dI_real - u[atLm(l,m)].getImag() * dI_imag);
                        }
                    }
                }
                // div by r
                Fr /= sph.getR();
                FO /= sph.getR();
                Fp /= sph.getR() * sph.getSinTheta();

                // copy variable from spherical position
                const FReal cosPhi     = FMath::Cos(sph.getPhi());
                const FReal sinPhi     = FMath::Sin(sph.getPhi());
                const FReal physicalValue = physicalValues[idxPart];

                // compute forces
                const FReal forceX = (
                        cosPhi * sph.getSinTheta() * Fr  +
                        cosPhi * sph.getCosTheta() * FO +
                        (-sinPhi) * Fp) * physicalValue;

                const FReal forceY = (
                        sinPhi * sph.getSinTheta() * Fr  +
                        sinPhi * sph.getCosTheta() * FO +
                        cosPhi * Fp) * physicalValue;

                const FReal forceZ = (
                        sph.getCosTheta() * Fr +
                        (-sph.getSinTheta()) * FO) * physicalValue;

                // inc particles forces
                forcesX[idxPart] += forceX;
                forcesY[idxPart] += forceY;
                forcesZ[idxPart] += forceZ;
            }

            { // Result for potential
                FReal magnitude = 0;
                // E = sum( l = 0:P, sum(m = -l:l, u{l,m} ))
                for(int l = 0 ; l <= P ; ++l ){
                    {
                        // (l-|m|)! * P{l,0} / r^(l+1)
                        const FReal coef = FMath::pow(FReal(-1.0),l) * FMath::pow(r,l) * legendre[atLm(l,0)] / fact(l+0);
                        const FReal I_real = coef * FMath::Cos(FReal(0) * sph.getPhi() + i_pow_m[0]);
                        const FReal I_imag = coef * FMath::Sin(FReal(0) * sph.getPhi() + i_pow_m[0]);
                        magnitude += u[atLm(l,0)].getReal() * I_real - u[atLm(l,0)].getImag() * I_imag;
                    }
                    for(int m = 1 ; m <= l ; ++m ){
                        const FReal coef = FMath::pow(FReal(-1.0),l) * FMath::pow(r,l) * legendre[atLm(l,m)] / fact(l+m);
                        const FReal I_real = coef * FMath::Cos(FReal(m) * sph.getPhi() + i_pow_m[m & 0x3]);
                        const FReal I_imag = coef * FMath::Sin(FReal(m) * sph.getPhi() + i_pow_m[m & 0x3]);
                        magnitude += FReal(2.0) * ( u[atLm(l,m)].getReal() * I_real - u[atLm(l,m)].getImag() * I_imag );
                    }
                }
                // inc potential
                potentials[idxPart] += magnitude;
            }
        }
    }


    /** P2P
      * This function proceed the P2P using particlesMutualInteraction
      * The computation is done for interactions with an index <= 13.
      * (13 means current leaf (x;y;z) = (0;0;0)).
      * Calling this method in multi thread should be done carrefully.
      */
    void P2P(const FTreeCoordinate& inPosition,
             ContainerClass* const FRestrict inTargets, const ContainerClass* const FRestrict inSources,
             ContainerClass* const inNeighbors[], const int neighborPositions[],
             const int inSize) override {
        if(inTargets == inSources){
            FP2PRT<FReal>::template Inner<ContainerClass>(inTargets);
            P2POuter(inPosition, inTargets, inNeighbors, neighborPositions, inSize);
        }
        else{
            const ContainerClass* const srcPtr[1] = {inSources};
            FP2PRT<FReal>::template FullRemote<ContainerClass>(inTargets,srcPtr,1);
            FP2PRT<FReal>::template FullRemote<ContainerClass>(inTargets,inNeighbors,inSize);
        }
    }

    void P2POuter(const FTreeCoordinate& /*inLeafPosition*/,
             ContainerClass* const FRestrict inTargets,
             ContainerClass* const inNeighbors[], const int neighborPositions[],
             const int inSize) override {
        int nbNeighborsToCompute = 0;
        while(nbNeighborsToCompute < inSize
              && neighborPositions[nbNeighborsToCompute] < 14){
            nbNeighborsToCompute += 1;
        }
        FP2PRT<FReal>::template FullMutual<ContainerClass>(inTargets,inNeighbors,nbNeighborsToCompute);
    }


    /** Use mutual even if it not useful and call particlesMutualInteraction */
    void P2PRemote(const FTreeCoordinate& /*inPosition*/,
                   ContainerClass* const FRestrict inTargets, const ContainerClass* const FRestrict /*inSources*/,
                   const ContainerClass* const inNeighbors[], const int neighborPositions[],
                   const int inSize) override {
        FP2PRT<FReal>::template FullRemote<ContainerClass>(inTargets,inNeighbors,inSize);
    }
};


#endif // FROTATIONORIGINALKERNEL_HPP
