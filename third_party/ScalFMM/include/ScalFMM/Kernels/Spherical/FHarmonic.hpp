// See LICENCE file at project root
#ifndef FHARMONIC_HPP
#define FHARMONIC_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Utils/FComplex.hpp"
#include "../../Utils/FSpherical.hpp"
#include "../../Utils/FNoCopyable.hpp"


/** This class compute the spherical harmonic.
  * It computes the inner, outter, and legendre.
  */
template <class FReal>
class FHarmonic : public FNoAssignement {
    const int devP;     //<  Degre of the Local and Multipole expansions, p.
    const int expSize;  //<  Number of elements in the expansion expSize = (p+1)^2
    const int nExpSize; //<

    FComplex<FReal>* harmonic;//< Harmonic Result
    FComplex<FReal>* cosSin;  //< Cos/Sin precomputed values
    FReal*     legendre;//< Legendre results

    FComplex<FReal>* thetaDerivatedResult; //< the theta derivated result
    FReal*     sphereHarmoInnerCoef; //< sphere innter pre computed coefficients
    FReal*     sphereHarmoOuterCoef; //< sphere outer pre computed coefficients
    //!
    //! An indirection to access the good value in the preX complexes vector. Array of size devP + 1
    //!  The storage of the triangular shape is  0,-1,0,1,-2,-1,...2, ...
    //!     and the first element of level l is stored at position l^2
    int* preExpRedirJ;

    /** Allocate and init */
    void allocAndInit(){
        harmonic = new FComplex<FReal>[expSize];
        cosSin   = new FComplex<FReal>[devP + 1];
        legendre = new FReal[expSize];
        thetaDerivatedResult = new FComplex<FReal>[expSize];

        // Pre compute coef
        sphereHarmoOuterCoef = new FReal[devP + 1];
        FReal factOuter = 1.0;
        for(int idxP = 0 ; idxP <= devP; ++idxP ){
            sphereHarmoOuterCoef[idxP] = factOuter;
            factOuter *= FReal(idxP+1);
        }

        // Pre compute coef
        sphereHarmoInnerCoef = new FReal[expSize];
        int index_l_m = 0;
        FReal factInner = 1.0;
        FReal minus_1_pow_l = 1.0;
        for(int l = 0 ; l <= devP ; ++l){
            FReal fact_l_m = factInner;
            for(int m = 0 ; m <= l ; ++m){
                sphereHarmoInnerCoef[index_l_m] = minus_1_pow_l / fact_l_m;

                fact_l_m *= FReal(l + m + 1);
                ++index_l_m;
            }
            minus_1_pow_l = -minus_1_pow_l;
            factInner *= FReal(l+1);
        }

        // Smart redirection
        preExpRedirJ = new int[2 * devP + 1];
        for( int idxP = 0; idxP <= (2 * devP) ; ++idxP ){
            preExpRedirJ[idxP] = static_cast<int>( idxP * ( idxP + 1 ) * 0.5 );
        }
    }

    //! fill in the array cosSin  such that
    //!    cosSin[l] = exp( l*inPhi) * i^l for l = 0..p
    //! \param inPhi the phi angle
    //!
    void computeCosSin(const FReal inPhi, const FReal piArray[4]){
        for(int l = 0 ; l <= devP ; ++l){
            const FReal angle = FReal(l) * inPhi + piArray[l & 0x3];
            cosSin[l].setReal( FMath::Sin(angle + FMath::FPiDiv2<FReal>()) );  // add Pi/2 for the cosine

            cosSin[l].setImag( FMath::Sin(angle) );
        }
    }

    /** Compute legendre */
    void computeLegendre(const FReal inCosTheta, const FReal inSinTheta){
        const FReal invSinTheta = -inSinTheta;

        legendre[0] = 1.0;        // P_0,0(cosTheta) = 1
        legendre[1] = inCosTheta; // P_1,0(cosTheta) = cosTheta
        legendre[2] = invSinTheta;// P_1,1(cosTheta) = -sinTheta

        int idxCurrentLM  = 3; //current pointer on P_l,m
        int idxCurrentL1M = 1; //pointer on P_{l-1},m => P_1,0
        int idxCurrentL2M = 0; //pointer on P_{l-2},m => P_0,0
        FReal fact = 3.0;

        for(int l = 2; l <= devP ; ++l ){
            // m from 0 to l - 2
            for( int m = 0; m <= l - 2 ; ++m ){
                //         cosTheta x (2 x l - 1) x P_l-1_m - (l + m - 1) x P_l-2_m
                // P_l_m = --------------------------------------------------------
                //                               l - m
                legendre[idxCurrentLM] = (inCosTheta * FReal( 2 * l - 1 ) * legendre[idxCurrentL1M]
                                          - FReal( l + m - 1 ) * legendre[idxCurrentL2M] )
                                          / FReal( l - m );


                // progress
                ++idxCurrentLM;
                ++idxCurrentL1M;
                ++idxCurrentL2M;
            }

            // Compute P_l,{l-1}
            legendre[idxCurrentLM++] = inCosTheta * FReal( 2 * l - 1 ) * legendre[idxCurrentL1M];

            // Compute P_l,l
            legendre[idxCurrentLM++] = fact * invSinTheta * legendre[idxCurrentL1M];

            fact += FReal(2.0);
            ++idxCurrentL1M;
        }
    }

public:
    /////////////////////////////////////////////////////////////////
    // Constructor & Accessor
    /////////////////////////////////////////////////////////////////

    explicit FHarmonic(const int inDevP)
        : devP(inDevP),expSize(int(((inDevP)+1) * ((inDevP)+2) * 0.5)),
          nExpSize((inDevP + 1) * (inDevP + 1)),
          harmonic(nullptr), cosSin(nullptr), legendre(nullptr), thetaDerivatedResult(nullptr),
          sphereHarmoInnerCoef(nullptr), sphereHarmoOuterCoef(nullptr), preExpRedirJ(nullptr)  {

        allocAndInit();
    }

    FHarmonic(const FHarmonic& other)
        : devP(other.devP),expSize(other.expSize), nExpSize(other.expSize),
          harmonic(nullptr), cosSin(nullptr), legendre(nullptr), thetaDerivatedResult(nullptr),
          sphereHarmoInnerCoef(nullptr), sphereHarmoOuterCoef(nullptr), preExpRedirJ(nullptr)  {

        allocAndInit();
    }

    ~FHarmonic(){
        delete[] harmonic;
        delete[] cosSin;
        delete[] legendre;
        delete[] thetaDerivatedResult;
        delete[] sphereHarmoInnerCoef;
        delete[] sphereHarmoOuterCoef;
        delete[] preExpRedirJ;
    }

    int getExpSize() const{
        return expSize;
    }

    int getNExpSize() const{
        return nExpSize;
    }

    FComplex<FReal>* result(){
        return harmonic;
    }

    const FComplex<FReal>* result() const {
        return harmonic;
    }

    FComplex<FReal>& result(const int index){
        return harmonic[index];
    }

    const FComplex<FReal>& result(const int index) const{
        return harmonic[index];
    }

    FComplex<FReal>* resultThetaDerivated(){
        return thetaDerivatedResult;
    }

    const FComplex<FReal>* resultThetaDerivated() const {
        return thetaDerivatedResult;
    }

    FComplex<FReal>& resultThetaDerivated(const int index){
        return thetaDerivatedResult[index];
    }

    const FComplex<FReal>& resultThetaDerivated(const int index) const{
        return thetaDerivatedResult[index];
    }

    int getPreExpRedirJ(const int index) const{
        return preExpRedirJ[index];
    }

    /////////////////////////////////////////////////////////////////
    // Computation
    /////////////////////////////////////////////////////////////////

    void computeInner(const FSpherical<FReal>& inSphere){
        // For i^|m|
        const FReal PiArrayInner[4] = {0, FMath::FPiDiv2<FReal>(), FMath::FPi<FReal>(), -FMath::FPiDiv2<FReal>()};
        //
        computeCosSin(inSphere.getPhi(), PiArrayInner);

        // fill legendre array
        computeLegendre(inSphere.getCosTheta(), inSphere.getSinTheta());

        int index_l_m = 0;
        FReal r_pow_l = 1.0 ;

        for(int l = 0; l <= devP ; ++l){
            for(int m = 0 ; m <= l ; ++m){
                const FReal magnitude = sphereHarmoInnerCoef[index_l_m] * r_pow_l * legendre[index_l_m];
                harmonic[index_l_m].setReal( magnitude * cosSin[m].getReal() );
                harmonic[index_l_m].setImag( magnitude * cosSin[m].getImag() );

                ++index_l_m;
            }

            r_pow_l *= inSphere.getR();
        }
    }
    /*
    * \brief Compute The inner expansion on the z axis
    *   On the z axis theta = phi = 0 and the the expression of the coefficients are
    *     I_l^m= (-1)^l i^m r^l/(l+|m|)! Kronecker(m,0)
    *   \result P+1 cofficients of the Inner expansion.
    */
    void computeInnerOnZaxis(const FReal& inRadius,FReal* Inner){
        // Not used : const FReal PiArrayInner[4] = {0, FMath::FPiDiv2<FReal>(), FMath::FPi<FReal>(), -FMath::FPiDiv2<FReal>()};
        //
        int   index_l = 0   ;
        FReal r_pow_l = 1.0 ;

        for(int l = 0; l <= devP ; ++l){
            Inner[l] = sphereHarmoInnerCoef[index_l] * r_pow_l;
            r_pow_l *= inRadius ;
            index_l +=  l+1 ;
        }
    }

    void computeOuter(const FSpherical<FReal>& inSphere){
        const FReal PiArrayOuter[4] = {0, -FMath::FPiDiv2<FReal>(), FMath::FPi<FReal>(), FMath::FPiDiv2<FReal>()};
        computeCosSin(inSphere.getPhi(), PiArrayOuter);

        // fill legendre array
        computeLegendre(inSphere.getCosTheta(), inSphere.getSinTheta());

        const FReal invR = 1/inSphere.getR();
        int index_l_m = 0;
        FReal invR_pow_l1 = invR;

        for(int l = 0 ; l <= devP ; ++l){
            for(int m = 0 ; m <= l ; ++m){
                const FReal magnitude = sphereHarmoOuterCoef[l-m] * invR_pow_l1 * legendre[index_l_m];
                harmonic[index_l_m].setReal( magnitude * cosSin[m].getReal() );
                harmonic[index_l_m].setImag( magnitude * cosSin[m].getImag() );

                ++index_l_m;
            }

            invR_pow_l1 *= invR;
        }
    }

    /** spherical_harmonic_Inner_and_theta_derivated
        * Returns the value of the partial derivative of the spherical harmonic
        *relative to theta. We have for all m such that -(l-1) <= m <= l-1 :
        *
        *(d H_l^m(theta, phi))/(d theta)
        *= (-1)^m sqrt((l-|m|)!/(l+|m|)!) (d P_l^{|m|}(cos theta))/(d theta) e^{i.m.phi}
        *= (-1)^m sqrt((l-|m|)!/(l+|m|)!) 1/sqrt{1-cos^2 theta} [l cos theta P_l^{|m|}(cos theta) - (l+|m|) P_{l-1}^{|m|}(cos theta) ] e^{i.m.phi}
        *Since theta is in the range [0, Pi], we have: sin theta > 0 and therefore
        *sqrt{1-cos^2 theta} = sin theta. Thus:
        *
        *(d H_l^m(theta, phi))/(d theta)
        *= (-1)^m sqrt((l-|m|)!/(l+|m|)!) 1/(sin theta) [l cos theta P_l^{|m|}(cos theta) - (l+|m|) P_{l-1}^{|m|}(cos theta) ] e^{i.m.phi}
        *For |m|=l, we have~:
        *(d H_l^l(theta, phi))/(d theta)
        *= (-1)^m sqrt(1/(2l)!) 1/(sin theta) [l cos theta P_l^l(cos theta) ] e^{i.m.phi}
        *
        *Remark: for 0<m<=l:
        *(d H_l^{-m}(theta, phi))/(d theta) = [(d H_l^{-m}(theta, phi))/(d theta)]*
        *
        *
        *
        *Therefore, we have for (d Inner_l^m(r, theta, phi))/(d theta):
        *
        *|m|<l: (d Inner_l^m(r, theta, phi))/(d theta) =
        *(i^m (-1)^l / (l+|m|)!) 1/(sin theta) [l cos theta P_l^{|m|}(cos theta) - (l+|m|) P_{l-1}^{|m|}(cos theta) ] e^{i.m.phi} r^l
        *|m|=l: (d Inner_l^m(r, theta, phi))/(d theta) =
        *(i^m (-1)^l / (l+|m|)!) 1/(sin theta) [l cos theta P_l^l(cos theta) ] e^{i.m.phi} r^l
        *
        *
      */
    void computeInnerTheta(const FSpherical<FReal>& inSphere){
        const FReal PiArrayInner[4] = {0, FMath::FPiDiv2<FReal>(), FMath::FPi<FReal>(), -FMath::FPiDiv2<FReal>()};
        computeCosSin(inSphere.getPhi(),PiArrayInner);

        // fill legendre array
        computeLegendre(inSphere.getCosTheta(), inSphere.getSinTheta());

        int index_l_m = 0;
        // r^l
        FReal r_pow_l = 1.0;

        for (int l = 0 ; l <= devP ; ++l){
            FReal magnitude;
            // m<l:
            int m = 0;
            for(; m < l ; ++m){
                magnitude = sphereHarmoInnerCoef[index_l_m] * r_pow_l * legendre[index_l_m];

                // Computation of Inner_l^m(r, theta, phi):
                harmonic[index_l_m].setReal( magnitude * cosSin[m].getReal());
                harmonic[index_l_m].setImag( magnitude * cosSin[m].getImag());

                // Computation of {\partial Inner_l^m(r, theta, phi)}/{\partial theta}:
                magnitude = sphereHarmoInnerCoef[index_l_m] * r_pow_l * ((FReal(l)*inSphere.getCosTheta()*legendre[index_l_m]
                                                                                       - FReal(l+m)*(legendre[getPreExpRedirJ(l-1) + m])) / inSphere.getSinTheta());
                thetaDerivatedResult[index_l_m].setReal(magnitude * cosSin[m].getReal());
                thetaDerivatedResult[index_l_m].setImag(magnitude * cosSin[m].getImag());

                ++index_l_m;
            }

            // m=l:
            // Computation of Inner_m^m(r, theta, phi):
            magnitude = sphereHarmoInnerCoef[index_l_m] * r_pow_l * legendre[index_l_m];
            harmonic[index_l_m].setReal(magnitude * cosSin[m].getReal());
            harmonic[index_l_m].setImag(magnitude * cosSin[m].getImag());

            // Computation of {\partial Inner_m^m(r, theta, phi)}/{\partial theta}:
            magnitude = sphereHarmoInnerCoef[index_l_m] * r_pow_l * (FReal(m) * inSphere.getCosTheta() * legendre[index_l_m] / inSphere.getSinTheta());
            thetaDerivatedResult[index_l_m].setReal(magnitude * cosSin[m].getReal());
            thetaDerivatedResult[index_l_m].setImag(magnitude * cosSin[m].getImag());

            r_pow_l *= inSphere.getR();
            ++index_l_m;
        }
    }

};



#endif // FHARMONIC_HPP
