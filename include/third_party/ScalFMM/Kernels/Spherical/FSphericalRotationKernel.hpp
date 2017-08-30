// ===================================================================================
// Copyright ScalFmm 2011 INRIA,
// olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.  
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info". 
// "http://www.gnu.org/licenses".
// ===================================================================================
#ifndef FSPHERICALROTATIONKERNEL_HPP
#define FSPHERICALROTATIONKERNEL_HPP

#include "FAbstractSphericalKernel.hpp"
#include "Utils/FMemUtils.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* This class is the rotation spherical harmonic kernel
*/
template<  class FReal, class CellClass, class ContainerClass>
class FSphericalRotationKernel : public FAbstractSphericalKernel<FReal, CellClass,ContainerClass> {
protected:
    typedef FAbstractSphericalKernel<FReal, CellClass,ContainerClass> Parent;

    /** This class define some information to use rotation computation
      */
    struct RotationInfo{
        FReal* rotation_a;
        FReal* rotation_b;

        FComplex<FReal>* p_rot_multipole_exp;
        FComplex<FReal>* p_rot_local_exp;

        /** Get z vector size */
        static int ZAxisExpensionSize(const int inDevP){
            return int( (inDevP&1) == 0 ? ((inDevP+1) + (inDevP*inDevP)*.25) : ((inDevP+1) + (inDevP*inDevP-1)*.25));
        }

        /** Constructor */
        RotationInfo(const int inDevP){
            rotation_a = new FReal[int( ((inDevP)+1) * ((inDevP)+2) * 0.5 )];
            for(int n = 0 ; n <= inDevP ; ++n){
                for(int m = 0 ; m <= n ; ++m){
                    rotation_a[int(n*(n+1) * 0.5 + m)] = FMath::Sqrt( FReal((n+1+m)*(n+1-m)) / FReal(((2*n+1)*(2*n+3))) );
                }
            }
            rotation_b = new FReal[(inDevP+1) * (inDevP+1)];
            for(int n = 0 ; n <= inDevP ; ++n){
                for(int m = -n ; m < 0 ; ++m){
                    rotation_b[n*(n+1) + m] = -FMath::Sqrt( FReal((n-m-1)*(n-m)) / FReal(((2*n-1)*(2*n+1))) );
                }
                for(int m = 0 ; m <= n ; ++m){
                    rotation_b[n*(n+1) + m] = FMath::Sqrt( FReal((n-m-1)*(n-m)) / FReal(((2*n-1)*(2*n+1))) );
                }
            }
            const int z_size = ZAxisExpensionSize(inDevP);
            p_rot_multipole_exp = new FComplex<FReal>[z_size];
            p_rot_local_exp = new FComplex<FReal>[z_size];
        }

        /** Destructor */
        ~RotationInfo(){
            delete[] rotation_a;
            delete[] rotation_b;
            delete[] p_rot_multipole_exp;
            delete[] p_rot_local_exp;
        }
    };


    /** This class holds the data need to do a M2L by rotation
      * it is precomputed at the beginning
      */
    struct RotationM2LTransfer {
        const int devP;
        const int expSize;
        FComplex<FReal>** rcc_outer;
        FComplex<FReal>** rcc_inner;
        FReal* outer_array;

        /** Constructor */
        RotationM2LTransfer(const int inDevP, const int inDevM2lP, const int inExpSize)
            : devP(inDevP), expSize(inExpSize){
            rcc_outer = new FComplex<FReal>*[devP + 1];
            rcc_inner = new FComplex<FReal>*[devP + 1];
            for( int idxP = 0 ; idxP <= devP ; ++idxP){
                const int rotationSize = ((idxP+1)*(2*idxP+1));
                rcc_outer[idxP] = new FComplex<FReal>[rotationSize];
                rcc_inner[idxP] = new FComplex<FReal>[rotationSize];
            }
            outer_array = new FReal[inDevM2lP + 1];
        }

        /** Used in the initialisation */
        void spherical_harmonic_Outer_null_order_z_axis(const FReal r){
            const FReal inv_r = FReal(1.0 / r);
            FReal tmp = inv_r;

            // l=0
            outer_array[0] = tmp;

            // l>0
            for(int l = 1 ; l <= devP ; ++l){
                tmp *= inv_r * FReal(l);
                outer_array[l] = tmp;
            }
        }

        /** Used in the initialisation */
        void computeLegendre(const FReal inCosTheta, const FReal inSinTheta, FReal legendre[]){
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

        /** Used in the initialisation */
        static int getTranspRotationCoefP(const int n, const int nu, const int m){
            return m*(2*n+1) + (nu+n);
        }
        /** Used in the initialisation */
        static int getRotationCoefP(const int n, const int nu, const int m){
            return (nu)*(2*(n)+1) + (m+n);
        }
        /** Used in the initialisation */
        static int getRotationB(const int n, const int m){
            return n*(n+1) + m;
        }
        /** Used in the initialisation */
        static int getRotationA(const int n, const int m){
            return int(n*(n+1) * 0.5 + (FMath::Abs(m)));
        }

        /** Used in the initialisation */
        static FReal A_div_A(int n, int m, int nu){
            m = FMath::Abs(m);
            nu = FMath::Abs(nu);
            const int min = FMath::Min(m, nu);
            const int max = FMath::Max(m, nu);
            const int i_stop = max - min - 1; /* = n-min - (n-max+1) = n+max - (n+min+1) */

            FReal num = FReal(n-max+1);
            FReal denom = FReal(n+min+1);
            FReal A_min_max = 1;
            for (int i=0; i<=i_stop; ++i, ++num, ++denom){
                A_min_max *= (num/denom);
            }

            if (nu == min)
                return FMath::Sqrt(A_min_max);
            else
                return 1/FMath::Sqrt(A_min_max);
        }

        /** Pre-Compute */
        void rotation_coefficient_container_Fill(const FReal omega,
                                                 const FReal cos_gamma, const FReal sin_gamma,
                                                 const FReal chi, const RotationInfo& rotation_Info){

            FComplex<FReal>** rcc_tmp_transposed = new FComplex<FReal>*[devP + 1];
            for( int idxP = 0 ; idxP <= devP ; ++idxP){
                const int rotationSize = ((idxP+1)*(2*idxP+1));
                rcc_tmp_transposed[idxP] = new FComplex<FReal>[rotationSize];
            }

            FComplex<FReal> _pow_of_I_array[7];
            _pow_of_I_array[0].setRealImag(0 , 1 ) /* I^{-3} */;
            _pow_of_I_array[1].setRealImag(-1, 0 ) /* I^{-2} */;
            _pow_of_I_array[2].setRealImag(0 , -1) /* I^{-1} */;
            _pow_of_I_array[3].setRealImag(1 , 0 ) /* I^0 */;
            _pow_of_I_array[4].setRealImag(0 , 1 ) /* I^1 */;
            _pow_of_I_array[5].setRealImag(-1, 0 ) /* I^2 */;
            _pow_of_I_array[6].setRealImag(0 , -1) /* I^3 */;

            const FComplex<FReal>* pow_of_I_array = _pow_of_I_array + 3; /* points on I^0 */

            FComplex<FReal>* const _precomputed_exp_I_chi_array = new FComplex<FReal>[2*devP + 1];
            FComplex<FReal>* precomputed_exp_I_chi_array = _precomputed_exp_I_chi_array + devP;

            FComplex<FReal>* const _precomputed_exp_I_omega_array  = new FComplex<FReal>[2*devP + 1];
            FComplex<FReal>* precomputed_exp_I_omega_array = _precomputed_exp_I_omega_array + devP;


            // cos(x) = sin(x + Pi/2)
            for(int m = -devP ; m <= devP ; ++m){
                precomputed_exp_I_chi_array[m].setReal(FMath::Sin(FReal(m)*chi + FMath::FPiDiv2<FReal>()));
                precomputed_exp_I_chi_array[m].setImag(FMath::Sin(FReal(m)*chi));
            }
            for(int nu = -devP ; nu <= devP ; ++nu){
                precomputed_exp_I_omega_array[nu].setReal(FMath::Sin(FReal(nu)*omega + FMath::FPiDiv2<FReal>()));
                precomputed_exp_I_omega_array[nu].setImag(FMath::Sin(FReal(nu)*omega));
            }

            FReal*const ass_Legendre_func_Array = new FReal[expSize];
            FReal* p_ass_Legendre_func_Array = ass_Legendre_func_Array;
            computeLegendre(cos_gamma, sin_gamma, ass_Legendre_func_Array);

            for(int n = 0 ; n <= devP ; ++n){
                // nu == 0:
                FReal c_n_nu = 1;
                rcc_tmp_transposed[n][getTranspRotationCoefP(n,0,0)].setReal(c_n_nu * (*p_ass_Legendre_func_Array));

                ++p_ass_Legendre_func_Array;

                // nu > 0:
                FReal minus_1_pow_nu = -1;
                for(int nu = 1 ; nu <= n; ++nu){
                    c_n_nu /= FMath::Sqrt(FReal((n-nu+1)*(n+nu)));
                    rcc_tmp_transposed[n][getTranspRotationCoefP(n, nu, 0)].setReal(minus_1_pow_nu * c_n_nu * (*p_ass_Legendre_func_Array));
                    rcc_tmp_transposed[n][getTranspRotationCoefP(n, -nu, 0)] = rcc_tmp_transposed[n][getTranspRotationCoefP(n, nu, 0)];
                    minus_1_pow_nu = -minus_1_pow_nu;
                    ++p_ass_Legendre_func_Array;
                } // for nu

                for(int m = 1 ; m <= n ; ++m){
                    for(int nu = -m; nu <= m ; ++nu){
                        const FReal H_nu_minus_1 = ( nu-1 <= -n ?
                                             FReal(0.0) :
                                             (cos_gamma +1) * rotation_Info.rotation_b[getRotationB(n, -nu)]
                                             * rcc_tmp_transposed[n-1][getTranspRotationCoefP(n-1, nu-1, m-1)].getReal());
                        const FReal H_nu_plus_1 = ( nu+1 >= n ?
                                            FReal(0.0) :
                                            (cos_gamma -1) * rotation_Info.rotation_b[getRotationB(n, nu)]
                                            * rcc_tmp_transposed[n-1][getTranspRotationCoefP(n-1, nu+1, m-1)].getReal());

                        const FReal H_nu  = ( FMath::Abs(nu) >= n ?
                                      FReal(0.0) :
                                      sin_gamma * rotation_Info.rotation_a[getRotationA(n-1, nu)]
                                      * rcc_tmp_transposed[n-1][getTranspRotationCoefP(n-1, nu, m-1)].getReal() );


                        rcc_tmp_transposed[n][getTranspRotationCoefP(n, nu, m)].setReal( (FReal(0.5) * (-H_nu_minus_1 - H_nu_plus_1) - H_nu)
                                                                                         / rotation_Info.rotation_b[getRotationB(n, -m)]);
                    } // for nu
                } // for m

                for(int m = 1 ; m <= n ; ++m){
                    for(int nu = -n ; nu <= -m-1; ++nu){
                        rcc_tmp_transposed[n][getTranspRotationCoefP(n, nu, m)] = rcc_tmp_transposed[n][getTranspRotationCoefP(n, -m, -nu)];
                    } // for nu

                    for(int nu = m+1 ; nu <= n; ++nu){
                        rcc_tmp_transposed[n][getTranspRotationCoefP(n, nu, m)] = rcc_tmp_transposed[n][getTranspRotationCoefP(n, m, nu)];
                    } // for nu
                } // for m
            } // for n

            for(int n = 0 ; n <= devP ; ++n){
                for(int nu = 0 ; nu <= n; ++nu){
                    for(int m = -n; m <= n; ++m){
                        FReal A_terms = A_div_A(n, m, nu); /*  A_n^m / A_n^nu */
                        int abs_m_minus_abs_nu_mod4 = (FMath::Abs(m) - FMath::Abs(nu)) % 4; /* can be negative! */
                        const FComplex<FReal> p_H_tmp = ( m >= 0 ?
                                        rcc_tmp_transposed[n][getTranspRotationCoefP(n, nu, m)] :
                                        rcc_tmp_transposed[n][getTranspRotationCoefP(n, -nu, -m)]) ;

                        /*************** T_Outer_n^{nu, m}(omega, gamma, chi): ***************/
                        rcc_outer[n][getRotationCoefP(n, nu, m)] = p_H_tmp;
                        /* H_n^{nu, m}(gamma) => T_n^{nu, m}(omega, gamma, chi) */
                        rcc_outer[n][getRotationCoefP(n, nu, m)] *= precomputed_exp_I_chi_array[ + m];
                        rcc_outer[n][getRotationCoefP(n, nu, m)] *= precomputed_exp_I_omega_array[ - nu];
                        /* T_Outer_j^{nu, k}(omega, gamma, chi) = i^{|k|-|nu|} (A_j^nu / A_j^k) T_j^{nu, k}(omega, gamma, chi)     (6) */
                        rcc_outer[n][getRotationCoefP(n, nu, m)] *= pow_of_I_array[abs_m_minus_abs_nu_mod4];
                        rcc_outer[n][getRotationCoefP(n, nu, m)] *= (FReal(1.0) / A_terms);


                        /*************** T_Inner_n^{nu, m}(chi, gamma, omega): ***************/
                        rcc_inner[n][getRotationCoefP(n, nu, m)] = p_H_tmp;
                        /* H_n^{nu, m}(gamma) => T_n^{nu, m}(chi, gamma, omega) */
                        rcc_inner[n][getRotationCoefP(n, nu, m)] *= precomputed_exp_I_omega_array[ + m];
                        rcc_inner[n][getRotationCoefP(n, nu, m)] *= precomputed_exp_I_chi_array[ - nu];
                        /* T_Inner_j^{nu, k}(omega, gamma, chi) = i^{|nu|-|k|} (A_j^k / A_j^nu) T_j^{nu, k}(omega, gamma, chi)    (7) */
                        rcc_inner[n][getRotationCoefP(n, nu, m)] *= pow_of_I_array[- abs_m_minus_abs_nu_mod4];
                        rcc_inner[n][getRotationCoefP(n, nu, m)] *= A_terms;

                    }// for m
                } // for nu
            } // for n


            delete[] (ass_Legendre_func_Array);
            delete[] (_precomputed_exp_I_chi_array);
            delete[] (_precomputed_exp_I_omega_array);
            FMemUtils::DeleteAllArray( rcc_tmp_transposed, devP);
            delete[] rcc_tmp_transposed;
        }

        /** Pre-compute */
        void transfer_M2L_rotation_Fill(const FSpherical<FReal>& inSphere, const RotationInfo& rotation_Info){

            // Computes rotation coefficients:
            rotation_coefficient_container_Fill(FMath::FPi<FReal>(), inSphere.getCosTheta(),
                                                inSphere.getSinTheta(), inSphere.getPhi(), rotation_Info);

            // Computes Outer terms:
            spherical_harmonic_Outer_null_order_z_axis(inSphere.getR());


        }

        ~RotationM2LTransfer(){
            FMemUtils::DeleteAllArray( rcc_outer, devP);
            FMemUtils::DeleteAllArray( rcc_inner, devP);
            delete[] rcc_outer;
            delete[] rcc_inner;
            delete[] outer_array;
        }
    };

    const int devM2lP;               //< A secondary P

    FSmartPointer<RotationM2LTransfer*> preM2LTransitions;   //< The pre-computation for the M2L based on the level and the 189 possibilities
    RotationInfo rotation_Info;

    /** To access te pre computed M2L transfer vector */
    int indexM2LTransition(const int idxX,const int idxY,const int idxZ) const {
        return (((((idxX+3) * 7) + (idxY+3)) * 7 ) + (idxZ+3));
    }

    /** Alloc and init pre-vectors*/
    void allocAndInit(){
        // M2L transfer, there is a maximum of 3 neighbors in each direction,
        // so 6 in each dimension
        preM2LTransitions = new RotationM2LTransfer*[Parent::treeHeight];
        memset(preM2LTransitions.getPtr(), 0, sizeof(FComplex<FReal>*) * (Parent::treeHeight));
        // We start from the higher level
        FReal treeWidthAtLevel = Parent::boxWidth;
        for(int idxLevel = 0 ; idxLevel < Parent::treeHeight ; ++idxLevel ){
            // Allocate data for this level
            preM2LTransitions[idxLevel] = reinterpret_cast<RotationM2LTransfer*>(new char[(7 * 7 * 7) * sizeof(RotationM2LTransfer)]);
            // Precompute transfer vector
            for(int idxX = -3 ; idxX <= 3 ; ++idxX ){
                for(int idxY = -3 ; idxY <= 3 ; ++idxY ){
                    for(int idxZ = -3 ; idxZ <= 3 ; ++idxZ ){
                        new (&preM2LTransitions[idxLevel][indexM2LTransition(idxX,idxY,idxZ)]) RotationM2LTransfer(Parent::devP,devM2lP,Parent::harmonic.getExpSize());

                        if(FMath::Abs(idxX) > 1 || FMath::Abs(idxY) > 1 || FMath::Abs(idxZ) > 1){
                            const FPoint<FReal> relativePos( FReal(-idxX) * treeWidthAtLevel , FReal(-idxY) * treeWidthAtLevel , FReal(-idxZ) * treeWidthAtLevel );
                            preM2LTransitions[idxLevel][indexM2LTransition(idxX,idxY,idxZ)].transfer_M2L_rotation_Fill(FSpherical<FReal>(relativePos), rotation_Info);
                        }
                    }
                }
            }
            // We divide the bow per 2 when we go down
            treeWidthAtLevel /= 2;
        }
    }


public:
    /** Constructor
      * @param inDevP the polynomial degree
      * @param inThreeHeight the height of the tree
      * @param inBoxWidth the size of the simulation box
      * @param inPeriodicLevel the number of level upper to 0 that will be requiried
      */
    FSphericalRotationKernel(const int inDevP, const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter)
        : Parent(inDevP, inTreeHeight, inBoxWidth, inBoxCenter),
          devM2lP(int(((inDevP*2)+1) * ((inDevP*2)+2) * 0.5)),
          preM2LTransitions(nullptr),
          rotation_Info(inDevP) {
        allocAndInit();
    }

    /** Copy constructor */
    FSphericalRotationKernel(const FSphericalRotationKernel& other)
        : Parent(other), devM2lP(other.devM2lP),
          preM2LTransitions(other.preM2LTransitions),
          rotation_Info(other.devP) {

    }

    /** Destructor */
    ~FSphericalRotationKernel(){
        if( preM2LTransitions.isLast() ){
            for(int idxLevel = 0 ; idxLevel < Parent::treeHeight ; ++idxLevel ){
                for(int idx = 0 ; idx < 7*7*7 ; ++idx ){
                    preM2LTransitions[idxLevel][idx].~RotationM2LTransfer();
                }
                delete[] reinterpret_cast<char*>(preM2LTransitions[idxLevel]);
            }
        }
    }

    /** M2L with a cell and all the existing neighbors */
    void M2L(CellClass* const FRestrict inLocal, const CellClass* distantNeighbors[],
             const int neighborPositions[], const int inSize, const int inLevel)  override {
        // For all neighbors compute M2L
        for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
            const int idxNeigh = neighborPositions[idxExistingNeigh];
            const RotationM2LTransfer& transitionVector = preM2LTransitions[inLevel][idxNeigh];
            multipoleToLocal(inLocal->getLocal(), distantNeighbors[idxExistingNeigh]->getMultipole(), transitionVector);
        }
    }


    /** M2L With rotation
      */
    void multipoleToLocal(FComplex<FReal>*const FRestrict local_exp_target, const FComplex<FReal>* const FRestrict multipole_exp_src,
                          const RotationM2LTransfer& transfer_M2L_rotation){

        memset(rotation_Info.p_rot_multipole_exp, 0, RotationInfo::ZAxisExpensionSize(Parent::devP) * sizeof(FComplex<FReal>));
        memset(rotation_Info.p_rot_local_exp, 0, RotationInfo::ZAxisExpensionSize(Parent::devP) * sizeof(FComplex<FReal>));

        rotation_Rotate_multipole_expansion_terms(multipole_exp_src, transfer_M2L_rotation.rcc_outer, rotation_Info.p_rot_multipole_exp);

        M2L_z_axis(rotation_Info.p_rot_local_exp, rotation_Info.p_rot_multipole_exp, transfer_M2L_rotation.outer_array);

        rotation_Rotate_local_expansion_terms(rotation_Info.p_rot_local_exp, transfer_M2L_rotation.rcc_inner, local_exp_target);
    }

    /** Needed when doing the M2L */
    void rotation_Rotate_multipole_expansion_terms(const FComplex<FReal>*const FRestrict multipole_exp,
                                                   const FComplex<FReal>* const FRestrict * const FRestrict rcc_outer,
                                                   FComplex<FReal>*const FRestrict rot_multipole_exp){

        FComplex<FReal>* p_rot_multipole_exp = rot_multipole_exp;

        for(int nu = 0 ; nu <= (Parent::devP/2) ; ++nu){
            for(int j = nu; j <= (Parent::devP-nu) ; ++j){
                const FComplex<FReal>* p_rcc_outer = &rcc_outer[j][RotationM2LTransfer::getRotationCoefP(j, nu, j)];
                const FComplex<FReal>* p_multipole_exp = &multipole_exp[Parent::harmonic.getPreExpRedirJ(j) + j];
                FReal minus_1_pow_k = FReal(j&1 ? -1 : 1);

                for(int k = -j ; k < 0 ; ++k){ /* k < 0 */
                    p_rot_multipole_exp->incReal( minus_1_pow_k *
                                                  ((p_multipole_exp->getReal() * p_rcc_outer->getReal()) +
                                                   (p_multipole_exp->getImag() * p_rcc_outer->getImag())) );
                    p_rot_multipole_exp->incImag( minus_1_pow_k *
                                                  ((p_multipole_exp->getReal() * p_rcc_outer->getImag()) -
                                                   (p_multipole_exp->getImag() * p_rcc_outer->getReal())) );

                    minus_1_pow_k = -minus_1_pow_k;
                    --p_rcc_outer;
                    --p_multipole_exp;
                } /* for k */

                for(int k = 0; k <= j ; ++k){ /* k >= 0 */
                    p_rot_multipole_exp->incReal(
                                ((p_multipole_exp->getReal() * p_rcc_outer->getReal()) -
                                 (p_multipole_exp->getImag() * p_rcc_outer->getImag())) );
                    p_rot_multipole_exp->incImag(
                                ((p_multipole_exp->getReal() * p_rcc_outer->getImag()) +
                                 (p_multipole_exp->getImag() * p_rcc_outer->getReal())) );

                    --p_rcc_outer;
                    ++p_multipole_exp;
                } /* for k */

                ++p_rot_multipole_exp;
            } /* for j */
        } /* for nu */
    }

    /** Needed when doing the M2L */
    void M2L_z_axis(FComplex<FReal>* const FRestrict rot_local_exp,
                    const FComplex<FReal>* const FRestrict rot_multipole_exp,
                    const FReal* const outer_array){
        FComplex<FReal>* p_rot_local_exp = rot_local_exp;

        for(int j = 0 ; j <= Parent::devP; ++j){
            const FReal* p_outer_array_j = outer_array + j;
            const int stop_for_n = Parent::devP-j;
            const int min_j = FMath::Min(j, stop_for_n);
            for(int k = 0 ; k <= min_j ; ++k){
                const FComplex<FReal>* p_rot_multipole_exp = rot_multipole_exp + k * (Parent::devP + 2 - k);
                for(int n = k ; n <= stop_for_n ; ++n){
                    p_rot_local_exp->incReal(p_rot_multipole_exp->getReal() * p_outer_array_j[n]);
                    p_rot_local_exp->incImag(p_rot_multipole_exp->getImag() * p_outer_array_j[n]);
                    ++p_rot_multipole_exp;
                } /* for n */
                ++p_rot_local_exp;
            } /* for k */
        } /* for j */
    }

    /** Needed when doing the M2L */
    void rotation_Rotate_local_expansion_terms(const FComplex<FReal>*const rot_local_exp,
                                               const FComplex<FReal>*const FRestrict *const FRestrict rcc_inner,
                                               FComplex<FReal>*const FRestrict local_exp){
        const int Q = Parent::devP/2;

        FComplex<FReal>* FRestrict p_local_exp = local_exp;

        for(int j = 0 ; j <= Q ; ++j){
            const int min_j = j;
            const FComplex<FReal>* const FRestrict p_rot_local_exp_j = &rot_local_exp[Parent::harmonic.getPreExpRedirJ(j) + j];

            for (int nu = 0 ; nu <= j; ++nu){
                const FComplex<FReal>* FRestrict p_rcc_inner = &rcc_inner[j][RotationM2LTransfer::getRotationCoefP(j, nu, -min_j)];
                const FComplex<FReal>* FRestrict p_rot_local_exp = p_rot_local_exp_j;
                FReal minus_1_pow_k = FReal(min_j&1 ? -1 : 1);

                for(int k = -min_j ; k < 0 ; ++k){  /* k < 0 */
                    p_local_exp->incReal( minus_1_pow_k *
                                          ((p_rot_local_exp->getReal() * p_rcc_inner->getReal()) +
                                           (p_rot_local_exp->getImag() * p_rcc_inner->getImag())));
                            p_local_exp->incImag( minus_1_pow_k *
                                                  ((p_rot_local_exp->getReal() * p_rcc_inner->getImag()) -
                                                   (p_rot_local_exp->getImag() * p_rcc_inner->getReal())));

                            minus_1_pow_k = -minus_1_pow_k;
                    --p_rot_local_exp;
                    ++p_rcc_inner;
                } /* for k */

                for(int k = 0; k <= min_j ; ++k){  /* k >= 0 */
                    p_local_exp->incReal(
                                ((p_rot_local_exp->getReal() * p_rcc_inner->getReal()) -
                                 (p_rot_local_exp->getImag() * p_rcc_inner->getImag())));
                            p_local_exp->incImag(
                                ((p_rot_local_exp->getReal() * p_rcc_inner->getImag()) +
                                 (p_rot_local_exp->getImag() * p_rcc_inner->getReal())));

                            ++p_rot_local_exp;
                    ++p_rcc_inner;
                } /* for k */


                ++p_local_exp;
            } /* for nu */
        } /* for j */

        const FComplex<FReal>* FRestrict p_rot_local_exp_j = &rot_local_exp[Parent::harmonic.getPreExpRedirJ(Q) + Q];

        for(int j = Q + 1; j <= Parent::devP ; ++j){
            p_rot_local_exp_j += Parent::devP - j +1;
            const int min_j = Parent::devP-j;

            for(int nu = 0 ; nu <= j; ++nu){
                const FComplex<FReal>* FRestrict p_rcc_inner = &rcc_inner[j][RotationM2LTransfer::getRotationCoefP(j, nu, -min_j)];
                const FComplex<FReal>* FRestrict p_rot_local_exp = p_rot_local_exp_j;
                FReal minus_1_pow_k = FReal(min_j&1 ? -1 : 1);

                for(int k = -min_j ; k < 0; ++k){  /* k < 0 */
                    p_local_exp->incReal( minus_1_pow_k *
                                          ((p_rot_local_exp->getReal() * p_rcc_inner->getReal()) +
                                           (p_rot_local_exp->getImag() * p_rcc_inner->getImag())));
                            p_local_exp->incImag( minus_1_pow_k *
                                                  ((p_rot_local_exp->getReal() * p_rcc_inner->getImag()) -
                                                   (p_rot_local_exp->getImag() * p_rcc_inner->getReal())));

                            minus_1_pow_k = -minus_1_pow_k;
                    --p_rot_local_exp;
                    ++p_rcc_inner;
                } /* for k */
                for(int k = 0; k<=min_j; ++k){  /* k >= 0 */
                    p_local_exp->incReal(
                                ((p_rot_local_exp->getReal() * p_rcc_inner->getReal()) -
                                 (p_rot_local_exp->getImag() * p_rcc_inner->getImag())));
                            p_local_exp->incImag(
                                ((p_rot_local_exp->getReal() * p_rcc_inner->getImag()) +
                                 (p_rot_local_exp->getImag() * p_rcc_inner->getReal())));

                            ++p_rot_local_exp;
                    ++p_rcc_inner;
                } /* for k */
                ++p_local_exp;
            } /* for nu */
        } /* for j */
    }
};



#endif // FSPHERICALROTATIONKERNEL_HPP
