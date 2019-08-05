// See LICENCE file at project root
#ifndef FSPHERICALKERNEL_HPP
#define FSPHERICALKERNEL_HPP

#include "FAbstractSphericalKernel.hpp"
#include "../../Utils/FMemUtils.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* This class is the basic spherical harmonic kernel
*/
template< class FReal, class CellClass, class ContainerClass>
class FSphericalKernel : public FAbstractSphericalKernel<FReal, CellClass,ContainerClass> {
protected:
    typedef FAbstractSphericalKernel<FReal, CellClass,ContainerClass> Parent;

    const int devM2lP;               //< A secondary P

    FSmartPointer<FComplex<FReal>*> preM2LTransitions;   //< The pre-computation for the M2L based on the level and the 189 possibilities

    /** To access te pre computed M2L transfer vector */
    int indexM2LTransition(const int idxX,const int idxY,const int idxZ) const {
        return (((((idxX+3) * 7) + (idxY+3)) * 7 ) + (idxZ+3)) * devM2lP;
    }

    /** Alloc and init pre-vectors*/
    void allocAndInit(){
        // M2L transfer, there is a maximum of 3 neighbors in each direction,
        // so 6 in each dimension
        preM2LTransitions = new FComplex<FReal>*[Parent::treeHeight];
        memset(preM2LTransitions.getPtr(), 0, sizeof(FComplex<FReal>*) * (Parent::treeHeight));
        // We start from the higher level
        FReal treeWidthAtLevel = Parent::boxWidth;
        for(int idxLevel = 0 ; idxLevel < Parent::treeHeight ; ++idxLevel ){
            // Allocate data for this level
            preM2LTransitions[idxLevel] = new FComplex<FReal>[(7 * 7 * 7) * devM2lP];
            // Precompute transfer vector
            for(int idxX = -3 ; idxX <= 3 ; ++idxX ){
                for(int idxY = -3 ; idxY <= 3 ; ++idxY ){
                    for(int idxZ = -3 ; idxZ <= 3 ; ++idxZ ){
                        if(FMath::Abs(idxX) > 1 || FMath::Abs(idxY) > 1 || FMath::Abs(idxZ) > 1){
                            const FPoint<FReal> relativePos( FReal(-idxX) * treeWidthAtLevel , FReal(-idxY) * treeWidthAtLevel , FReal(-idxZ) * treeWidthAtLevel );
                            Parent::harmonic.computeOuter(FSpherical<FReal>(relativePos));
                            FMemUtils::copyall<FComplex<FReal>>(&preM2LTransitions[idxLevel][indexM2LTransition(idxX,idxY,idxZ)], Parent::harmonic.result(), Parent::harmonic.getExpSize());
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
    FSphericalKernel(const int inDevP, const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter)
        : Parent(inDevP, inTreeHeight, inBoxWidth, inBoxCenter),
          devM2lP(int(((inDevP*2)+1) * ((inDevP*2)+2) * 0.5)),
          preM2LTransitions(nullptr) {
        allocAndInit();
    }

    /** Copy constructor */
    FSphericalKernel(const FSphericalKernel& other)
        : Parent(other), devM2lP(other.devM2lP),
          preM2LTransitions(other.preM2LTransitions) {

    }

    /** Destructor */
    ~FSphericalKernel(){
        if( preM2LTransitions.isLast() ){
            FMemUtils::DeleteAllArray(preM2LTransitions.getPtr(), Parent::treeHeight);
        }
    }

    /** M2L with a cell and all the existing neighbors */
    void M2L(CellClass* const FRestrict inLocal, const CellClass* distantNeighbors[],
             const int neighborPositions[], const int inSize, const int inLevel)  override {
        // For all neighbors compute M2L
        for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
            const int idxNeigh = neighborPositions[idxExistingNeigh];
            const FComplex<FReal>* const transitionVector = &preM2LTransitions[inLevel][idxNeigh * devM2lP];
            multipoleToLocal(inLocal->getLocal(), distantNeighbors[idxExistingNeigh]->getMultipole(), transitionVector);
        }
    }


    /** M2L
    *We compute the conversion of multipole_exp_src in *p_center_of_exp_src to
    *a local expansion in *p_center_of_exp_target, and add the result to local_exp_target.
    *
    *O_n^l (with n=0..P, l=-n..n) being the former multipole expansion terms
    *(whose center is *p_center_of_multipole_exp_src) we have for the new local
    *expansion terms (whose center is *p_center_of_local_exp_target):
    *
    *L_j^k = sum{n=0..+}
    *sum{l=-n..n}
    *O_n^l Outer_{j+n}^{-k-l}(rho, alpha, beta)
    *
    *where (rho, alpha, beta) are the spherical coordinates of the vector :
    *p_center_of_local_exp_src - *p_center_of_multipole_exp_target
    *
    *Remark: here we have always j+n >= |-k-l|
      *
      */
    void multipoleToLocal(FComplex<FReal>*const FRestrict local_exp, const FComplex<FReal>* const FRestrict multipole_exp_src,
                          const FComplex<FReal>* const FRestrict M2L_Outer_transfer){
        int index_j_k = 0;

        // L_j^k
        // HPMSTART(51, "M2L computation (loops)");
        // j from 0 to P
        for (int j = 0 ; j <= Parent::devP ; ++j){
            // (-1)^k
            FReal pow_of_minus_1_for_k = 1.0;
            //k from 0 to j
            for (int k = 0 ; k <= j ; ++k, ++index_j_k){
                // (-1)^n
                FReal pow_of_minus_1_for_n = 1.0;

                // work with a local variable
                FComplex<FReal> L_j_k = local_exp[index_j_k];
                // n from 0 to P or do P-j
                //for (int n = 0 ; n <= Parent::devP /*or*/ /*Parent::devP-j*/ ; ++n){
                for (int n = 0 ; n <= Parent::devP-j ; ++n){  // faster than double height Parent::devP
                    // O_n^l : here points on the source multipole expansion term of degree n and order |l|
                    const int index_n = Parent::harmonic.getPreExpRedirJ(n);

                    // Outer_{j+n}^{-k-l} : here points on the M2L transfer function/expansion term of degree j+n and order |-k-l|
                    const int index_n_j = Parent::harmonic.getPreExpRedirJ(n+j);

                    FReal pow_of_minus_1_for_l = pow_of_minus_1_for_n; // (-1)^l

                    // We start with l=n (and not l=-n) so that we always set p_Outer_term to a correct value in the first loop.
                    int l = n;
                    for(/* l = n */ ; l > 0 ; --l){ // we have -k-l<0 and l>0
                        const FComplex<FReal> M_n_l = multipole_exp_src[index_n + l];
                        const FComplex<FReal> O_n_j__k_l = M2L_Outer_transfer[index_n_j + k + l];

                        L_j_k.incReal( pow_of_minus_1_for_l * pow_of_minus_1_for_k *
                                       ((M_n_l.getReal() * O_n_j__k_l.getReal()) +
                                        (M_n_l.getImag() * O_n_j__k_l.getImag())));
                        L_j_k.incImag( pow_of_minus_1_for_l * pow_of_minus_1_for_k *
                                       ((M_n_l.getImag() * O_n_j__k_l.getReal()) -
                                        (M_n_l.getReal() * O_n_j__k_l.getImag())));

                        pow_of_minus_1_for_l = -pow_of_minus_1_for_l;
                    }

                    for(/* l = 0 */; l >= -n &&  (-k-l) < 0 ; --l){ // we have -k-l<0 and l<=0
                        const FComplex<FReal> M_n_l = multipole_exp_src[index_n - l];
                        const FComplex<FReal> O_n_j__k_l = M2L_Outer_transfer[index_n_j + k + l];

                        L_j_k.incReal( pow_of_minus_1_for_k *
                                       ((M_n_l.getReal() * O_n_j__k_l.getReal()) -
                                        (M_n_l.getImag() * O_n_j__k_l.getImag())));
                        L_j_k.decImag(  pow_of_minus_1_for_k *
                                        ((M_n_l.getImag() * O_n_j__k_l.getReal()) +
                                         (M_n_l.getReal() * O_n_j__k_l.getImag())));

                        pow_of_minus_1_for_l = -pow_of_minus_1_for_l;
                    }

                    for(/*l = -n-1 or l = -k-1 */; l >= -n ; --l){ // we have -k-l>=0 and l<=0
                        const FComplex<FReal> M_n_l = multipole_exp_src[index_n - l];
                        const FComplex<FReal> O_n_j__k_l = M2L_Outer_transfer[index_n_j - (k + l)];

                        L_j_k.incReal( pow_of_minus_1_for_l *
                                       ((M_n_l.getReal() * O_n_j__k_l.getReal()) +
                                        (M_n_l.getImag() * O_n_j__k_l.getImag())));
                        L_j_k.incImag( pow_of_minus_1_for_l *
                                       ((M_n_l.getReal() * O_n_j__k_l.getImag()) -
                                        (M_n_l.getImag() * O_n_j__k_l.getReal())));

                        pow_of_minus_1_for_l = -pow_of_minus_1_for_l;
                    }

                    pow_of_minus_1_for_n = -pow_of_minus_1_for_n;
                }//n

                // put in the local vector
                local_exp[index_j_k] = L_j_k;

                pow_of_minus_1_for_k = -pow_of_minus_1_for_k;
            }//k
        }
    }
};

#endif // FSPHERICALKERNEL_HPP
