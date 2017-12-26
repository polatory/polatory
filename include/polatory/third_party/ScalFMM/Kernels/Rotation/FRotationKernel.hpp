// See LICENCE file at project root
#ifndef FROTATIONKERNEL_HPP
#define FROTATIONKERNEL_HPP

#include "Components/FAbstractKernels.hpp"
#include "Utils/FSmartPointer.hpp"
#include "Utils/FComplex.hpp"
#include "Utils/FMemUtils.hpp"
#include "Utils/FSpherical.hpp"
#include "Utils/FMath.hpp"
#include "Utils/FAssert.hpp"

#include "../P2P/FP2PR.hpp"

/** This is a recursion to get the minimal size of the matrix dlmk
  */
template<int N> struct NumberOfValuesInDlmk{
    enum {Value = (N*2+1)*(N*2+1) + NumberOfValuesInDlmk<N - 1>::Value};
};
template<> struct NumberOfValuesInDlmk<0>{
    enum {Value = 1};
};

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FRotationKernel
* @brief
*
* This kernels is a complete rotation based kernel with spherical
* harmonic.
*
* Here is the optimizated kernel, please refer to FRotationOriginalKernel
* to see the non optimized easy to understand kernel.
*/
template<class FReal, class CellClass, class ContainerClass, int P>
class FRotationKernel : public FAbstractKernels<CellClass,ContainerClass> {

    //< Size of the data array computed using a suite relation
    static const int SizeArray = ((P+2)*(P+1))/2;
    //< To have P*2 where needed
    static const int P2 = P*2;

    ///////////////////////////////////////////////////////
    // Object attributes
    ///////////////////////////////////////////////////////

    const FReal boxWidth;               //< the box width at leaf level
    const int   treeHeight;             //< The height of the tree
    const FReal widthAtLeafLevel;       //< width of box at leaf level
    const FReal widthAtLeafLevelDiv2;   //< width of box at leaf leve div 2
    const FPoint<FReal> boxCorner;             //< position of the box corner

    FReal factorials[P2+1];             //< This contains the factorial until 2*P+1

    ///////////// Translation /////////////////////////////
    FSmartPointer<FReal[P+1]>      M2MTranslationCoef;  //< This contains some precalculated values for M2M translation
    FSmartPointer<FReal[343][P+1]> M2LTranslationCoef;  //< This contains some precalculated values for M2L translation
    FSmartPointer<FReal[P+1]>      L2LTranslationCoef;  //< This contains some precalculated values for L2L translation

    ///////////// Rotation    /////////////////////////////
    FComplex<FReal> rotationExpMinusImPhi[8][SizeArray];  //< This is the vector use for the rotation around z for the M2M (multipole)
    FComplex<FReal> rotationExpImPhi[8][SizeArray];       //< This is the vector use for the rotation around z for the L2L (taylor)

    FComplex<FReal> rotationM2LExpMinusImPhi[343][SizeArray]; //< This is the vector use for the rotation around z for the M2L (multipole)
    FComplex<FReal> rotationM2LExpImPhi[343][SizeArray];      //< This is the vector use for the rotation around z for the M2L (taylor)

    ///////////// Rotation    /////////////////////////////
    // First we compute the size of the d{l,m,k} matrix.

    static const int SizeDlmkMatrix = NumberOfValuesInDlmk<P>::Value;

    FReal DlmkCoefOTheta[8][SizeDlmkMatrix];        //< d_lmk for Multipole rotation
    FReal DlmkCoefOMinusTheta[8][SizeDlmkMatrix];   //< d_lmk for Multipole reverse rotation

    FReal DlmkCoefMTheta[8][SizeDlmkMatrix];        //< d_lmk for Local rotation
    FReal DlmkCoefMMinusTheta[8][SizeDlmkMatrix];   //< d_lmk for Local reverse rotation

    FReal DlmkCoefM2LOTheta[343][SizeDlmkMatrix];       //< d_lmk for Multipole rotation
    FReal DlmkCoefM2LMMinusTheta[343][SizeDlmkMatrix];  //< d_lmk for Local reverse rotation

    ///////////////////////////////////////////////////////
    // Precomputation
    ///////////////////////////////////////////////////////

    /** Compute the factorial from 0 to P*2
      * Then the data is accessible in factorials array:
      * factorials[n] = n! with n <= 2*P
      */
    void precomputeFactorials(){
        factorials[0] = 1;
        FReal fidx = 1;
        for(int idx = 1 ; idx <= P2 ; ++idx, ++fidx){
            factorials[idx] = fidx * factorials[idx-1];
        }
    }

    /** This function precompute the translation coef.
      * Translation are independant of the angle between both cells.
      * So in the M2M/L2L the translation is the same for all children.
      * In the M2L the translation depend on the distance between the
      * source and the target (so a few number of possibilities exist)
      *
      * The number of possible translation depend of the tree height,
      * so the memory is allocated dynamically with a smart pointer to share
      * the data between threads.
      */
    void precomputeTranslationCoef(){
        {// M2M & L2L
            // Allocate
            M2MTranslationCoef = new FReal[treeHeight-1][P+1];
            L2LTranslationCoef = new FReal[treeHeight-1][P+1];
            // widthAtLevel represents half of the size of a box
            FReal widthAtLevel = boxWidth/4;
            // we go from the root to the leaf-1
            for( int idxLevel = 0 ; idxLevel < treeHeight - 1 ; ++idxLevel){
                // b is the parent-child distance = norm( vec(widthAtLevel,widthAtLevel,widthAtLevel))
                const FReal b = FMath::Sqrt(widthAtLevel*widthAtLevel*3);
                // we compute b^idx iteratively
                FReal bPowIdx = 1.0;
                // we compute -1^idx iteratively
                FReal minus_1_pow_idx = 1.0;
                for(int idx = 0 ; idx <= P ; ++idx){
                    // coef m2m = (-b)^j/j!
                    M2MTranslationCoef[idxLevel][idx] = minus_1_pow_idx * bPowIdx / factorials[idx];
                    // coef l2l = b^j/j!
                    L2LTranslationCoef[idxLevel][idx] = bPowIdx / factorials[idx];
                    // increase
                    bPowIdx *= b;
                    minus_1_pow_idx = -minus_1_pow_idx;
                }
                // divide by two per level
                widthAtLevel /= 2;
            }
        }
        {// M2L
            // Allocate
            M2LTranslationCoef = new FReal[treeHeight][343][P+1];
            // This is the width of a box at each level
            FReal boxWidthAtLevel = widthAtLeafLevel;
            // from leaf level to the root
            for(int idxLevel = treeHeight-1 ; idxLevel > 0 ; --idxLevel){
                // we compute all possibilities
                for(int idxX = -3 ; idxX <= 3 ; ++idxX ){
                    for(int idxY = -3 ; idxY <= 3 ; ++idxY ){
                        for(int idxZ = -3 ; idxZ <= 3 ; ++idxZ ){
                            // if this is not a neighbour
                            if( idxX < -1 || 1 < idxX || idxY < -1 || 1 < idxY || idxZ < -1 || 1 < idxZ ){
                                // compute the relative position
                                const FPoint<FReal> relativePosition( -FReal(idxX)*boxWidthAtLevel,
                                                                      -FReal(idxY)*boxWidthAtLevel,
                                                                      -FReal(idxZ)*boxWidthAtLevel);
                                // this is the position in the index system from 0 to 343
                                const int position = ((( (idxX+3) * 7) + (idxY+3))) * 7 + idxZ + 3;
                                // b is the distance between the two cells
                                const FReal b = FMath::Sqrt( (relativePosition.getX() * relativePosition.getX()) +
                                                             (relativePosition.getY() * relativePosition.getY()) +
                                                             (relativePosition.getZ() * relativePosition.getZ()));
                                // compute b^idx+1 iteratively
                                FReal bPowIdx1 = b;
                                for(int idx = 0 ; idx <= P ; ++idx){
                                    // factorials[j+l] / FMath::pow(b,j+l+1)
                                    M2LTranslationCoef[idxLevel][position][idx] = factorials[idx] / bPowIdx1;
                                    bPowIdx1 *= b;
                                }
                            }
                        }
                    }
                }
                // multiply per two at each level
                boxWidthAtLevel *= FReal(2.0);
            }
        }
    }

    ///////////////////////////////////////////////////////
    // Precomputation rotation vector
    // This is a all in one function
    // First we compute the d_lmk needed,
    // then we compute vectors for M2M/L2L
    // finally we compute the vectors for M2L
    ///////////////////////////////////////////////////////


    /** The following comments include formula taken from the original vectors
      *
      *
      * This function rotate a multipole vector by an angle azimuth phi
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1
      * \f[
      * O_{l,m}( \alpha, \beta + \phi ) = e^{-i \phi m} O_{l,m}( \alpha, \beta )
      * \f]
      * The computation is simply a multiplication per a complex number \f$ e^{-i \phi m} \f$
      * Phi should be in [0,2pi]
      *
      * This function rotate a local vector by an angle azimuth phi
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1
      * \f[
      * M_{l,m}( \alpha, \beta + \phi ) = e^{i \phi m} M_{l,m}( \alpha, \beta )
      * \f]
      * The computation is simply a multiplication per a complex number \f$ e^{i \phi m} \f$
      * Phi should be in [0,2pi]
      *
      * This function rotate a multipole vector by an angle inclination \theta
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
      *
      * This function rotate a local vector by an angle inclination \theta
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
      *
      * Remark about the structure of the structure of the matrixes DlmkCoef[O/M](Minus)Theta.
      * It is composed of "P" small matrix.
      * The matrix M(l) (0 <= l <= P) has a size of (l*2+1)
      * It means indexes are going from -l to l for column and row.
      * l = 0: ( -0 <= m <= 0 ; -0 <= k <= 0)
      * [X]
      * l = 1: ( -1 <= m <= 1 ; -1 <= k <= 1)
      * [X X X]
      * [X X X]
      * [X X X]
      * etc.
      * The real size of such matrix is :
      * 1x1 + 3x3 + ... + (2P+1)x(2P+1)
      */
    void precomputeRotationVectors(){
        /////////////////////////////////////////////////////////////////
        // We will need a Sqrt(factorial[x-y]*factorial[x+y])
        // so we precompute it
        FReal sqrtDoubleFactorials[P+1][P+1];
        for(int l = 0 ; l <= P ; ++l ){
            for(int m = 0 ; m <= l ; ++m ){
                sqrtDoubleFactorials[l][m] = FMath::Sqrt(factorials[l-m]*factorials[l+m]);
            }
        }

        /////////////////////////////////////////////////////////////////
        // We compute the rotation matrix, we do not need 343 matrix
        // We will compute only a part of the since we compute the inclinaison
        // angle. inclinaison(+/-x,+/-y,z) = inclinaison(+/-y,+/-x,z)
        // we put the negative (-theta) with a negative x
        typedef FReal (*pMatrixDlmk) /*[P+1]*/[P2+1][P2+1];
        pMatrixDlmk dlmkMatrix[7][4][7];
        // Allocate matrix
        for(int idxX = 0 ; idxX < 7 ; ++idxX)
            for(int idxY = 0 ; idxY < 4 ; ++idxY)
                for(int idxZ = 0 ; idxZ < 7 ; ++idxZ) {
                    dlmkMatrix[idxX][idxY][idxZ] = new FReal[P+1][P2+1][P2+1];
                }

        // First we compute special vectors:
        DlmkBuild0(dlmkMatrix[0+3][0][1+3]);    // theta = 0
        DlmkBuildPi(dlmkMatrix[0+3][0][-1+3]);  // theta = Pi
        DlmkBuild(dlmkMatrix[1+3][0][0+3],FMath::FPiDiv2<FReal>());              // theta = Pi/2
        DlmkInverse(dlmkMatrix[-1+3][0][0+3],dlmkMatrix[1+3][0][0+3]);  // theta = -Pi/2
        // Then other angle
        for(int x = 1 ; x <= 3 ; ++x){
            for(int y = 0 ; y <= x ; ++y){
                for(int z = 1 ; z <= 3 ; ++z){
                    const FReal inclinaison = FSpherical<FReal>(FPoint<FReal>(FReal(x),FReal(y),FReal(z))).getInclination();
                    DlmkBuild(dlmkMatrix[x+3][y][z+3],inclinaison);
                    // For inclinaison between ]pi/2;pi[
                    DlmkZNegative(dlmkMatrix[x+3][y][(-z)+3],dlmkMatrix[x+3][y][z+3]);
                    // For inclinaison between ]pi;3pi/2[
                    DlmkInverseZNegative(dlmkMatrix[(-x)+3][y][(-z)+3],dlmkMatrix[x+3][y][z+3]);
                    // For inclinaison between ]3pi/2;2pi[
                    DlmkInverse(dlmkMatrix[(-x)+3][y][z+3],dlmkMatrix[x+3][y][z+3]);
                }
            }
        }

        /////////////////////////////////////////////////////////////////
        // Manage angle for M2M/L2L

        const int index_P0 = atLm(P,0);
        // For all possible child (morton indexing from 0 to 7)
        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            // Retrieve relative position of child to parent
            const FReal x = FReal((idxChild&4)? -boxWidth : boxWidth);
            const FReal y = FReal((idxChild&2)? -boxWidth : boxWidth);
            const FReal z = FReal((idxChild&1)? -boxWidth : boxWidth);
            const FPoint<FReal> relativePosition( x , y , z );
            // compute azimuth
            const FSpherical<FReal> sph(relativePosition);

            // First compute azimuth rotation
            // compute the last part with l == P
            {
                int index_lm = index_P0;
                for(int m = 0 ; m <= P ; ++m, ++index_lm ){
                    const FReal mphi = (sph.getPhiZero2Pi() + FMath::FPiDiv2<FReal>()) * FReal(m);
                    // O_{l,m}( \alpha, \beta + \phi ) = e^{-i \phi m} O_{l,m}( \alpha, \beta )
                    rotationExpMinusImPhi[idxChild][index_lm].setRealImag(FMath::Cos(-mphi), FMath::Sin(-mphi));
                    // M_{l,m}( \alpha, \beta + \phi ) = e^{i \phi m} M_{l,m}( \alpha, \beta )
                    rotationExpImPhi[idxChild][index_lm].setRealImag(FMath::Cos(mphi), FMath::Sin(mphi));
                }
            }
            // Then for l < P it just a copy of the previous computed vector
            {
                int index_lm = 0;
                // for l < P
                for(int l = 0 ; l < P ; ++l){
                    // take the l + 1 numbers from the vector with l' = P
                    FMemUtils::copyall(rotationExpMinusImPhi[idxChild] + index_lm,
                                       rotationExpMinusImPhi[idxChild] + index_P0,
                                       l + 1);
                    FMemUtils::copyall(rotationExpImPhi[idxChild] + index_lm,
                                       rotationExpImPhi[idxChild] + index_P0,
                                       l + 1);
                    // index(l+1,0) = index(l,0) + l + 1
                    index_lm += l + 1;
                }
            }
            { // Then compute the inclinaison rotation
                // For the child parent relation we always have a inclinaison
                // for (1,1,1) or (1,1,-1)
                const int dx = 1;
                const int dy = 1;
                const int dz = (idxChild&1)?-1:1;

                //
                int index_lmk = 0;
                for(int l = 0 ; l <= P ; ++l){
                    for(int m = 0 ; m <= l ; ++m ){
                        { // for k == 0
                            const FReal d_lmk_minusTheta = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][0+P];
                            const FReal d_lmk            = dlmkMatrix[dx+3][dy][dz+3][l][m+P][0+P];
                            // \sqrt{ \frac{(l-k)!(l+k)!}{(l-|m|)!(l+|m|)!} }
                            const FReal Ofactor = sqrtDoubleFactorials[l][0]/sqrtDoubleFactorials[l][m];
                            const FReal Mfactor = sqrtDoubleFactorials[l][m]/sqrtDoubleFactorials[l][0];

                            DlmkCoefOTheta[idxChild][index_lmk]      = Ofactor * d_lmk;
                            DlmkCoefMTheta[idxChild][index_lmk]      = Mfactor * d_lmk;
                            DlmkCoefOMinusTheta[idxChild][index_lmk] = Ofactor * d_lmk_minusTheta;
                            DlmkCoefMMinusTheta[idxChild][index_lmk] = Mfactor * d_lmk_minusTheta;

                            ++index_lmk;
                        }
                        // for 0 < k
                        FReal minus_1_pow_k = -1.0;
                        for(int k = 1 ; k <= l ; ++k){
                            const FReal d_lm_minus_k            = dlmkMatrix[dx+3][dy][dz+3][l][m+P][-k+P];
                            const FReal d_lmk                   = dlmkMatrix[dx+3][dy][dz+3][l][m+P][k+P];
                            const FReal d_lm_minus_k_minusTheta = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][-k+P];
                            const FReal d_lmk_minusTheta        = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][k+P];

                            const FReal Ofactor = sqrtDoubleFactorials[l][k]/sqrtDoubleFactorials[l][m];
                            const FReal Mfactor = sqrtDoubleFactorials[l][m]/sqrtDoubleFactorials[l][k];

                            // for k negatif
                            DlmkCoefOTheta[idxChild][index_lmk]      = Ofactor * (d_lmk + minus_1_pow_k * d_lm_minus_k);
                            DlmkCoefMTheta[idxChild][index_lmk]      = Mfactor * (d_lmk + minus_1_pow_k * d_lm_minus_k);
                            DlmkCoefOMinusTheta[idxChild][index_lmk] = Ofactor * (d_lmk_minusTheta + minus_1_pow_k * d_lm_minus_k_minusTheta);
                            DlmkCoefMMinusTheta[idxChild][index_lmk] = Mfactor * (d_lmk_minusTheta + minus_1_pow_k * d_lm_minus_k_minusTheta);
                            ++index_lmk;
                            // for k positif
                            DlmkCoefOTheta[idxChild][index_lmk]      = Ofactor * (d_lmk - minus_1_pow_k * d_lm_minus_k);
                            DlmkCoefMTheta[idxChild][index_lmk]      = Mfactor * (d_lmk - minus_1_pow_k * d_lm_minus_k);
                            DlmkCoefOMinusTheta[idxChild][index_lmk] = Ofactor * (d_lmk_minusTheta - minus_1_pow_k * d_lm_minus_k_minusTheta);
                            DlmkCoefMMinusTheta[idxChild][index_lmk] = Mfactor * (d_lmk_minusTheta - minus_1_pow_k * d_lm_minus_k_minusTheta);
                            ++index_lmk;

                            minus_1_pow_k = -minus_1_pow_k;
                        }
                    }
                }
            }
        }

        /////////////////////////////////////////////////////////////////
        // Manage angle for M2L
        // For all possible cases
        for(int idxX = -3 ; idxX <= 3 ; ++idxX ){
            for(int idxY = -3 ; idxY <= 3 ; ++idxY ){
                for(int idxZ = -3 ; idxZ <= 3 ; ++idxZ ){
                    // Test if it is not a neighbors
                    if( idxX < -1 || 1 < idxX || idxY < -1 || 1 < idxY || idxZ < -1 || 1 < idxZ ){
                        // Build relative position between target and source
                        const FPoint<FReal> relativePosition( -FReal(idxX)*boxWidth,
                                                              -FReal(idxY)*boxWidth,
                                                              -FReal(idxZ)*boxWidth);
                        const int position = ((( (idxX+3) * 7) + (idxY+3))) * 7 + idxZ + 3;
                        const FSpherical<FReal> sph(relativePosition);

                        // Compute azimuth rotation vector
                        // first compute the last part with l == P
                        {
                            int index_lm = index_P0;
                            for(int m = 0 ; m <= P ; ++m, ++index_lm ){
                                const FReal mphi = (sph.getPhiZero2Pi() + FMath::FPiDiv2<FReal>()) * FReal(m);
                                // O_{l,m}( \alpha, \beta + \phi ) = e^{-i \phi m} O_{l,m}( \alpha, \beta )
                                rotationM2LExpMinusImPhi[position][index_lm].setRealImag(FMath::Cos(-mphi), FMath::Sin(-mphi));
                                // M_{l,m}( \alpha, \beta + \phi ) = e^{i \phi m} M_{l,m}( \alpha, \beta )
                                rotationM2LExpImPhi[position][index_lm].setRealImag(FMath::Cos(mphi), FMath::Sin(mphi));
                            }
                        }
                        // Then for l < P copy the subpart of the previous vector
                        {
                            int index_lm = 0;
                            for(int l = 0 ; l < P ; ++l){
                                FMemUtils::copyall(rotationM2LExpMinusImPhi[position] + index_lm,
                                                   rotationM2LExpMinusImPhi[position] + index_P0,
                                                   l + 1);
                                FMemUtils::copyall(rotationM2LExpImPhi[position] + index_lm,
                                                   rotationM2LExpImPhi[position] + index_P0,
                                                   l + 1);
                                index_lm += l + 1;
                            }
                        }
                        // Compute inclination vector
                        {
                            // We have to find the right d_lmk matrix
                            int dx = 0 , dy = 0, dz = 0;
                            // if x == 0 && y == 0 it means we have an inclination of 0 or PI
                            if(idxX == 0 && idxY == 0){
                                dx = 0;
                                dy = 0;
                                // no matter if z is big, we want [0][0][1] or [0][0][-1]
                                if( idxZ < 0 ) dz = 1;
                                else dz = -1;
                            }
                            // if z == 0 we have an inclination of Pi/2
                            else if ( idxZ == 0){
                                dx = 1;
                                dy = 0;
                                dz = 0;
                            }
                            // else we take the right indexes
                            else {
                                dx = FMath::Max(FMath::Abs(idxX),FMath::Abs(idxY));
                                dy = FMath::Min(FMath::Abs(idxX),FMath::Abs(idxY));
                                dz = -idxZ;
                            }

                            int index_lmk = 0;
                            for(int l = 0 ; l <= P ; ++l){
                                for(int m = 0 ; m <= l ; ++m ){
                                    { // k == 0
                                        const FReal d_lmk            = dlmkMatrix[dx+3][dy][dz+3][l][m+P][0+P];
                                        const FReal d_lmk_minusTheta = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][0+P];

                                        // \sqrt{ \frac{(l-k)!(l+k)!}{(l-|m|)!(l+|m|)!} }
                                        const FReal Ofactor = sqrtDoubleFactorials[l][0]/sqrtDoubleFactorials[l][m];
                                        const FReal Mfactor = sqrtDoubleFactorials[l][m]/sqrtDoubleFactorials[l][0];

                                        DlmkCoefM2LOTheta[position][index_lmk]      = Ofactor * d_lmk;
                                        DlmkCoefM2LMMinusTheta[position][index_lmk] = Mfactor * d_lmk_minusTheta;
                                        ++index_lmk;
                                    }
                                    FReal minus_1_pow_k = -1.0;
                                    for(int k = 1 ; k <= l ; ++k){
                                        const FReal d_lm_minus_k            = dlmkMatrix[dx+3][dy][dz+3][l][m+P][-k+P];
                                        const FReal d_lmk                   = dlmkMatrix[dx+3][dy][dz+3][l][m+P][k+P];

                                        const FReal d_lm_minus_k_minusTheta = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][-k+P];
                                        const FReal d_lmk_minusTheta        = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][k+P];

                                        const FReal Ofactor = sqrtDoubleFactorials[l][k]/sqrtDoubleFactorials[l][m];
                                        const FReal Mfactor = sqrtDoubleFactorials[l][m]/sqrtDoubleFactorials[l][k];

                                        DlmkCoefM2LOTheta[position][index_lmk]      = Ofactor * (d_lmk + minus_1_pow_k * d_lm_minus_k);
                                        DlmkCoefM2LMMinusTheta[position][index_lmk] = Mfactor * (d_lmk_minusTheta + minus_1_pow_k * d_lm_minus_k_minusTheta);
                                        ++index_lmk;

                                        DlmkCoefM2LOTheta[position][index_lmk]      = Ofactor * (d_lmk - minus_1_pow_k * d_lm_minus_k);
                                        DlmkCoefM2LMMinusTheta[position][index_lmk] = Mfactor * (d_lmk_minusTheta - minus_1_pow_k * d_lm_minus_k_minusTheta);
                                        ++index_lmk;

                                        minus_1_pow_k = -minus_1_pow_k;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Deallocate matrix
        for(int idxX = 0 ; idxX < 7 ; ++idxX)
            for(int idxY = 0 ; idxY < 4 ; ++idxY)
                for(int idxZ = 0 ; idxZ < 7 ; ++idxZ) {
                    delete[] dlmkMatrix[idxX][idxY][idxZ];
                }
    }



    ///////////////////////////////////////////////////////
    // d_lmk computation
    // This part is constitued of 6 functions :
    // DlmkBuild computes the matrix from a angle ]0;pi/2]
    // DlmkBuild0 computes the matrix for angle 0
    // DlmkBuildPi computes the matrix for angle pi
    // Then, others use the d_lmk to build a rotated matrix:
    // DlmkZNegative computes for angle \theta ]pi/2;pi[ using d_lmk(pi- \theta)
    // DlmkInverseZNegative computes for angle \theta ]pi;3pi/2[ using d_lmk(\theta-pi)
    // DlmkInverse computes for angle \theta ]3pi/2;2pi[ using d_lmk(2pi- \theta)
    ///////////////////////////////////////////////////////

    /** Compute d_mlk for \theta = 0
      * \f[
      * d^l_{m,k}( \theta ) = \delta_{m,k,} \,\, \mbox{\textrm{ $\delta$ Kronecker symbol }}
      * \f]
      */
    void DlmkBuild0(FReal dlmk[P+1][P2+1][P2+1]) const {
        for(int l = 0 ; l <= P ; ++l){
            for(int m = -l ; m <= l ; ++m){
                // first put 0 every where
                for(int k = -l ; k <= l ; ++k){
                    dlmk[l][P+m][P+k] = FReal(0.0);
                }
                // then replace per 1 for m == k
                dlmk[l][P+m][P+m] = FReal(1.0);
            }
        }
    }

    /** Compute d_mlk for \theta = PI
      * \f[
      * d^l_{m,k}( \theta ) = (-1)^{l+k} \delta_{m,k},\,\, \mbox{\textrm{ $\delta$ Kronecker delta } }
      * \f]
      */
    void DlmkBuildPi(FReal dlmk[P+1][P2+1][P2+1]) const {
        for(int l = 0 ; l <= P ; ++l){
            for(int m = -l ; m <= l ; ++m){
                // put 0 every where
                for(int k = -l ; k <= l ; ++k){
                    dlmk[l][P+m][P+k] = FReal(0.0);
                }
                // -1^l+k * 1 where m == k
                dlmk[l][P+m][P-m] = ((l+m)&0x1 ? FReal(-1) : FReal(1));
            }
        }
    }

    /** Compute d_mlk for \theta = ]PI/2;PI[
      * \f[
      * d^l_{m,k}( \theta ) = (-1)^{l+m} d^l_{m,-k}( \Pi - \theta )
      * \f]
      */
    void DlmkZNegative(FReal dlmk[P+1][P2+1][P2+1], const FReal dlmkZPositif[P+1][P2+1][P2+1]) const {
        for(int l = 0 ; l <= P ; ++l){
            for(int m = -l ; m <= l ; ++m){
                // if l+m is odd
                if( (l+m)&0x1 ){
                    // put -1 every where
                    for(int k = -l ; k <= l ; ++k){
                        dlmk[l][P+m][P+k] = -dlmkZPositif[l][P+m][P-k];
                    }
                }
                else{
                    // else just copy
                    for(int k = -l ; k <= l ; ++k){
                        dlmk[l][P+m][P+k] = dlmkZPositif[l][P+m][P-k];
                    }
                }
            }
        }
    }

    /** Compute d_mlk for \theta = ]PI;3PI/2[
      * \f[
      * d^l_{m,k}( \theta ) = (-1)^{l+m} d^l_{-m,k}( \theta - \Pi )
      * \f]
      */
    void DlmkInverseZNegative(FReal dlmk[P+1][P2+1][P2+1], const FReal dlmkZPositif[P+1][P2+1][P2+1]) const {
        for(int l = 0 ; l <= P ; ++l){
            for(int m = -l ; m <= l ; ++m){
                if( (l+m)&0x1 ){
                    for(int k = -l ; k <= l ; ++k){
                        dlmk[l][P+m][P+k] = -dlmkZPositif[l][P-m][P+k];
                    }
                }
                else{
                    for(int k = -l ; k <= l ; ++k){
                        dlmk[l][P+m][P+k] = dlmkZPositif[l][P-m][P+k];
                    }
                }
            }
        }
    }

    /** Compute d_mlk for \theta = ]3PI/2;2PI[
      * \f[
      * d^l_{m,k}( \theta ) = (-1)^{m+k} d^l_{m,k}( 2 \Pi - \theta )
      * \f]
      */
    void DlmkInverse(FReal dlmk[P+1][P2+1][P2+1], const FReal dlmkZPositif[P+1][P2+1][P2+1]) const {
        for(int l = 0 ; l <= P ; ++l){
            for(int m = -l ; m <= l ; ++m){
                // we start with k == -l, so if (k+m) is odd
                if( (l+m)&0x1 ){
                    // then we start per (-1)
                    for(int k = -l ; k < l ; k+=2){
                        dlmk[l][P+m][P+k] = -dlmkZPositif[l][P+m][P+k];
                        dlmk[l][P+m][P+k+1] = dlmkZPositif[l][P+m][P+k+1];
                    }
                    // l is always odd
                    dlmk[l][P+m][P+l] = -dlmkZPositif[l][P+m][P+l];
                }
                else{
                    // else we start per (+1)
                    for(int k = -l ; k < l ; k+=2){
                        dlmk[l][P+m][P+k] = dlmkZPositif[l][P+m][P+k];
                        dlmk[l][P+m][P+k+1] = -dlmkZPositif[l][P+m][P+k+1];
                    }
                    // l is always odd
                    dlmk[l][P+m][P+l] = dlmkZPositif[l][P+m][P+l];
                }
            }
        }
    }


    /** Compute d_mlk for \theta = ]0;PI/2[
      * This used the second formula from the paper:
      * Fast and accurate determination of the wigner rotation matrices in FMM
      *
      * We use formula 28,29 to compute "g"
      * Then 25,26,27 for the recurrence.
      */
    // theta should be between [0;pi] as the inclinaison angle
    void DlmkBuild(FReal dlmk[P+1][P2+1][P2+1], const FReal inTheta) const {
        FAssertLF(0 <= inTheta && inTheta < FMath::FTwoPi<FReal>());
        // To have constants for very used values
        const FReal F0 = FReal(0.0);
        const FReal F1 = FReal(1.0);
        const FReal F2 = FReal(2.0);

        const FReal cosTheta = FMath::Cos(inTheta);
        const FReal sinTheta = FMath::Sin(inTheta);

        // First compute g
        FReal g[SizeArray];
        {// Equ 29
            // g{0,0} = 1
            g[0] = F1;

            // g{l,0} = sqrt( (2l - 1) / 2l) g{l-1,0}  for l > 0
            {
                int index_l0 = 1;
                FReal fl = F1;
                for(int l = 1; l <= P ; ++l, ++fl ){
                    g[index_l0] = FMath::Sqrt((fl*F2-F1)/(fl*F2)) * g[index_l0-l];
                    index_l0 += l + 1;
                }
            }
            // g{l,m} = sqrt( (l - m + 1) / (l+m)) g{l,m-1}  for l > 0, 0 < m <= l
            {
                int index_lm = 2;
                FReal fl = F1;
                for(int l = 1; l <= P ; ++l, ++fl ){
                    FReal fm = F1;
                    for(int m = 1; m <= l ; ++m, ++index_lm, ++fm ){
                        g[index_lm] = FMath::Sqrt((fl-fm+F1)/(fl+fm)) * g[index_lm-1];
                    }
                    ++index_lm;
                }
            }
        }
        { // initial condition
            // Equ 28
            // d{l,m,l} = -1^(l+m) g{l,m} (1+cos(theta))^m sin(theta)^(l-m) , For l > 0, 0 <= m <= l
            int index_lm = 0;
            FReal sinTheta_pow_l = F1;
            for(int l = 0 ; l <= P ; ++l){
                // build variable iteratively
                FReal minus_1_pow_lm = l&0x1 ? FReal(-1) : FReal(1);
                FReal cosTheta_1_pow_m = F1;
                FReal sinTheta_pow_l_minus_m = sinTheta_pow_l;
                for(int m = 0 ; m <= l ; ++m, ++index_lm){
                    dlmk[l][P+m][P+l] = minus_1_pow_lm * g[index_lm] * cosTheta_1_pow_m * sinTheta_pow_l_minus_m;
                    // update
                    minus_1_pow_lm = -minus_1_pow_lm;
                    cosTheta_1_pow_m *= F1 + cosTheta;
                    sinTheta_pow_l_minus_m /= sinTheta;
                }
                // update
                sinTheta_pow_l *= sinTheta;
            }
        }
        { // build the rest of the matrix
            FReal fl = F1;
            for(int l = 1 ; l <= P ; ++l, ++fl){
                FReal fk = fl;
                for(int k = l ; k > -l ; --k, --fk){
                    // Equ 25
                    // For l > 0, 0 <= m < l, -l < k <= l, cos(theta) >= 0
                    // d{l,m,k-1} = sqrt( l(l+1) - m(m+1) / l(l+1) - k(k-1)) d{l,m+1,k}
                    //            + (m+k) sin(theta) d{l,m,k} / sqrt(l(l+1) - k(k-1)) (1+cos(theta))
                    FReal fm = F0;
                    for(int m = 0 ; m < l ; ++m, ++fm){
                        dlmk[l][P+m][P+k-1] =
                                (FMath::Sqrt((fl*(fl+F1)-fm*(fm+F1))/(fl*(fl+F1)-fk*(fk-F1))) * dlmk[l][P+m+1][P+k])
                                + ((fm+fk)*sinTheta*dlmk[l][P+m][P+k]/(FMath::Sqrt(fl*(fl+F1)-fk*(fk-F1))*(F1+cosTheta)));
                    }
                    // Equ 26
                    // For l > 0, -l < k <= l, cos(theta) >= 0
                    // d{l,l,k-1} = (l+k) sin(theta) d{l,l,k}
                    //             / sqrt(l(l+1)-k(k-1)) (1+cos(theta))
                    dlmk[l][P+l][P+k-1] = (fl+fk)*sinTheta*dlmk[l][P+l][P+k]/(FMath::Sqrt(fl*(fl+F1)-fk*(fk-F1))*(F1+cosTheta));
                }
                // Equ 27
                // d{l,m,k} = -1^(m+k) d{l,-m,-k}  , For l > 0, -l <= m < 0, -l <= k <= l
                for(int m = -l ; m < 0 ; ++m){
                    FReal minus_1_pow_mk = (m-l)&0x1 ? FReal(-1) : FReal(1);
                    for(int k = -l ; k <= l ; ++k){
                        dlmk[l][P+m][P+k] = minus_1_pow_mk * dlmk[l][P-m][P-k];
                        minus_1_pow_mk = -minus_1_pow_mk;
                    }
                }
            }
        }
    }

    /** Compute the legendre polynomial from {0,0} to {P,P}
      * the computation is made by recurence (P cannot be equal to 0)
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
    void computeLegendre(FReal legendre[], const FReal inCosTheta, const FReal inSinTheta) const {
        const FReal invSinTheta = -inSinTheta;

        legendre[0] = 1.0;             // P_0,0(1) = 1

        legendre[1] = inCosTheta;      // P_1,0 = cos(theta)
        legendre[2] = invSinTheta;     // P_1,1 = -sin(theta)

        // work with pointers
        FReal* FRestrict legendre_l1_m1 = legendre;     // P{l-2,m} starts with P_{0,0}
        FReal* FRestrict legendre_l1_m  = legendre + 1; // P{l-1,m} starts with P_{1,0}
        FReal* FRestrict legendre_lm  = legendre + 3;   // P{l,m} starts with P_{2,0}

        // Compute using recurrence
        FReal l2_minus_1 = 3; // 2 * l - 1
        FReal fl = FReal(2.0);// To get 'l' as a float
        for(int l = 2; l <= P ; ++l, ++fl ){
            FReal lm_minus_1 = fl - FReal(1.0); // l + m - 1
            FReal l_minus_m = fl;               // l - m
            for( int m = 0; m < l - 1 ; ++m ){
                // P_{l,m} = \frac{(2l-1) cos( \theta ) P_{l-1,m} - (l+m-1) P_{l-2,m}x}{(l-m)}
                *(legendre_lm++) = (l2_minus_1 * inCosTheta * (*legendre_l1_m++) - (lm_minus_1++) * (*legendre_l1_m1++) )
                        / (l_minus_m--);
            }
            // P_{l,l-1} = (2l-1) cos( \theta ) P_{l-1,l-1}
            *(legendre_lm++) = l2_minus_1 * inCosTheta * (*legendre_l1_m);
            // P_{l,l} = (2l-1) sin( \theta ) P_{l-1,l-1}
            *(legendre_lm++) = l2_minus_1 * invSinTheta * (*legendre_l1_m);
            // goto P_{l-1,0}
            ++legendre_l1_m;
            l2_minus_1 += FReal(2.0); // 2 * l - 1 => progress by two
        }
    }

    ///////////////////////////////////////////////////////
    // Multiplication for rotation
    // Here we have two function that are optimized
    // to compute the rotation fast!
    ///////////////////////////////////////////////////////

    /** This function use a d_lmk vector to rotate the vec
      * multipole or local vector.
      * The result is copyed in vec.
      * Please see the structure of dlmk to understand this function.
      * Warning we cast the vec FComplex<FReal> array into a FReal array
      */
    static void RotationYWithDlmk(FComplex<FReal> vec[], const FReal* dlmkCoef){
        FReal originalVec[2*SizeArray];
        FMemUtils::copyall((FComplex<FReal>*)originalVec,vec,SizeArray);
        // index_lm == atLm(l,m) but progress iteratively to write the result
        int index_lm = 0;
        for(int l = 0 ; l <= P ; ++l){
            const FReal*const FRestrict originalVecAtL0 = originalVec + (index_lm * 2);
            for(int m = 0 ; m <= l ; ++m, ++index_lm ){
                FReal res_lkm_real = 0.0;
                FReal res_lkm_imag = 0.0;
                // To read all "m" value for current "l"
                const FReal* FRestrict iterOrignalVec = originalVecAtL0;
                { // for k == 0
                    // same coef for real and imaginary
                    res_lkm_real += (*dlmkCoef) * (*iterOrignalVec++);
                    res_lkm_imag += (*dlmkCoef++) * (*iterOrignalVec++);
                }
                for(int k = 1 ; k <= l ; ++k){
                    // coef contains first real value
                    res_lkm_real += (*dlmkCoef++) * (*iterOrignalVec++);
                    // then imaginary
                    res_lkm_imag += (*dlmkCoef++) * (*iterOrignalVec++);
                }
                // save the result
                vec[index_lm].setRealImag(res_lkm_real, res_lkm_imag);
            }
        }
    }

    /** This function is computing dest[:] *= src[:]
      * it computes inSize FComplex<FReal> multiplication
      * to do so we first proceed per 4 and the the inSize%4 rest
      */
    static void RotationZVectorsMul(FComplex<FReal>* FRestrict dest, const FComplex<FReal>* FRestrict src, const int inSize = SizeArray){
        const FComplex<FReal>*const FRestrict lastElement = dest + inSize;
        const FComplex<FReal>*const FRestrict intermediateLastElement = dest + (inSize & ~0x3);
        // first the inSize - inSize%4 elements
        for(; dest != intermediateLastElement ;) {
            (*dest++) *= (*src++);
            (*dest++) *= (*src++);
            (*dest++) *= (*src++);
            (*dest++) *= (*src++);
        }
        // then the rest
        for(; dest != lastElement ;) {
            (*dest++) *= (*src++);
        }
    }

    ///////////////////////////////////////////////////////
    // Utils
    ///////////////////////////////////////////////////////


    /** Return the position of a leaf from its tree coordinate
      * This is used only for the leaf
      */
    FPoint<FReal> getLeafCenter(const FTreeCoordinate coordinate) const {
        return FPoint<FReal>(
                    FReal(coordinate.getX()) * widthAtLeafLevel + widthAtLeafLevelDiv2 + boxCorner.getX(),
                    FReal(coordinate.getY()) * widthAtLeafLevel + widthAtLeafLevelDiv2 + boxCorner.getY(),
                    FReal(coordinate.getZ()) * widthAtLeafLevel + widthAtLeafLevelDiv2 + boxCorner.getZ());
    }

    /** Return position in the array of the l/m couple
      * P[atLm(l,m)] => P{l,m}
      * 0
      * 1 2
      * 3 4 5
      * 6 7 8 9 ...
      */
    int atLm(const int l, const int m) const {
        // summation series over l + m => (l*(l+1))/2 + m
        return ((l*(l+1))>>1) + m;
    }

public:

    /** Constructor, needs system information */
    FRotationKernel( const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter) :
        boxWidth(inBoxWidth),
        treeHeight(inTreeHeight),
        widthAtLeafLevel(inBoxWidth/FReal(1 << (inTreeHeight-1))),
        widthAtLeafLevelDiv2(widthAtLeafLevel/2),
        boxCorner(inBoxCenter.getX()-(inBoxWidth/2),inBoxCenter.getY()-(inBoxWidth/2),inBoxCenter.getZ()-(inBoxWidth/2))
    {
        // simply does the precomputation
        precomputeFactorials();
        precomputeTranslationCoef();
        precomputeRotationVectors();
    }

    /** Default destructor */
    virtual ~FRotationKernel(){
    }

    /** P2M
      * The computation is based on the paper :
      * Parallelization of the fast multipole method
      * Formula number 10, page 3
      * \f[
      * \omega (q,a) = q \frac{a^{l}}{(l+|m|)!} P_{lm}(cos( \alpha ) )e^{-im \beta}
      * \f]
      */
    void P2M(CellClass* const inPole, const ContainerClass* const inParticles ) override  {
        const FReal i_pow_m[4] = {0, FMath::FPiDiv2<FReal>(), FMath::FPi<FReal>(), -FMath::FPiDiv2<FReal>()};
        // w is the multipole moment
        FComplex<FReal>* FRestrict const w = inPole->getMultipole();

        // Copying the position is faster than using cell position
        const FPoint<FReal> cellPosition = getLeafCenter(inPole->getCoordinate());

        // We need a legendre array
        FReal legendre[SizeArray];
        FReal angles[P+1][2];

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
            FReal q_aPowL = q; // To consutrct q*a^l continously
            int index_l_m = 0; // To construct the index of (l,m) continously
            FReal fl = 0.0;
            for(int l = 0 ; l <= P ; ++l, ++fl ){
                { // We need to compute the angles to use in the "m" loop
                    // So we can compute only the one needed after "l" inc
                    const FReal angle = fl * sph.getPhi() + i_pow_m[l & 0x3];
                    angles[l][0] = FMath::Cos(angle);
                    angles[l][1] = FMath::Sin(angle);
                }
                for(int m = 0 ; m <= l ; ++m, ++index_l_m){
                    const FReal magnitude = q_aPowL * legendre[index_l_m] / factorials[l+m];
                    w[index_l_m].incReal(magnitude * angles[m][0]);
                    w[index_l_m].incImag(magnitude * angles[m][1]);
                }
                q_aPowL *= a;
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
    void M2M(CellClass* const FRestrict inPole, const CellClass*const FRestrict *const FRestrict inChildren, const int inLevel) override  {
        // Get the translation coef for this level (same for all child)
        const FReal*const coef = M2MTranslationCoef[inLevel];
        // A buffer to copy the source w allocated once
        FComplex<FReal> source_w[SizeArray];
        // For all children
        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            // if child exists
            if(inChildren[idxChild]){
                // Copy the source
                FMemUtils::copyall(source_w, inChildren[idxChild]->getMultipole(), SizeArray);

                // rotate it forward
                RotationZVectorsMul(source_w,rotationExpMinusImPhi[idxChild]);
                RotationYWithDlmk(source_w,DlmkCoefOTheta[idxChild]);

                // Translate it
                FComplex<FReal> target_w[SizeArray];
                int index_lm = 0;
                for(int l = 0 ; l <= P ; ++l ){
                    for(int m = 0 ; m <= l ; ++m, ++index_lm ){
                        // w{l,m}(a+b) = sum(j=m:l, b^(l-j)/(l-j)! w{j,m}(a)
                        FReal w_lm_real = 0.0;
                        FReal w_lm_imag = 0.0;
                        int index_jm = atLm(m,m);   // get atLm(l,m)
                        int index_l_minus_j = l-m;  // get l-j continuously
                        for(int j = m ; j <= l ; ++j, --index_l_minus_j, index_jm += j ){
                            //const coef = (b^l-j) / (l-j)!;
                            w_lm_real += coef[index_l_minus_j] * source_w[index_jm].getReal();
                            w_lm_imag += coef[index_l_minus_j] * source_w[index_jm].getImag();
                        }
                        target_w[index_lm].setRealImag(w_lm_real,w_lm_imag);
                    }
                }

                // Rotate it back
                RotationYWithDlmk(target_w,DlmkCoefOMinusTheta[idxChild]);
                RotationZVectorsMul(target_w,rotationExpImPhi[idxChild]);

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
      * M_{l,m}(a-b') = \sum_{j=|m|}^{\infty}{ \frac{ (j+l)! } { b^{j+l+1} } O_{j,-m}(a) } , \mbox{\textrm{ j bounded by P-l } }
      * \f]
      * As describe in the paper, when need first to rotate the SH
      * then transfer using the formula
      * and finaly rotate back.
      */
    void M2L(CellClass* const FRestrict inLocal, const CellClass* inInteractions[], const int /*inSize*/, const int inLevel) {
        // To copy the multipole data allocated once
        FComplex<FReal> source_w[SizeArray];
        // For all children
        for(int idxNeigh = 0 ; idxNeigh < 343 ; ++idxNeigh){
            // if interaction exits
            if(inInteractions[idxNeigh]){
                const FReal*const coef = M2LTranslationCoef[inLevel][idxNeigh];
                // Copy multipole data into buffer
                FMemUtils::copyall(source_w, inInteractions[idxNeigh]->getMultipole(), SizeArray);

                // Rotate
                RotationZVectorsMul(source_w,rotationM2LExpMinusImPhi[idxNeigh]);
                RotationYWithDlmk(source_w,DlmkCoefM2LOTheta[idxNeigh]);

                // Transfer to u
                FComplex<FReal> target_u[SizeArray];
                int index_lm = 0;
                for(int l = 0 ; l <= P ; ++l ){
                    FReal minus_1_pow_m = 1.0;
                    for(int m = 0 ; m <= l ; ++m, ++index_lm ){
                        // u{l,m}(a-b) = sum(j=|m|:P-l, (j+l)!/b^(j+l+1) w{j,-m}(a)
                        FReal u_lm_real = 0.0;
                        FReal u_lm_imag = 0.0;
                        int index_jl = m + l;       // get j+l
                        int index_jm = atLm(m,m);   // get atLm(l,m)
                        for(int j = m ; j <= P-l ; ++j, ++index_jl, index_jm += j ){
                            // coef = (j+l)!/b^(j+l+1)
                            // because {l,-m} => {l,m} conjugate -1^m with -i
                            u_lm_real += minus_1_pow_m * coef[index_jl] * source_w[index_jm].getReal();
                            u_lm_imag -= minus_1_pow_m * coef[index_jl] * source_w[index_jm].getImag();
                        }
                        target_u[index_lm].setRealImag(u_lm_real,u_lm_imag);
                        minus_1_pow_m = -minus_1_pow_m;
                    }
                }

                // Rotate it back
                RotationYWithDlmk(target_u,DlmkCoefM2LMMinusTheta[idxNeigh]);
                RotationZVectorsMul(target_u,rotationM2LExpMinusImPhi[idxNeigh]);

                // Sum
                FMemUtils::addall(inLocal->getLocal(), target_u, SizeArray);
            }
        }
    }

    void M2L(CellClass* const FRestrict inLocal, const CellClass* inInteractions[],
             const int neighborPositions[], const int inSize, const int inLevel)  override {
        // To copy the multipole data allocated once
        FComplex<FReal> source_w[SizeArray];
        // For all children
        for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
            const int idxNeigh = neighborPositions[idxExistingNeigh];
            // if interaction exits
            const FReal*const coef = M2LTranslationCoef[inLevel][idxNeigh];
            // Copy multipole data into buffer
            FMemUtils::copyall(source_w, inInteractions[idxExistingNeigh]->getMultipole(), SizeArray);

            // Rotate
            RotationZVectorsMul(source_w,rotationM2LExpMinusImPhi[idxNeigh]);
            RotationYWithDlmk(source_w,DlmkCoefM2LOTheta[idxNeigh]);

            // Transfer to u
            FComplex<FReal> target_u[SizeArray];
            int index_lm = 0;
            for(int l = 0 ; l <= P ; ++l ){
                FReal minus_1_pow_m = 1.0;
                for(int m = 0 ; m <= l ; ++m, ++index_lm ){
                    // u{l,m}(a-b) = sum(j=|m|:P-l, (j+l)!/b^(j+l+1) w{j,-m}(a)
                    FReal u_lm_real = 0.0;
                    FReal u_lm_imag = 0.0;
                    int index_jl = m + l;       // get j+l
                    int index_jm = atLm(m,m);   // get atLm(l,m)
                    for(int j = m ; j <= P-l ; ++j, ++index_jl, index_jm += j ){
                        // coef = (j+l)!/b^(j+l+1)
                        // because {l,-m} => {l,m} conjugate -1^m with -i
                        u_lm_real += minus_1_pow_m * coef[index_jl] * source_w[index_jm].getReal();
                        u_lm_imag -= minus_1_pow_m * coef[index_jl] * source_w[index_jm].getImag();
                    }
                    target_u[index_lm].setRealImag(u_lm_real,u_lm_imag);
                    minus_1_pow_m = -minus_1_pow_m;
                }
            }

            // Rotate it back
            RotationYWithDlmk(target_u,DlmkCoefM2LMMinusTheta[idxNeigh]);
            RotationZVectorsMul(target_u,rotationM2LExpMinusImPhi[idxNeigh]);

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
    void L2L(const CellClass* const FRestrict inLocal, CellClass* FRestrict *const FRestrict  inChildren, const int inLevel)  override {
        // Get the translation coef for this level (same for all chidl)
        const FReal*const coef = L2LTranslationCoef[inLevel];
        // To copy the source local allocated once
        FComplex<FReal> source_u[SizeArray];
        // For all children
        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            // if child exists
            if(inChildren[idxChild]){
                // Copy the local data into the buffer
                FMemUtils::copyall(source_u, inLocal->getLocal(), SizeArray);

                // Rotate
                RotationZVectorsMul(source_u,rotationExpImPhi[idxChild]);
                RotationYWithDlmk(source_u,DlmkCoefMTheta[idxChild]);

                // Translate
                FComplex<FReal> target_u[SizeArray];
                for(int l = 0 ; l <= P ; ++l ){
                    for(int m = 0 ; m <= l ; ++m ){
                        // u{l,m}(r-b) = sum(j=0:P, b^(j-l)/(j-l)! u{j,m}(r);
                        FReal u_lm_real = 0.0;
                        FReal u_lm_imag = 0.0;
                        int index_jm = atLm(l,m);   // get atLm(j,m)
                        int index_j_minus_l = 0;    // get l-j continously
                        for(int j = l ; j <= P ; ++j, ++index_j_minus_l, index_jm += j){
                            // coef = b^j-l/j-l!
                            u_lm_real += coef[index_j_minus_l] * source_u[index_jm].getReal();
                            u_lm_imag += coef[index_j_minus_l] * source_u[index_jm].getImag();
                        }
                        target_u[atLm(l,m)].setRealImag(u_lm_real,u_lm_imag);
                    }
                }

                // Rotate
                RotationYWithDlmk(target_u,DlmkCoefMMinusTheta[idxChild]);
                RotationZVectorsMul(target_u,rotationExpMinusImPhi[idxChild]);

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
    void L2P(const CellClass* const inLocal, ContainerClass* const inParticles) override {
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

            // pre compute what is used more than once
            FReal minus_r_pow_l_div_fact_lm[SizeArray];
            FReal minus_r_pow_l_legendre_div_fact_lm[SizeArray];
            {
                int index_lm = 0;
                FReal minus_r_pow_l = 1.0;  // To get (-1*r)^l
                for(int l = 0 ; l <= P ; ++l){
                    for(int m = 0 ; m <= l ; ++m, ++index_lm){
                        minus_r_pow_l_div_fact_lm[index_lm] = minus_r_pow_l / factorials[l+m];
                        minus_r_pow_l_legendre_div_fact_lm[index_lm] = minus_r_pow_l_div_fact_lm[index_lm] * legendre[index_lm];
                    }
                    minus_r_pow_l *= -r;
                }
            }
            // pre compute what is use more than once
            FReal cos_m_phi_i_pow_m[P+1];
            FReal sin_m_phi_i_pow_m[P+1];
            {
                for(int m = 0 ; m <= P ; ++m){
                    const FReal m_phi_i_pow_m = FReal(m) * sph.getPhi() + i_pow_m[m & 0x3];
                    cos_m_phi_i_pow_m[m] = FMath::Cos(m_phi_i_pow_m);
                    sin_m_phi_i_pow_m[m] = FMath::Sin(m_phi_i_pow_m);
                }
            }

            // compute the forces
            {
                FReal Fr = 0;
                FReal FO = 0;
                FReal Fp = 0;

                int index_lm = 1;          // To get atLm(l,m), warning starts with l = 1
                FReal fl = 1.0;            // To get "l" as a float

                for(int l = 1 ; l <= P ; ++l, ++fl){
                    // first m == 0
                    {
                        Fr += fl * u[index_lm].getReal() * minus_r_pow_l_legendre_div_fact_lm[index_lm];
                    }
                    {
                        const FReal coef = minus_r_pow_l_div_fact_lm[index_lm] * (fl * (sph.getCosTheta()*legendre[index_lm]
                                                                                        - legendre[index_lm-l]) / sph.getSinTheta());
                        const FReal dI_real = coef;
                        // F(O) += 2 * Real(L dI/dO)
                        FO += u[index_lm].getReal() * dI_real;
                    }
                    ++index_lm;
                    // then 0 < m
                    for(int m = 1 ; m <= l ; ++m, ++index_lm){
                        {
                            const FReal coef = minus_r_pow_l_legendre_div_fact_lm[index_lm];
                            const FReal I_real = coef * cos_m_phi_i_pow_m[m];
                            const FReal I_imag = coef * sin_m_phi_i_pow_m[m];
                            // F(r) += 2 x l x Real(LI)
                            Fr += 2 * fl * (u[index_lm].getReal() * I_real - u[index_lm].getImag() * I_imag);
                            // F(p) += -2 x m x Imag(LI)
                            Fp -= 2 * FReal(m) * (u[index_lm].getReal() * I_imag + u[index_lm].getImag() * I_real);
                        }
                        {
                            const FReal legendre_l_minus_1 = (m == l) ? FReal(0.0) : FReal(l+m)*legendre[index_lm-l];
                            const FReal coef = minus_r_pow_l_div_fact_lm[index_lm] * ((fl * sph.getCosTheta()*legendre[index_lm]
                                                                                       - legendre_l_minus_1) / sph.getSinTheta());
                            const FReal dI_real = coef * cos_m_phi_i_pow_m[m];
                            const FReal dI_imag = coef * sin_m_phi_i_pow_m[m];
                            // F(O) += 2 * Real(L dI/dO)
                            FO += FReal(2.0) * (u[index_lm].getReal() * dI_real - u[index_lm].getImag() * dI_imag);
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
            // compute the potential
            {
                FReal magnitude = 0;
                // E = sum( l = 0:P, sum(m = -l:l, u{l,m} ))
                int index_lm = 0;
                for(int l = 0 ; l <= P ; ++l ){
                    {//for m == 0
                        // (l-|m|)! * P{l,0} / r^(l+1)
                        magnitude += u[index_lm].getReal() * minus_r_pow_l_legendre_div_fact_lm[index_lm];
                        ++index_lm;
                    }
                    for(int m = 1 ; m <= l ; ++m, ++index_lm ){
                        const FReal coef = minus_r_pow_l_legendre_div_fact_lm[index_lm];
                        const FReal I_real = coef * cos_m_phi_i_pow_m[m];
                        const FReal I_imag = coef * sin_m_phi_i_pow_m[m];
                        magnitude += FReal(2.0) * ( u[index_lm].getReal() * I_real - u[index_lm].getImag() * I_imag );
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


#endif // FROTATIONKERNEL_HPP
