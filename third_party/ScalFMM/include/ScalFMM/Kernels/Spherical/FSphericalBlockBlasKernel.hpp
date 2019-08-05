// See LICENCE file at project root
#ifndef FSPHERICALBLOCKBLASKERNEL_HPP
#define FSPHERICALBLOCKBLASKERNEL_HPP

#include "FAbstractSphericalKernel.hpp"

#include "../../Utils/FMemUtils.hpp"
#include "../../Utils/FBlas.hpp"
#include "../../Containers/FVector.hpp"

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* This class is a spherical harmonic kernels using block blas
*/
template< class FReal, class CellClass, class ContainerClass>
class FSphericalBlockBlasKernel : public FAbstractSphericalKernel<FReal, CellClass,ContainerClass> {
protected:
    typedef FAbstractSphericalKernel<FReal, CellClass,ContainerClass> Parent;

    /** A interaction properties */
    struct ComputationPair {
        const FComplex<FReal>* FRestrict pole;
        FComplex<FReal>* FRestrict local;
        explicit ComputationPair(const FComplex<FReal>* const inPole = 0, FComplex<FReal>*const inLocal = 0)
            : pole(inPole), local(inLocal) {}
    };

    const int FF_MATRIX_ROW_DIM;     //< The blas matrix number of rows
    const int FF_MATRIX_COLUMN_DIM;  //< The blas matrix number of columns
    const int FF_MATRIX_SIZE;        //< The blas matrix size

    const int BlockSize;             //< The size of a block

    FComplex<FReal>*const multipoleMatrix;                //< To copy all the multipole vectors
    FComplex<FReal>*const localMatrix;                    //< To save all the local vectors result
    FSmartPointer<FComplex<FReal>**> preM2LTransitions;    //< The pre-computation for the M2L based on the level and the 189 possibilities

    FVector<ComputationPair> interactions[343];     //< All the current interaction


    /** To access te precomputed M2L transfer matrixes */
    int indexM2LTransition(const int idxX,const int idxY,const int idxZ) const {
        return (( ( ((idxX+3) * 7) + (idxY+3))*7 ) + (idxZ+3));
    }


    /** Alloc and init pre-vectors*/
    void allocAndInit(){
        FHarmonic<FReal> blasHarmonic(Parent::devP * 2);

        // Matrix to fill and then transposed
        FComplex<FReal>*const workMatrix = new FComplex<FReal>[FF_MATRIX_SIZE];

        // M2L transfer, there is a maximum of 3 neighbors in each direction,
        // so 6 in each dimension
        FReal treeWidthAtLevel = Parent::boxWidth;
        preM2LTransitions = new FComplex<FReal>**[Parent::treeHeight];
        memset(preM2LTransitions.getPtr(), 0, sizeof(FComplex<FReal>**) * (Parent::treeHeight));

        for(int idxLevel = 0 ; idxLevel < Parent::treeHeight ; ++idxLevel ){
            preM2LTransitions[idxLevel] = new FComplex<FReal>*[(7 * 7 * 7)];
            memset(preM2LTransitions[idxLevel], 0, sizeof(FComplex<FReal>*) * (7*7*7));

            for(int idxX = -3 ; idxX <= 3 ; ++idxX ){
                for(int idxY = -3 ; idxY <= 3 ; ++idxY ){
                    for(int idxZ = -3 ; idxZ <= 3 ; ++idxZ ){
                        if(FMath::Abs(idxX) > 1 || FMath::Abs(idxY) > 1 || FMath::Abs(idxZ) > 1){
                            // Compute harmonic
                            const FPoint<FReal> relativePos( FReal(-idxX) * treeWidthAtLevel , FReal(-idxY) * treeWidthAtLevel , FReal(-idxZ) * treeWidthAtLevel );
                            blasHarmonic.computeOuter(FSpherical<FReal>(relativePos));

                            // Reset Matrix
                            FMemUtils::setall<FComplex<FReal>>(workMatrix, FComplex<FReal>(), FF_MATRIX_SIZE);
                            FComplex<FReal>* FRestrict fillTransfer = workMatrix;

                            for(int M = 0 ; M <= Parent::devP ; ++M){
                                for (int m = 0 ;  m <= M ; ++m){
                                    for (int N = 0 ; N <= Parent::devP ; ++N){
                                        for (int n = 0 ; n <= 2*N ;  ++n, ++fillTransfer){
                                            const int k = N-n-m;
                                            if (k < 0){
                                                const FReal pow_of_minus_1 = FReal((k&1) ? -1 : 1);
                                                fillTransfer->setReal( pow_of_minus_1 * blasHarmonic.result()[blasHarmonic.getPreExpRedirJ(M+N)-k].getReal());
                                                fillTransfer->setImag((-pow_of_minus_1) * blasHarmonic.result()[blasHarmonic.getPreExpRedirJ(M+N)-k].getImag());
                                            }
                                            else{
                                                (*fillTransfer) = blasHarmonic.result()[blasHarmonic.getPreExpRedirJ(M+N)+k];
                                            }

                                        }
                                    }
                                }
                            }

                            // Transpose and copy result
                            FComplex<FReal>*const matrix = new FComplex<FReal>[FF_MATRIX_SIZE];
                            for(int idxRow = 0 ; idxRow < FF_MATRIX_ROW_DIM ; ++idxRow){
                                for(int idxCol = 0 ; idxCol < FF_MATRIX_COLUMN_DIM ; ++idxCol){
                                    matrix[idxCol * FF_MATRIX_ROW_DIM + idxRow] = workMatrix[idxCol + idxRow * FF_MATRIX_COLUMN_DIM];
                                }
                            }
                            preM2LTransitions[idxLevel][indexM2LTransition(idxX,idxY,idxZ)] = matrix;
                        }
                    }
                }
            }
            treeWidthAtLevel /= 2;
        }

        // Clean
        delete[] workMatrix;
    }


public:
    /** Constructor
      * @param inDevP the polynomial degree
      * @param inThreeHeight the height of the tree
      * @param inBoxWidth the size of the simulation box
      * @param inPeriodicLevel the number of level upper to 0 that will be requiried
      */
    FSphericalBlockBlasKernel(const int inDevP, const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter, const int inBlockSize = 512)
        : Parent(inDevP, inTreeHeight, inBoxWidth, inBoxCenter),
          FF_MATRIX_ROW_DIM(Parent::harmonic.getExpSize()), FF_MATRIX_COLUMN_DIM(Parent::harmonic.getNExpSize()),
          FF_MATRIX_SIZE(FF_MATRIX_ROW_DIM * FF_MATRIX_COLUMN_DIM),
          BlockSize(inBlockSize),
          multipoleMatrix(new FComplex<FReal>[inBlockSize * FF_MATRIX_COLUMN_DIM]),
          localMatrix(new FComplex<FReal>[inBlockSize * FF_MATRIX_ROW_DIM]),
          preM2LTransitions(nullptr){
        allocAndInit();
    }

    /** Copy constructor */
    FSphericalBlockBlasKernel(const FSphericalBlockBlasKernel& other)
        : Parent(other),
          FF_MATRIX_ROW_DIM(other.FF_MATRIX_ROW_DIM), FF_MATRIX_COLUMN_DIM(other.FF_MATRIX_COLUMN_DIM),
          FF_MATRIX_SIZE(other.FF_MATRIX_SIZE),
          BlockSize(other.BlockSize),
          multipoleMatrix(new FComplex<FReal>[other.BlockSize * FF_MATRIX_COLUMN_DIM]),
          localMatrix(new FComplex<FReal>[other.BlockSize * FF_MATRIX_ROW_DIM]),
          preM2LTransitions(other.preM2LTransitions) {

    }

    /** Destructor */
    ~FSphericalBlockBlasKernel(){
        delete[] multipoleMatrix;
        delete[] localMatrix;
        if(preM2LTransitions.isLast()){
            for(int idxLevel = 0 ; idxLevel < Parent::treeHeight ; ++idxLevel ){
                for(int idxX = -3 ; idxX <= 3 ; ++idxX ){
                    for(int idxY = -3 ; idxY <= 3 ; ++idxY ){
                        for(int idxZ = -3 ; idxZ <= 3 ; ++idxZ ){
                            delete[] preM2LTransitions[idxLevel][indexM2LTransition(idxX,idxY,idxZ)];
                        }
                    }
                }
            }
            FMemUtils::DeleteAllArray(preM2LTransitions.getPtr(), Parent::treeHeight);
        }
    }

    /** M2L with a cell and all the existing neighbors */
    void M2L(CellClass* const FRestrict inLocal, const CellClass* distantNeighbors[],
             const int neighborPositions[], const int inSize, const int inLevel)  override {
        // For all neighbors compute M2L
        for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
            const int idxNeigh = neighborPositions[idxExistingNeigh];
            interactions[idxNeigh].push(ComputationPair(distantNeighbors[idxExistingNeigh]->getMultipole(), inLocal->getLocal()));

            if( interactions[idxNeigh].getSize() == BlockSize){
                multipoleToLocal( idxNeigh, inLevel);
            }
        }
    }
    /** the needFinishedM2LEvent method is needed   */
    constexpr static bool NeedFinishedM2LEvent(){
   	 	 return true ;
    }
    /** Do we have some computation to do in the buffers */
    void finishedLevelM2L(const int inLevel){
        for(int idxNeigh = 0 ; idxNeigh < 343 ; ++idxNeigh){
            if( interactions[idxNeigh].getSize() ){
                multipoleToLocal( idxNeigh, inLevel);
            }
        }
    }

    /** preExpNExp
      * @param exp an exponent vector to create an computable vector
      */
    void preExpNExp(FComplex<FReal>* const exp) const {
        for(int j = Parent::devP; j>= 0 ; --j){
            // Position in 'exp':  (j*(j+1)*0.5) + k
            // Position in 'nexp':  j*(j+1)      + k
            const int j_j1       = j*(j+1);
            const int j_j1_div_2 = int(j_j1 * 0.5);

            // Positive (or null) orders:
            for(int k = j ; k >= 0; --k){
                exp[j_j1 + k] = exp[j_j1_div_2 + k];
            }

            // Negative orders:
            FReal minus_1_pow_k = FReal( j&1 ? -1 : 1);
            for(int k = -j ; k < 0 ; ++k ){
                exp[j_j1 + k].setReal(minus_1_pow_k * exp[j_j1 + (-k)].getReal());
                exp[j_j1 + k].setImag((-minus_1_pow_k) * exp[j_j1 + (-k)].getImag());
                minus_1_pow_k = -minus_1_pow_k;
            }
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
    * const FComplex<FReal>*const M2LTransfer = preM2LTransitions[inLevel][interactionIndex];
     *   for(int idxK = 0 ; idxK < interactions[interactionIndex].getSize() ; ++idxK){
     *       for(int idxRow = 0 ; idxRow < FF_MATRIX_ROW_DIM ; ++idxRow){
     *           FComplex<FReal> compute;
     *           for(int idxCol = 0 ; idxCol < FF_MATRIX_COLUMN_DIM ; ++idxCol){
     *               compute.addMul(M2LTransfer[idxCol * FF_MATRIX_ROW_DIM + idxRow], multipoleMatrix[idxCol + idxK * FF_MATRIX_COLUMN_DIM]);
     *           }
     *           localMatrix[FF_MATRIX_ROW_DIM * idxK + idxRow] = compute;
     *       }
     *   }
    */
    void multipoleToLocal(const int interactionIndex, const int inLevel){
        for(int idxInter = 0 ; idxInter < interactions[interactionIndex].getSize() ; ++idxInter){
            // Copy original vector and compute exp2nexp
            FMemUtils::copyall<FComplex<FReal>>(&multipoleMatrix[idxInter * FF_MATRIX_COLUMN_DIM],
                                          interactions[interactionIndex][idxInter].pole, FF_MATRIX_COLUMN_DIM);

            // Get a computable vector
            preExpNExp(&multipoleMatrix[idxInter * FF_MATRIX_COLUMN_DIM]);
        }

        const FReal one[2] = {1.0 , 0.0};

        FBlas::c_gemm(
                    FF_MATRIX_ROW_DIM,
                    FF_MATRIX_COLUMN_DIM,
                    int(interactions[interactionIndex].getSize()),
                    one,
                    FComplex<FReal>::ToFReal(preM2LTransitions[inLevel][interactionIndex]),
                    FF_MATRIX_ROW_DIM,
                    FComplex<FReal>::ToFReal(multipoleMatrix),
                    FF_MATRIX_COLUMN_DIM,
                    FComplex<FReal>::ToFReal(localMatrix),
                    FF_MATRIX_ROW_DIM);

        for(int idxInter = 0 ; idxInter < interactions[interactionIndex].getSize() ; ++idxInter){
            FMemUtils::addall<FComplex<FReal>>(interactions[interactionIndex][idxInter].local, &localMatrix[idxInter * FF_MATRIX_ROW_DIM],  FF_MATRIX_ROW_DIM);
        }

        interactions[interactionIndex].clear();
    }
};



#endif // FSPHERICALBLOCKBLASKERNEL_HPP
