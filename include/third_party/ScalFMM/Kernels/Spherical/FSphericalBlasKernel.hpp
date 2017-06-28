// ===================================================================================
// Copyright ScalFmm 2011 INRIA
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
#ifndef FSPHERICALBLASKERNEL_HPP
#define FSPHERICALBLASKERNEL_HPP

#include "FAbstractSphericalKernel.hpp"

#include "Utils/FMemUtils.hpp"
#include "Utils/FBlas.hpp"

/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * This class is a spherical harmonic kernels using blas
 */
template< class FReal, class CellClass, class ContainerClass>
class FSphericalBlasKernel : public FAbstractSphericalKernel<FReal,CellClass,ContainerClass> {
protected:
    typedef FAbstractSphericalKernel<FReal,CellClass,ContainerClass> Parent;

	const int FF_MATRIX_ROW_DIM;     //< The blas matrix number of rows
	const int FF_MATRIX_COLUMN_DIM;  //< The blas matrix number of columns
	const int FF_MATRIX_SIZE;        //< The blas matrix size

    FComplex<FReal>* temporaryMultiSource;               //< To perform the M2L without allocating at each call
    FSmartPointer<FComplex<FReal>**> preM2LTransitions;   //< The pre-computation for the M2L based on the level and the 189 possibilities

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
			    //
			    //   Single Layer
			    //
			    //			    std::cout << std::endl ;
                FComplex<FReal> Czero(0.,0.);
			    int idxRow =   0 ;
			    for(int j =0 ; j<= Parent::devP  ; ++j ){ // Row
				for(int k=0;    k<=j  ; ++k){                  // Row
				    int idxCol = 0 ;
				    for(int n =0 ; n<= Parent::devP  ; ++n){ // Col
					for(int l=-n; l<=n ; ++l){
					    //										std::cout << " j " << j << " k " << k << " n " <<n << " ,j+n  " << j+n  << " l " << l		  << " idxRow " << idxRow		  << " idxCol " << idxCol		<< std::endl;
					    if ( j+n <=Parent::devP){
						// Col
						++idxCol;
					    }
					    else {
						matrix[idxCol * FF_MATRIX_ROW_DIM + idxRow] = Czero;
						++idxCol;
					    }

					}
				    }
				    ++idxRow  ;
				}
			    }      //
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
    FSphericalBlasKernel(const int inDevP, const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter)
: Parent(inDevP, inTreeHeight, inBoxWidth, inBoxCenter),
  FF_MATRIX_ROW_DIM(Parent::harmonic.getExpSize()), FF_MATRIX_COLUMN_DIM(Parent::harmonic.getNExpSize()),
  FF_MATRIX_SIZE(FF_MATRIX_ROW_DIM * FF_MATRIX_COLUMN_DIM),
  temporaryMultiSource(new FComplex<FReal>[FF_MATRIX_COLUMN_DIM]),
  preM2LTransitions(nullptr){
		allocAndInit();
	}

	/** Copy constructor */
	FSphericalBlasKernel(const FSphericalBlasKernel& other)
	: Parent(other),
	  FF_MATRIX_ROW_DIM(other.FF_MATRIX_ROW_DIM), FF_MATRIX_COLUMN_DIM(other.FF_MATRIX_COLUMN_DIM),
	  FF_MATRIX_SIZE(other.FF_MATRIX_SIZE),
      temporaryMultiSource(new FComplex<FReal>[FF_MATRIX_COLUMN_DIM]),
	  preM2LTransitions(other.preM2LTransitions) {

	}

	/** Destructor */
	~FSphericalBlasKernel(){
		delete[] temporaryMultiSource;
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
            const FComplex<FReal>* const transitionVector = preM2LTransitions[inLevel][idxNeigh];
            multipoleToLocal(inLocal->getLocal(), distantNeighbors[idxExistingNeigh]->getMultipole(), transitionVector);
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
	 * for(int idxRow = 0 ; idxRow < FF_MATRIX_ROW_DIM ; ++idxRow){
	 *       for(int idxCol = 0 ; idxCol < FF_MATRIX_COLUMN_DIM ; ++idxCol){
	 *           local_exp[idxRow].addMul(M2L_Outer_transfer[idxCol * FF_MATRIX_ROW_DIM + idxRow], temporaryMultiSource[idxCol]);
	 *       }
	 *   }
	 */
    void multipoleToLocal(FComplex<FReal>*const FRestrict local_exp, const FComplex<FReal>* const FRestrict multipole_exp_src,
            const FComplex<FReal>* const FRestrict M2L_Outer_transfer){
		// Copy original vector and compute exp2nexp
        FMemUtils::copyall<FComplex<FReal>>(temporaryMultiSource, multipole_exp_src, CellClass::GetPoleSize());
		// Get a computable vector
		preExpNExp(temporaryMultiSource);

		const FReal one[2] = {1.0 , 0.0};

		FBlas::c_gemva(
				FF_MATRIX_ROW_DIM,
				FF_MATRIX_COLUMN_DIM,
				one,
                FComplex<FReal>::ToFReal(M2L_Outer_transfer),
                FComplex<FReal>::ToFReal(temporaryMultiSource),
                FComplex<FReal>::ToFReal(local_exp));

// #ifdef DEBUG_SPHERICAL_M2L

//		std::cout << "\n ====== Multipole expansion ====== \n"<<std::endl;

//		for(int idxCol =0 ;idxCol< FF_MATRIX_COLUMN_DIM ; ++idxCol ){ // Col
//			std::cout << temporaryMultiSource[idxCol] <<" ";
//		}
//		std::cout << std::endl ;

//		std::cout << "\n ====== MultipolToLocal MatrixTransfer ====== \n"<<std::endl;
//		std::cout << "    FF_MATRIX_ROW_DIM:        " << FF_MATRIX_ROW_DIM<<std::endl;
//		std::cout << "    FF_MATRIX_COLUMN_DIM: " << FF_MATRIX_COLUMN_DIM<<std::endl;
//		for(int idxRow =0 ;idxRow< FF_MATRIX_ROW_DIM ; ++idxRow ){ // Row
//			std::cout << "Row="<<idxRow<<" : " ;
//			for(int idxCol =0 ;idxCol< FF_MATRIX_COLUMN_DIM ; ++idxCol ){ // Col
//				std::cout << M2L_Outer_transfer[idxCol * FF_MATRIX_ROW_DIM + idxRow] <<" ";
//			}
//			std::cout << std::endl ;
//		}
//		std::cout << std::endl ;

//		std::cout << "\n ====== Local expansion ====== \n"<<std::endl;

//		for(int idxCol =0 ;idxCol< FF_MATRIX_ROW_DIM ; ++idxCol ){ // Col
//			std::cout << local_exp[idxCol] <<" ";
//		}
//		std::cout << std::endl ;

//		std::cout << "============================ \n"<<std::endl;

// #endif
	}
};

#endif // FSPHERICALBLASKERNEL_HPP
