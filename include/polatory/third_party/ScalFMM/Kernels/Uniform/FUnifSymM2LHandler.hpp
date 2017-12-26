// See LICENCE file at project root
// Keep in private GIT

#ifndef FUNIFSYMM2LHANDLER_HPP
#define FUNIFSYMM2LHANDLER_HPP

#include <climits>

#include "../../Utils/FBlas.hpp"
#include "../../Utils/FDft.hpp"

#include "../../Utils/FComplex.hpp"

#include "./FUnifTensor.hpp"
#include "../Interpolation/FInterpSymmetries.hpp"
#include "./FUnifM2LHandler.hpp"

/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * Please read the license
 */


/*!  Precomputes the 16 far-field interactions (due to symmetries in their
  arrangement all 316 far-field interactions can be represented by
  permutations of the 16 we compute in this function).
 */
template < class FReal, int ORDER, typename MatrixKernelClass>
static void precompute(const MatrixKernelClass *const MatrixKernel, const FReal CellWidth,
                       FComplex<FReal>* FC[343])
{
  //	std::cout << "\nComputing 16 far-field interactions (l=" << ORDER << ", eps=" << Epsilon
  //						<< ") for cells of width w = " << CellWidth << std::endl;

	static const unsigned int nnodes = ORDER*ORDER*ORDER;

	// interpolation points of source (Y) and target (X) cell
    FPoint<FReal> X[nnodes], Y[nnodes];
	// set roots of target cell (X)
    FUnifTensor<FReal,ORDER>::setRoots(FPoint<FReal>(0.,0.,0.), CellWidth, X);
	// temporary matrices
	FReal *_C;
    FComplex<FReal> *_FC;

  // reduce storage from nnodes^2=order^6 to (2order-1)^3
  const unsigned int rc = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1);
	_C = new FReal [rc];
    _FC = new FComplex<FReal> [rc]; // TODO: do it in the non-sym version!!!

  // init Discrete Fourier Transformator
  const int dimfft = 1; // unidim FFT since fully circulant embedding
  const int steps[dimfft] = {rc};
//	FDft Dft(rc);
	FFft<FReal,dimfft> Dft(steps);

  // reduce storage if real valued kernel
  const unsigned int opt_rc = rc/2+1;

  // get first column of K via permutation
  unsigned int perm[rc];
  for(unsigned int p=0; p<rc; ++p){
    if(p<rc-1) perm[p]=p+1;
    else perm[p]=p+1-rc;
  }
  unsigned int li,lj, mi,mj, ni,nj;
  unsigned int idi, idj, ido;

	// initialize timer
	FTic time;
	double overall_time(0.);
	double elapsed_time(0.);

	unsigned int counter = 0;
	for (int i=2; i<=3; ++i) {
		for (int j=0; j<=i; ++j) {
			for (int k=0; k<=j; ++k) {

				// set roots of source cell (Y)
                const FPoint<FReal> cy(CellWidth*FReal(i), CellWidth*FReal(j), CellWidth*FReal(k));
				FUnifTensor<FReal,ORDER>::setRoots(cy, CellWidth, Y);

				// start timer
				time.tic();

        // evaluate m2l operator
        ido=0;
        for(unsigned int l=0; l<2*ORDER-1; ++l)
          for(unsigned int m=0; m<2*ORDER-1; ++m)
            for(unsigned int n=0; n<2*ORDER-1; ++n){   
          
              // l=0:(2*ORDER-1) => li-lj=-(ORDER-1):(ORDER-1)
              // Convention:
              // lj=ORDER-1 & li=0:ORDER-1 => li-lj=1-ORDER:0
              // lj=1 & li=0:ORDER-1 => li-lj=1:ORDER-1
              if(l<ORDER-1) lj=ORDER-1; else lj=0;
              if(m<ORDER-1) mj=ORDER-1; else mj=0;
              if(n<ORDER-1) nj=ORDER-1; else nj=0;
              li=(l-(ORDER-1))+lj; mi=(m-(ORDER-1))+mj; ni=(n-(ORDER-1))+nj;
              // deduce corresponding index of K[nnodes x nnodes] 
              idi=li*ORDER*ORDER + mi*ORDER + ni;
              idj=lj*ORDER*ORDER + mj*ORDER + nj;
                
              // store value at current position in C
              // use permutation if DFT is used because 
              // the storage of the first column is required
              // i.e. C[0] C[rc-1] C[rc-2] .. C[1] < WRONG!
              // i.e. C[rc-1] C[0] C[1] .. C[rc-2] < RIGHT!
//                _C[counter*rc + ido]
              _C[perm[ido]]
                = MatrixKernel->evaluate(X[idi], Y[idj]);
              ido++;
            }

        // Apply Discrete Fourier Transformation
        Dft.applyDFT(_C,_FC);

        // determine new index
				const unsigned int idx = (i+3)*7*7 + (j+3)*7 + (k+3);

				// store
				{
					// allocate
					assert(FC[idx]==NULL);
                    FC[idx] = new FComplex<FReal>[opt_rc];
          FBlas::c_copy(opt_rc, reinterpret_cast<FReal*>(_FC), 
                        reinterpret_cast<FReal*>(FC[idx]));
				}

				elapsed_time = time.tacAndElapsed(); 
				overall_time += elapsed_time;	

				counter++;
			}
		}
	}

		std::cout << "The approximation of the " << counter
              << " far-field interactions (sizeM2L= " 
              << counter*opt_rc*sizeof(FComplex<FReal>) << " B"
              << ") took " << overall_time << "s\n" << std::endl;

    // Free _C & _FC
    delete [] _C;
    delete [] _FC;
}









/*!  \class FUnifSymM2LHandler

	\brief Deals with all the symmetries in the arrangement of the far-field interactions

	Stores permutation indices and permutation vectors to reduce 316 (7^3-3^3)
  different far-field interactions to 16 only. We use the number 343 (7^3)
  because it allows us to use to associate the far-field interactions based on
  the index \f$t = 7^2(i+3) + 7(j+3) + (k+3)\f$ where \f$(i,j,k)\f$ denotes
  the relative position of the source cell to the target cell. */
template <int ORDER, KERNEL_FUNCTION_TYPE TYPE> class FUnifSymM2LHandler;

/*! Specialization for homogeneous kernel functions */
template <int ORDER>
class FUnifSymM2LHandler<ORDER, HOMOGENEOUS>
{
  static const unsigned int nnodes = ORDER*ORDER*ORDER;

	// M2L operators
    FComplex<FReal>*    K[343];

public:
	
	// permutation vectors and permutated indices
	unsigned int pvectors[343][nnodes];
	unsigned int pindices[343];


	/** Constructor: with 16 small SVDs */
	template <typename MatrixKernelClass>
	FUnifSymM2LHandler(const MatrixKernelClass *const MatrixKernel,
                     const FReal, const unsigned int)
	{
		// init all 343 item to zero, because effectively only 16 exist
		for (unsigned int t=0; t<343; ++t)
			K[t] = NULL;
			
		// set permutation vector and indices
		const FInterpSymmetries<ORDER> Symmetries;
		for (int i=-3; i<=3; ++i)
			for (int j=-3; j<=3; ++j)
				for (int k=-3; k<=3; ++k) {
					const unsigned int idx = ((i+3) * 7 + (j+3)) * 7 + (k+3);
					pindices[idx] = 0;
					if (abs(i)>1 || abs(j)>1 || abs(k)>1)
						pindices[idx] = Symmetries.getPermutationArrayAndIndex(i,j,k, pvectors[idx]);
				}

		// precompute 16 M2L operators
		const FReal ReferenceCellWidth = FReal(2.);
		precompute<ORDER>(MatrixKernel, ReferenceCellWidth, K);
	}



	/** Destructor */
	~FUnifSymM2LHandler()
	{
		for (unsigned int t=0; t<343; ++t) if (K[t]!=NULL) delete [] K[t];
	}


	/*! return the t-th approximated far-field interactions*/
    const FComplex<FReal> *const getK(const unsigned int, const unsigned int t) const
	{	return K[t]; }

};






/*! Specialization for non-homogeneous kernel functions */
template <int ORDER>
class FUnifSymM2LHandler<ORDER, NON_HOMOGENEOUS>
{
  static const unsigned int nnodes = ORDER*ORDER*ORDER;

	// Height of octree; needed only in the case of non-homogeneous kernel functions
	const unsigned int TreeHeight;

	// M2L operators for all levels in the octree
    FComplex<FReal>***    K;

public:
	
	// permutation vectors and permutated indices
	unsigned int pvectors[343][nnodes];
	unsigned int pindices[343];


	/** Constructor: with 16 small SVDs */
	template <typename MatrixKernelClass>
	FUnifSymM2LHandler(const MatrixKernelClass *const MatrixKernel,
                     const FReal RootCellWidth, const unsigned int inTreeHeight)
		: TreeHeight(inTreeHeight)
	{
		// init all 343 item to zero, because effectively only 16 exist
        K       = new FComplex<FReal>** [TreeHeight];
		K[0]       = NULL; K[1]       = NULL;
		for (unsigned int l=2; l<TreeHeight; ++l) {
            K[l]       = new FComplex<FReal>* [343];
			for (unsigned int t=0; t<343; ++t)
				K[l][t]       = NULL;
		}
		

		// set permutation vector and indices
		const FInterpSymmetries<ORDER> Symmetries;
		for (int i=-3; i<=3; ++i)
			for (int j=-3; j<=3; ++j)
				for (int k=-3; k<=3; ++k) {
					const unsigned int idx = ((i+3) * 7 + (j+3)) * 7 + (k+3);
					pindices[idx] = 0;
					if (abs(i)>1 || abs(j)>1 || abs(k)>1)
						pindices[idx] = Symmetries.getPermutationArrayAndIndex(i,j,k, pvectors[idx]);
				}

		// precompute 16 M2L operators at all levels having far-field interactions
		FReal CellWidth = RootCellWidth / FReal(2.); // at level 1
		CellWidth /= FReal(2.);                      // at level 2
		for (unsigned int l=2; l<TreeHeight; ++l) {
			precompute<ORDER>(MatrixKernel, CellWidth, K[l]);
			CellWidth /= FReal(2.);                    // at level l+1 
		}
	}



	/** Destructor */
	~FUnifSymM2LHandler()
	{
		for (unsigned int l=0; l<TreeHeight; ++l) {
			if (K[l]!=NULL) {
				for (unsigned int t=0; t<343; ++t) if (K[l][t]!=NULL) delete [] K[l][t];
				delete [] K[l];
			}
		}
		delete [] K;
	}

	/*! return the t-th approximated far-field interactions*/
    const FComplex<FReal> *const getK(const unsigned int l, const unsigned int t) const
	{	return K[l][t]; }

};








//#include <fstream>
//#include <sstream>
//
//
///**
// * Computes, compresses and stores the 16 M2L kernels in a binary file.
// */
//template <int ORDER, typename MatrixKernelClass>
//static void ComputeAndCompressAndStoreInBinaryFile(const MatrixKernelClass *const MatrixKernel, const FReal Epsilon)
//{
//	static const unsigned int nnodes = ORDER*ORDER*ORDER;
//
//	// compute and compress ////////////
//	FReal* K[343];
//	int LowRank[343];
//	for (unsigned int idx=0; idx<343; ++idx) { K[idx] = NULL; LowRank[idx] = 0;	}
//	precompute<ORDER>(MatrixKernel, FReal(2.), Epsilon, K, LowRank);
//
//	// write to binary file ////////////
//	FTic time; time.tic();
//	// start computing process
//	const char precision = (typeid(FReal)==typeid(double) ? 'd' : 'f');
//	std::stringstream sstream;
//	sstream << "sym2l_" << precision << "_o" << ORDER << "_e" << Epsilon << ".bin";
//	const std::string filename(sstream.str());
//	std::ofstream stream(filename.c_str(),
//											 std::ios::out | std::ios::binary | std::ios::trunc);
//	if (stream.good()) {
//		stream.seekp(0);
//		for (unsigned int idx=0; idx<343; ++idx)
//			if (K[idx]!=NULL) {
//				// 1) write index
//				stream.write(reinterpret_cast<char*>(&idx), sizeof(int));
//				// 2) write low rank (int)
//				int rank = LowRank[idx];
//				stream.write(reinterpret_cast<char*>(&rank), sizeof(int));
//				// 3) write U and V (both: rank*nnodes * FReal)
//				FReal *const U = K[idx];
//				FReal *const V = K[idx] + rank*nnodes;
//				stream.write(reinterpret_cast<char*>(U), sizeof(FReal)*rank*nnodes);
//				stream.write(reinterpret_cast<char*>(V), sizeof(FReal)*rank*nnodes);
//			}
//	} else throw std::runtime_error("File could not be opened to write");
//	stream.close();
//	// write info
//	//	std::cout << "Compressed M2L operators stored in binary file " << filename
//	//					<< " in " << time.tacAndElapsed() << "sec."	<< std::endl;
//
//	// free memory /////////////////////
//	for (unsigned int t=0; t<343; ++t) if (K[t]!=NULL) delete [] K[t];
//}
//
//
///**
// * Reads the 16 compressed M2L kernels from the binary files and writes them
// * in K and the respective low-rank in LowRank.
// */
//template <int ORDER>
//void ReadFromBinaryFile(const FReal Epsilon, FReal* K[343], int LowRank[343])
//{
//	// compile time constants
//	const unsigned int nnodes = ORDER*ORDER*ORDER;
//	
//	// find filename
//	const char precision = (typeid(FReal)==typeid(double) ? 'd' : 'f');
//	std::stringstream sstream;
//	sstream << "sym2l_" << precision << "_o" << ORDER << "_e" << Epsilon << ".bin";
//	const std::string filename(sstream.str());
//
//	// read binary file
//	std::ifstream istream(filename.c_str(),
//												std::ios::in | std::ios::binary | std::ios::ate);
//	const std::ifstream::pos_type size = istream.tellg();
//	if (size<=0) throw std::runtime_error("The requested binary file does not yet exist. Exit.");
//	
//	if (istream.good()) {
//		istream.seekg(0);
//		// 1) read index (int)
//		int _idx;
//		istream.read(reinterpret_cast<char*>(&_idx), sizeof(int));
//		// loop to find 16 compressed m2l operators
//		for (int idx=0; idx<343; ++idx) {
//			K[idx] = NULL;
//			LowRank[idx] = 0;
//			// if it exists
//			if (idx == _idx) {
//				// 2) read low rank (int)
//				int rank;
//				istream.read(reinterpret_cast<char*>(&rank), sizeof(int));
//				LowRank[idx] = rank;
//				// 3) read U and V (both: rank*nnodes * FReal)
//				K[idx] = new FReal [2*rank*nnodes];
//				FReal *const U = K[idx];
//				FReal *const V = K[idx] + rank*nnodes;
//				istream.read(reinterpret_cast<char*>(U), sizeof(FReal)*rank*nnodes);
//				istream.read(reinterpret_cast<char*>(V), sizeof(FReal)*rank*nnodes);
//
//				// 1) read next index
//				istream.read(reinterpret_cast<char*>(&_idx), sizeof(int));
//			}
//		}
//	}	else throw std::runtime_error("File could not be opened to read");
//	istream.close();
//}





#endif
