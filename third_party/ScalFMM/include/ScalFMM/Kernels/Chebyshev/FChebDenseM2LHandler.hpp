// See LICENCE file at project root

#ifndef FCHEBDENSEM2LHANDLER_HPP
#define FCHEBDENSEM2LHANDLER_HPP

#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>
#include <typeinfo>

#include "Utils/FBlas.hpp"
#include "Utils/FTic.hpp"

#include "FChebTensor.hpp"


/**
 * @author Matthias Messner (matthias.messner@inria.fr)
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FChebDenseM2LHandler
 * Please read the license
 *
 * This class precomputes the M2L operators
 * \f$[K_1,\dots,K_{316}]\f$ for all (\f$7^3-3^3 = 316\f$ possible interacting
 * cells in the far-field) interactions for the Chebyshev interpolation
 * approach.
 *
 * PB: FChebDenseM2LHandler does not seem to support non_homogeneous kernels!
 * In fact nothing appears to handle this here (i.e., adapt scaling and storage
 * to MatrixKernelClass::Type). Given the relatively important cost of the
 * Chebyshev variant, it is probably a choice not to have implemented this
 * feature here but instead in the ChebyshevSym variant. But what if the
 * kernel is non homogeneous and non symmetric (e.g. Dislocations)...
 *
 * TODO Specialize class (see UnifM2LHandler) OR prevent from using this
 * class with non homogeneous kernels ?!
 *
 * @tparam ORDER interpolation order \f$\ell\f$
 */
template <class FReal, int ORDER, class MatrixKernelClass>
class FChebDenseM2LHandler : FNoCopyable
{
	enum {order = ORDER,
				nnodes = TensorTraits<ORDER>::nnodes,
				ninteractions = 316}; // 7^3 - 3^3 (max num cells in far-field)

	const MatrixKernelClass *const MatrixKernel;

	FReal *C;
	unsigned int rank;   //<! truncation rank, satisfies @p epsilon


	static const std::string getFileName()
	{
		const char precision_type = (typeid(FReal)==typeid(double) ? 'd' : 'f');
		std::stringstream stream;
		stream << "m2l_k"<< MatrixKernelClass::getID() << "_" << precision_type
					 << "_o" << order << ".bin";
		return stream.str();
	}


public:
	FChebDenseM2LHandler(const MatrixKernelClass *const inMatrixKernel, const FReal dummy)
		: MatrixKernel(inMatrixKernel), C(nullptr), rank(0)
	{}

	~FChebDenseM2LHandler()
	{
		if (C != nullptr) delete [] C;
	}

	/**
	 * Computes and sets the matrices \f$C_t\f$
	 */
	void ComputeAndCompressAndSet()
	{
		// measure time
		FTic time; time.tic();
		// check if aready set
		if (C) throw std::runtime_error("Full M2L operator already set");
		rank = Compute(MatrixKernel, C);

	    unsigned long sizeM2L = 343*rank*rank*sizeof(FReal);

		// write info
		std::cout << "Compute and set full M2L operators (" << long(sizeM2L) << " B) in "
							<< time.tacAndElapsed() << "sec."	<< std::endl;
	}

	/**
	 * Computes, compresses, writes to binary file, reads it and sets the matrices \f$Y, C_t, B\f$
	 */
	void ComputeAndStoreInBinaryFileAndReadFromFileAndSet()
	{
        FChebDenseM2LHandler<FReal, ORDER,MatrixKernelClass>::ComputeAndStoreInBinaryFile();
		this->ReadFromBinaryFileAndSet();
	}

	/**
	 * Computes and compressed all \f$K_t\f$.
	 *
	 * @param[in] MatrixKernel kernel function evaluator
	 * @param[out] C matrix of size \f$r\times 316 r\f$ storing \f$[C_1,\dots,C_{316}]\f$
	 */
	static unsigned int Compute(const MatrixKernelClass *const MatrixKernel, FReal* &C);

	/**
	 * Computes and stores the matrices \f$C_t\f$ in a binary
	 * file
	 */
	static void ComputeAndStoreInBinaryFile(const MatrixKernelClass *const MatrixKernel);

	/**
	 * Reads the matrices \f$C_t\f$ from the respective binary file
	 */
	void ReadFromBinaryFileAndSet();


	/**
	 * @return rank of the M2L operators
	 */
	unsigned int getRank() const {return rank;}


  /**
	 * Full M2L operation \f$X+=C_tY\f$, where \f$Y\f$ is the compressed
	 * multipole expansion and \f$X\f$ is the compressed local expansion, both
	 * of size \f$r\f$. The index \f$t\f$ denotes the transfer vector of the
	 * target cell to the source cell.
	 *
	 * @param[in] transfer transfer vector
	 * @param[in] Y compressed multipole expansion
	 * @param[out] X compressed local expansion
	 * @param[in] CellWidth needed for the scaling of the compressed M2L operators which are based on a homogeneous matrix kernel computed for the reference cell width \f$w=2\f$, ie in \f$[-1,1]^3\f$.
	 */
  void applyC(const int transfer[3], FReal CellWidth,
			  const FReal *const Y, FReal *const X) const
  {
	const unsigned int idx
		= (transfer[0]+3)*7*7 + (transfer[1]+3)*7 + (transfer[2]+3);
	const FReal scale(MatrixKernel->getScaleFactor(CellWidth));
    FBlas::gemva(rank, rank, scale, C + idx*rank*rank, const_cast<FReal*>(Y), X);
  }
  void applyC(const unsigned int idx, FReal CellWidth,
			  const FReal *const Y, FReal *const X) const
  {
	const FReal scale(MatrixKernel->getScaleFactor(CellWidth));
    FBlas::gemva(rank, rank, scale, C + idx*rank*rank, const_cast<FReal*>(Y), X);
  }
  void applyC(FReal CellWidth,
			  const FReal *const Y, FReal *const X) const
  {
	const FReal scale(MatrixKernel->getScaleFactor(CellWidth));
    FBlas::gemva(rank, rank * 343, scale, C, const_cast<FReal*>(Y), X);
  }

};






//////////////////////////////////////////////////////////////////////
// definition ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////






template <class FReal, int ORDER, class MatrixKernelClass>
unsigned int
FChebDenseM2LHandler<FReal, ORDER, MatrixKernelClass>::Compute(const MatrixKernelClass *const MatrixKernel, FReal* &C)
{
	// allocate memory and store compressed M2L operators
	if (C) throw std::runtime_error("Full M2L operators are already set");

	// interpolation points of source (Y) and target (X) cell
    FPoint<FReal> X[nnodes], Y[nnodes];
	// set roots of target cell (X)
    FChebTensor<FReal, order>::setRoots(FPoint<FReal>(0.,0.,0.), FReal(2.), X);

	// allocate memory and compute 316 m2l operators
	FReal *_C;
	_C = new FReal [nnodes*nnodes * ninteractions];
	unsigned int counter = 0;
	for (int i=-3; i<=3; ++i) {
		for (int j=-3; j<=3; ++j) {
			for (int k=-3; k<=3; ++k) {
				if (abs(i)>1 || abs(j)>1 || abs(k)>1) {
					// set roots of source cell (Y)
                    const FPoint<FReal> cy(FReal(2.*i), FReal(2.*j), FReal(2.*k));
                    FChebTensor<FReal, order>::setRoots(cy, FReal(2.), Y);
					// evaluate m2l operator
					for (unsigned int n=0; n<nnodes; ++n)
						for (unsigned int m=0; m<nnodes; ++m)
							_C[counter*nnodes*nnodes + n*nnodes + m]
								= MatrixKernel->evaluate(X[m], Y[n]);
					// increment interaction counter
					counter++;
				}
			}
		}
	}
	if (counter != ninteractions)
		throw std::runtime_error("Number of interactions must correspond to 316");

	// svd compression of M2L
    const unsigned int rank	= nnodes;

	// store C
	counter = 0;
	C = new FReal [343 * rank*rank];
	for (int i=-3; i<=3; ++i)
		for (int j=-3; j<=3; ++j)
			for (int k=-3; k<=3; ++k) {
				const unsigned int idx = (i+3)*7*7 + (j+3)*7 + (k+3);
				if (abs(i)>1 || abs(j)>1 || abs(k)>1) {
					FBlas::copy(rank*rank, _C + counter*rank*rank, C + idx*rank*rank);
					counter++;
				} else FBlas::setzero(rank*rank, C + idx*rank*rank);
			}
	if (counter != ninteractions)
		throw std::runtime_error("Number of interactions must correspond to 316");
	delete [] _C;

	// return low rank
	return rank;
}






template <class FReal, int ORDER, class MatrixKernelClass>
void
FChebDenseM2LHandler<FReal, ORDER, MatrixKernelClass>::ComputeAndStoreInBinaryFile(const MatrixKernelClass *const MatrixKernel)
{
	// measure time
	FTic time; time.tic();
	// start computing process
	FReal *C;
	C = nullptr;
	const unsigned int rank = nnodes;
	// store into binary file
	const std::string filename(getFileName());
	std::ofstream stream(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
	if (stream.good()) {
		stream.seekp(0);
		// 1) write number of interpolation points (int)
		int _nnodes = nnodes;
		stream.write(reinterpret_cast<char*>(&_nnodes), sizeof(int));
		// 2) write low rank (int)
		int _rank = rank;
		stream.write(reinterpret_cast<char*>(&_rank), sizeof(int));
		// 5) write 343 C (343 * rank*rank * FReal)
		stream.write(reinterpret_cast<char*>(C), sizeof(FReal)*rank*rank*343);
	} 	else throw std::runtime_error("File could not be opened to write");
	stream.close();
	// free memory
	if (C != nullptr) delete [] C;
	// write info
	std::cout << "Full M2L operators ("<< rank << ") stored in binary file "	<< filename
						<< " in " << time.tacAndElapsed() << "sec."	<< std::endl;
}


template <class FReal, int ORDER, class MatrixKernelClass>
void
FChebDenseM2LHandler<FReal, ORDER, MatrixKernelClass>::ReadFromBinaryFileAndSet()
{
	// measure time
	FTic time; time.tic();
	// start reading process
	if (C) throw std::runtime_error("Full M2L operator already set");
	const std::string filename(getFileName());
	std::ifstream stream(filename.c_str(),
											 std::ios::in | std::ios::binary | std::ios::ate);
	const std::ifstream::pos_type size = stream.tellg();
	if (size<=0) {
		std::cout << "Info: The requested binary file " << filename
							<< " does not yet exist. Compute it now ... " << std::endl;
		this->ComputeAndStoreInBinaryFileAndReadFromFileAndSet();
		return;
	}
	if (stream.good()) {
		stream.seekg(0);
		// 1) read number of interpolation points (int)
		int npts;
		stream.read(reinterpret_cast<char*>(&npts), sizeof(int));
		if (npts!=nnodes) throw std::runtime_error("nnodes and npts do not correspond");
		// 2) read low rank (int)
		stream.read(reinterpret_cast<char*>(&rank), sizeof(int));
		// 5) write 343 C (343 * rank*rank * FReal)
		C = new FReal [343 * rank*rank];
		stream.read(reinterpret_cast<char*>(C), sizeof(FReal)*rank*rank*343);
	}	else throw std::runtime_error("File could not be opened to read");
	stream.close();
	// write info
	std::cout << "Full M2L operators (" << rank << ") read from binary file "
						<< filename << " in " << time.tacAndElapsed() << "sec."	<< std::endl;
}




#endif // FCHEBDENSEM2LHANDLER_HPP

// [--END--]
