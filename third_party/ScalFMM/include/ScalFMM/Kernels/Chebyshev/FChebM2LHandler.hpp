// See LICENCE file at project root
#ifndef FCHEBM2LHANDLER_HPP
#define FCHEBM2LHANDLER_HPP

#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>
#include <typeinfo>

#include "../../Utils/FBlas.hpp"
#include "../../Utils/FTic.hpp"

#include "FChebTensor.hpp"

#include "../../Utils/FSvd.hpp"

template <class FReal, int ORDER>
unsigned int Compress(const FReal epsilon, const unsigned int ninteractions,
											FReal* &U,	FReal* &C, FReal* &B);


/**
 * @author Matthias Messner (matthias.messner@inria.fr)
 * @class FChebM2LHandler
 * Please read the license
 *
 * This class precomputes and compresses the M2L operators
 * \f$[K_1,\dots,K_{316}]\f$ for all (\f$7^3-3^3 = 316\f$ possible interacting
 * cells in the far-field) interactions for the Chebyshev interpolation
 * approach. The class uses the compression via a truncated SVD and represents
 * the compressed M2L operator as \f$K_t \sim U C_t B^\top\f$ with
 * \f$t=1,\dots,316\f$. The truncation rank is denoted by \f$r\f$ and is
 * determined by the prescribed accuracy \f$\varepsilon\f$. Hence, the
 * originally \f$K_t\f$ of size \f$\ell^3\times\ell^3\f$ times \f$316\f$ for
 * all interactions is reduced to only one \f$U\f$ and one \f$B\f$, each of
 * size \f$\ell^3\times r\f$, and \f$316\f$ \f$C_t\f$, each of size \f$r\times
 * r\f$.
 *
 * PB: FChebM2LHandler does not seem to support non-homogeneous kernels!
 * In fact nothing appears to handle this here (i.e. adapt scaling and storage 
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
class FChebM2LHandler : FNoCopyable
{
	enum {order = ORDER,
				nnodes = TensorTraits<ORDER>::nnodes,
				ninteractions = 316}; // 7^3 - 3^3 (max num cells in far-field)

	const MatrixKernelClass *const MatrixKernel;

	FReal *U, *C, *B;
	const FReal epsilon; //<! accuracy which determines trucation of SVD
	unsigned int rank;   //<! truncation rank, satisfies @p epsilon


	static const std::string getFileName(FReal epsilon)
	{
		const char precision_type = (typeid(FReal)==typeid(double) ? 'd' : 'f');
		std::stringstream stream;
		stream << "m2l_k"<< MatrixKernelClass::getID() << "_" << precision_type
					 << "_o" << order << "_e" << epsilon << ".bin";
		return stream.str();
	}

	
public:
	FChebM2LHandler(const MatrixKernelClass *const inMatrixKernel, const FReal _epsilon)
		: MatrixKernel(inMatrixKernel), U(nullptr), C(nullptr), B(nullptr), epsilon(_epsilon), rank(0)
	{}

	~FChebM2LHandler()
	{
		if (U != nullptr) delete [] U;
		if (B != nullptr) delete [] B;
		if (C != nullptr) delete [] C;
	}

	/**
	 * Computes, compresses and sets the matrices \f$Y, C_t, B\f$
	 */
	void ComputeAndCompressAndSet()
	{
		// measure time
		FTic time; time.tic();
		// check if aready set
		if (U||C||B) throw std::runtime_error("Compressed M2L operator already set");
		rank = ComputeAndCompress(MatrixKernel, epsilon, U, C, B);

	    unsigned long sizeM2L = 343*rank*rank*sizeof(FReal);

		// write info
		std::cout << "Compressed and set M2L operators (" << long(sizeM2L) << " B) in "
							<< time.tacAndElapsed() << "sec."	<< std::endl;
	}

	/**
	 * Computes, compresses, writes to binary file, reads it and sets the matrices \f$Y, C_t, B\f$
	 */
	void ComputeAndCompressAndStoreInBinaryFileAndReadFromFileAndSet()
	{
        FChebM2LHandler<FReal, ORDER,MatrixKernelClass>::ComputeAndCompressAndStoreInBinaryFile(epsilon);
		this->ReadFromBinaryFileAndSet();
	}

	/**
	 * Computes and compressed all \f$K_t\f$.
	 *
	 * @param[in] epsilon accuracy
	 * @param[out] U matrix of size \f$\ell^3\times r\f$
	 * @param[out] C matrix of size \f$r\times 316 r\f$ storing \f$[C_1,\dots,C_{316}]\f$
	 * @param[out] B matrix of size \f$\ell^3\times r\f$
	 */
	static unsigned int ComputeAndCompress(const MatrixKernelClass *const MatrixKernel, const FReal epsilon, FReal* &U, FReal* &C, FReal* &B);

	/**
	 * Computes, compresses and stores the matrices \f$Y, C_t, B\f$ in a binary
	 * file
	 */
	static void ComputeAndCompressAndStoreInBinaryFile(const MatrixKernelClass *const MatrixKernel, const FReal epsilon);

	/**
	 * Reads the matrices \f$Y, C_t, B\f$ from the respective binary file
	 */
	void ReadFromBinaryFileAndSet();
		

	/**
	 * @return rank of the SVD compressed M2L operators
	 */
	unsigned int getRank() const {return rank;}

  /**
	 * Expands potentials \f$x+=UX\f$ of a target cell. This operation can be
	 * seen as part of the L2L operation.
	 *
	 * @param[in] X compressed local expansion of size \f$r\f$
	 * @param[out] x local expansion of size \f$\ell^3\f$
	 */
  void applyU(const FReal *const X, FReal *const x) const
  {
    FBlas::gemva(nnodes, rank, 1., U, const_cast<FReal*>(X), x);
  }

  /**
	 * Compressed M2L operation \f$X+=C_tY\f$, where \f$Y\f$ is the compressed
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

  /**
	 * Compresses densities \f$Y=B^\top y\f$ of a source cell. This operation
	 * can be seen as part of the M2M operation.
	 *
	 * @param[in] y multipole expansion of size \f$\ell^3\f$
	 * @param[out] Y compressed multipole expansion of size \f$r\f$
	 */
  void applyB(FReal *const y, FReal *const Y) const
  {
    FBlas::gemtv(nnodes, rank, 1., B, y, Y);
  }


};






//////////////////////////////////////////////////////////////////////
// definition ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////






template <class FReal, int ORDER, class MatrixKernelClass>
unsigned int
FChebM2LHandler<FReal, ORDER, MatrixKernelClass>::ComputeAndCompress(const MatrixKernelClass *const MatrixKernel,
                                                              const FReal epsilon,
																															FReal* &U,
																															FReal* &C,
																															FReal* &B)
{
	// allocate memory and store compressed M2L operators
	if (U||C||B) throw std::runtime_error("Compressed M2L operators are already set");

	// interpolation points of source (Y) and target (X) cell
    FPoint<FReal> X[nnodes], Y[nnodes];
	// set roots of target cell (X)
    FChebTensor<FReal, order>::setRoots(FPoint<FReal>(0.,0.,0.), FReal(2.), X);

	// allocate memory and compute 316 m2l operators
	FReal *_U, *_C, *_B;
	_U = _B = nullptr;
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


	//////////////////////////////////////////////////////////		
	FReal weights[nnodes];
    FChebTensor<FReal, order>::setRootOfWeights(weights);
	for (unsigned int i=0; i<316; ++i)
		for (unsigned int n=0; n<nnodes; ++n) {
			FBlas::scal(nnodes, weights[n], _C+i*nnodes*nnodes + n,  nnodes); // scale rows
			FBlas::scal(nnodes, weights[n], _C+i*nnodes*nnodes + n * nnodes); // scale cols
		}
	//////////////////////////////////////////////////////////		

	// svd compression of M2L
    const unsigned int rank	= Compress<FReal, ORDER>(epsilon, ninteractions, _U, _C, _B);
	if (!(rank>0)) throw std::runtime_error("Low rank must be larger then 0!");


	// store U
	U = new FReal [nnodes * rank];
	FBlas::copy(rank*nnodes, _U, U);
	delete [] _U;
	// store B
	B = new FReal [nnodes * rank];
	FBlas::copy(rank*nnodes, _B, B);
	delete [] _B;
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
		

	//////////////////////////////////////////////////////////		
	for (unsigned int n=0; n<nnodes; ++n) {
		FBlas::scal(rank, FReal(1.) / weights[n], U+n, nnodes); // scale rows
		FBlas::scal(rank, FReal(1.) / weights[n], B+n, nnodes); // scale rows
	}
	//////////////////////////////////////////////////////////		


	// return low rank
	return rank;
}






template <class FReal, int ORDER, class MatrixKernelClass>
void
FChebM2LHandler<FReal, ORDER, MatrixKernelClass>::ComputeAndCompressAndStoreInBinaryFile(const MatrixKernelClass *const MatrixKernel, const FReal epsilon)
{
	// measure time
	FTic time; time.tic();
	// start computing process
	FReal *U, *C, *B;
	U = C = B = nullptr;
	const unsigned int rank = ComputeAndCompress(MatrixKernel, epsilon, U, C, B);
	// store into binary file
	const std::string filename(getFileName(epsilon));
	std::ofstream stream(filename.c_str(),
											 std::ios::out | std::ios::binary | std::ios::trunc);
	if (stream.good()) {
		stream.seekp(0);
		// 1) write number of interpolation points (int)
		int _nnodes = nnodes;
		stream.write(reinterpret_cast<char*>(&_nnodes), sizeof(int));
		// 2) write low rank (int)
		int _rank = rank;
		stream.write(reinterpret_cast<char*>(&_rank), sizeof(int));
		// 3) write U (rank*nnodes * FReal)
		stream.write(reinterpret_cast<char*>(U), sizeof(FReal)*rank*nnodes);
		// 4) write B (rank*nnodes * FReal)
		stream.write(reinterpret_cast<char*>(B), sizeof(FReal)*rank*nnodes);
		// 5) write 343 C (343 * rank*rank * FReal)
		stream.write(reinterpret_cast<char*>(C), sizeof(FReal)*rank*rank*343);
	} 	else throw std::runtime_error("File could not be opened to write");
	stream.close();
	// free memory
	if (U != nullptr) delete [] U;
	if (B != nullptr) delete [] B;
	if (C != nullptr) delete [] C;
	// write info
	std::cout << "Compressed M2L operators ("<< rank << ") stored in binary file "	<< filename
						<< " in " << time.tacAndElapsed() << "sec."	<< std::endl;
}


template <class FReal, int ORDER, class MatrixKernelClass>
void
FChebM2LHandler<FReal, ORDER, MatrixKernelClass>::ReadFromBinaryFileAndSet()
{
	// measure time
	FTic time; time.tic();
	// start reading process
	if (U||C||B) throw std::runtime_error("Compressed M2L operator already set");
	const std::string filename(getFileName(epsilon));
	std::ifstream stream(filename.c_str(),
											 std::ios::in | std::ios::binary | std::ios::ate);
	const std::ifstream::pos_type size = stream.tellg();
	if (size<=0) {
		std::cout << "Info: The requested binary file " << filename
							<< " does not yet exist. Compute it now ... " << std::endl;
		this->ComputeAndCompressAndStoreInBinaryFileAndReadFromFileAndSet();
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
		// 3) write U (rank*nnodes * FReal)
		U = new FReal [rank*nnodes];
		stream.read(reinterpret_cast<char*>(U), sizeof(FReal)*rank*nnodes);
		// 4) write B (rank*nnodes * FReal)
		B = new FReal [rank*nnodes];
		stream.read(reinterpret_cast<char*>(B), sizeof(FReal)*rank*nnodes);
		// 5) write 343 C (343 * rank*rank * FReal)
		C = new FReal [343 * rank*rank];
		stream.read(reinterpret_cast<char*>(C), sizeof(FReal)*rank*rank*343);
	}	else throw std::runtime_error("File could not be opened to read");
	stream.close();
	// write info
	std::cout << "Compressed M2L operators (" << rank << ") read from binary file "
						<< filename << " in " << time.tacAndElapsed() << "sec."	<< std::endl;
}

/*
unsigned int ReadRankFromBinaryFile(const std::string& filename)
{
	// start reading process
	std::ifstream stream(filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
	const std::ifstream::pos_type size = stream.tellg();
	if (size<=0) throw std::runtime_error("The requested binary file does not exist.");
	unsigned int rank = -1;
	if (stream.good()) {
		stream.seekg(0);
		// 1) read number of interpolation points (int)
		int npts;
		stream.read(reinterpret_cast<char*>(&npts), sizeof(int));
		// 2) read low rank (int)
		stream.read(reinterpret_cast<char*>(&rank), sizeof(int));
		return rank;
	}	else throw std::runtime_error("File could not be opened to read");
	stream.close();
	return rank;
}
*/


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/**
 * Compresses \f$[K_1,\dots,K_{316}]\f$ in \f$C\f$. Attention: the matrices
 * \f$U,B\f$ are not initialized, no memory is allocated as input, as output
 * they store the respective matrices. The matrix \f$C\f$ stores
 * \f$[K_1,\dots,K_{316}]\f$ as input and \f$[C_1,\dots,C_{316}]\f$ as output.
 *
 * @param[in] epsilon accuracy
 * @param[out] U matrix of size \f$\ell^3\times r\f$
 * @param[in] C matrix of size \f$\ell^3\times 316 \ell^e\f$ storing \f$[K_1,\dots,K_{316}]\f$
 * @param[out] C matrix of size \f$r\times 316 r\f$ storing \f$[C_1,\dots,C_{316}]\f$
 * @param[out] B matrix of size \f$\ell^3\times r\f$
 */
template <class FReal, int ORDER>
unsigned int Compress(const FReal epsilon, const unsigned int ninteractions,
											FReal* &U,	FReal* &C, FReal* &B)
{
	// compile time constants
	enum {order = ORDER,
				nnodes = TensorTraits<ORDER>::nnodes};

	// init SVD
	const unsigned int LWORK = 2 * (3*nnodes + ninteractions*nnodes);
	FReal *const WORK = new FReal [LWORK];
	
	// K_col ///////////////////////////////////////////////////////////
	FReal *const K_col = new FReal [ninteractions * nnodes*nnodes]; 
	for (unsigned int i=0; i<ninteractions; ++i)
		for (unsigned int j=0; j<nnodes; ++j)
			FBlas::copy(nnodes,
									C     + i*nnodes*nnodes + j*nnodes,
									K_col + j*ninteractions*nnodes + i*nnodes);
	// singular value decomposition
	FReal *const Q = new FReal [nnodes*nnodes];
	FReal *const S = new FReal [nnodes];
	const unsigned int info_col
		= FBlas::gesvd(ninteractions*nnodes, nnodes, K_col, S, Q, nnodes,
									 LWORK, WORK);
	if (info_col!=0){
		std::stringstream stream;
		stream << info_col;
		throw std::runtime_error("SVD did not converge with " + stream.str());
	}
	delete [] K_col;
    const unsigned int k_col = FSvd::getRank<FReal, ORDER>(S, epsilon);

	// Q' -> B 
	B = new FReal [nnodes*k_col];
	for (unsigned int i=0; i<k_col; ++i)
		FBlas::copy(nnodes, Q+i, nnodes, B+i*nnodes, 1);

	// K_row //////////////////////////////////////////////////////////////
	FReal *const K_row = C;

	const unsigned int info_row
		= FBlas::gesvdSO(nnodes, ninteractions*nnodes, K_row, S, Q, nnodes,
										 LWORK, WORK);
	if (info_row!=0){
		std::stringstream stream;
		stream << info_row;
		throw std::runtime_error("SVD did not converge with " + stream.str());
	}
    const unsigned int k_row = FSvd::getRank<FReal, ORDER>(S, epsilon);
	delete [] WORK;

	// Q -> U
	U = Q;

	// V' -> V
	FReal *const V = new FReal [nnodes*ninteractions * k_row];
	for (unsigned int i=0; i<k_row; ++i)
		FBlas::copy(nnodes*ninteractions, K_row+i, nnodes,
								V+i*nnodes*ninteractions, 1);

	// rank k(epsilon) /////////////////////////////////////////////////////
	const unsigned int k = k_row < k_col ? k_row : k_col;

	// C_row ///////////////////////////////////////////////////////////
	C = new FReal [ninteractions * k*k];
	for (unsigned int i=0; i<k; ++i) {
		FBlas::scal(nnodes*ninteractions, S[i], V + i*nnodes*ninteractions);
		for (unsigned int m=0; m<ninteractions; ++m)
			for (unsigned int j=0; j<k; ++j)
				C[m*k*k + j*k + i]
					= FBlas::scpr(nnodes,
												V + i*nnodes*ninteractions + m*nnodes,
												B + j*nnodes);
	}

	delete [] V;
	delete [] S;
	delete [] K_row;

	return k;
}




#endif // FCHEBM2LHANDLER_HPP

// [--END--]
