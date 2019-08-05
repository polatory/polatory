// See LICENCE file at project root
#ifndef FCHEBFLOPSSYMKERNEL_HPP
#define FCHEBFLOPSSYMKERNEL_HPP

#include <stdexcept>

#include "../../Utils/FGlobal.hpp"

#include "../../Utils/FSmartPointer.hpp"

#include "../../Components/FAbstractKernels.hpp"

#include "./FChebInterpolator.hpp"
#include "../Interpolation/FInterpSymmetries.hpp"

class FTreeCoordinate;



/**
 * @author Matthias Messner(matthias.messner@inria.fr)
 * @class FChebFlopsSymKernel
 * @brief
 * Please read the license
 *
 * This kernels implement the Chebyshev interpolation based FMM operators
 * exploiting the symmetries in the far-field. It implements all interfaces
 * (P2P, P2M, M2M, M2L, L2L, L2P) which are required by the FFmmAlgorithm and
 * FFmmAlgorithmThread.
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 * @tparam ORDER Chebyshev interpolation order
 */
template < class FReal, class CellClass,	class ContainerClass, class MatrixKernelClass, int ORDER>
class FChebFlopsSymKernel
    : public FAbstractKernels<CellClass, ContainerClass>
{
  enum {nnodes = TensorTraits<ORDER>::nnodes};
public:
	/// Handler to deal with all symmetries: Stores permutation indices and
	/// vectors to reduce 343 different interactions to 16 only.
	struct SymmetryHandler;

	/// Needed for handling all symmetries
	const FSmartPointer<MatrixKernelClass,FSmartPointerMemory> MatrixKernel;
	const FSmartPointer<SymmetryHandler,  FSmartPointerMemory> SymHandler;

	/// tree height
    const unsigned int inTreeHeight;

	// count permuted local and multipole expansions
    unsigned int* countExp;


    unsigned long long flopsP2M, flopsM2M, flopsM2L, flopsL2L, flopsL2P, flopsP2P;

    unsigned long long *flopsPerLevelM2M, *flopsPerLevelM2L, *flopsPerLevelL2L;

	// start flop counters 
    unsigned int countFlopsM2MorL2L() const
	{ return 3 * nnodes * (2*ORDER-1); }
    unsigned int countFlopsM2L(const unsigned int nexp, const unsigned int rank) const
	{ return nexp * (4*nnodes*rank - rank - nnodes); }
    unsigned int countFlopsP2P() const
	{ return 34; }
    unsigned int countFlopsP2Pmutual() const
	{ return 39; }
    unsigned int countFlopsP2M(const unsigned int N) const	{
        const unsigned first  = N * (18 + (ORDER-2) * 6 + (ORDER-1) * (6 + (ORDER-1) * (6 + (ORDER-1) * 2)));
        const unsigned W2 = 3 * ORDER*(2*(ORDER-1)-1);
        const unsigned W4 = 3 * (ORDER*(ORDER-1)*(2*(ORDER-1)-1) + ORDER*ORDER*(2*(ORDER-1)-1));
        const unsigned W8 = 3 * (2*(ORDER-1)-1) * (ORDER*(ORDER-1)*(ORDER-1) + ORDER*ORDER*(ORDER-1) + nnodes);
		return first + W2 + W4 + W8 + nnodes*11;
	}
    unsigned int countFlopsL2PTotal(const unsigned int N) const	{
        const unsigned W0 = nnodes;
        const unsigned W2 = 3 * (ORDER-1)*ORDER*ORDER * 2*ORDER;
        const unsigned W4 = 3 * ORDER*(ORDER-1)*(ORDER-1) * 2*ORDER;
        const unsigned W8 = (ORDER-1)*(ORDER-1)*(ORDER-1) * (2*ORDER-1);
        const unsigned second = N * (38 + (ORDER-2)*15 + (ORDER-1)*((ORDER-1) * (27 + (ORDER-1) * 16))) + 6;
		return W0 + W2 + W4 + W8 + second;
	}
	// end flop counters

public:
	/**
	 * The constructor initializes all constant attributes and it reads the
	 * precomputed and compressed M2L operators from a binary file (an
	 * runtime_error is thrown if the required file is not valid).
	 */
	FChebFlopsSymKernel(const int _inTreeHeight,
			    const FReal inBoxWidth,
                const FPoint<FReal>& inBoxCenter,
			    const FReal Epsilon)
		:	MatrixKernel(new MatrixKernelClass()),
			SymHandler(new SymmetryHandler(MatrixKernel.getPtr(), Epsilon)), inTreeHeight(_inTreeHeight),
			flopsP2M(0), flopsM2M(0), flopsM2L(0), flopsL2L(0), flopsL2P(0), flopsP2P(0),
			flopsPerLevelM2M(nullptr), flopsPerLevelM2L(nullptr), flopsPerLevelL2L(nullptr)
	{
        countExp = new unsigned int [343];
        flopsPerLevelM2M = new unsigned long long [inTreeHeight];
        flopsPerLevelM2L = new unsigned long long [inTreeHeight];
        flopsPerLevelL2L = new unsigned long long [inTreeHeight];
        for (unsigned int level = 0; level<inTreeHeight; ++level)
			flopsPerLevelM2M[level] = flopsPerLevelM2L[level] = flopsPerLevelL2L[level] = 0;
	}
	


	/** Copy constructor */
	FChebFlopsSymKernel(const FChebFlopsSymKernel& other)
		: SymHandler(other.SymHandler),
			flopsP2M(0), flopsM2M(0), flopsM2L(0), flopsL2L(0), flopsL2P(0), flopsP2P(0)
    {	countExp = new unsigned int [343]; }



	/** Destructor */
	~FChebFlopsSymKernel()
	{
		delete [] countExp;

		std::cout << "\n==================================================" 
							<< "\n- Flops for P2M = " << flopsP2M 
							<< "\n- Flops for M2M = " << flopsM2M
							<< "\n- Flops for M2L = " << flopsM2L
							<< "\n- Flops for L2L = " << flopsL2L
							<< "\n- Flops for L2P = " << flopsL2P
							<< "\n- Flops for P2P = " << flopsP2P
							<< "\n- Overall Flops = " << flopsP2M + flopsM2M + flopsM2L + flopsL2L + flopsL2P + flopsP2P
							<< "\n==================================================\n"
							<< std::endl;

		std::cout << "\n==================================================" 
							<< "\n- Flops for P2M/M2M" << std::endl;
        for (unsigned int level=0; level<inTreeHeight; ++level)
			if (level < inTreeHeight-1)
				std::cout << "  |- at level " << level << " flops = " << flopsPerLevelM2M[level] << std::endl;
			else
				std::cout << "  |- at level " << level << " flops = " << flopsP2M << std::endl;
		std::cout << "==================================================" 
							<< "\n- Flops for M2L" << std::endl;
        for (unsigned int level=0; level<inTreeHeight; ++level)
			std::cout << "  |- at level " << level << " flops = " << flopsPerLevelM2L[level] << std::endl;
		std::cout << "==================================================" 
							<< "\n- Flops for L2L/L2P" << std::endl;
        for (unsigned int level=0; level<inTreeHeight; ++level)
			if (level < inTreeHeight-1)
				std::cout << "  |- at level " << level << " flops = " << flopsPerLevelL2L[level] << std::endl;
			else
				std::cout << "  |- at level " << level << " flops = " << flopsL2P << std::endl;
		std::cout << "==================================================" << std::endl; 


		if (flopsPerLevelM2M) delete [] flopsPerLevelM2M;
		if (flopsPerLevelM2L) delete [] flopsPerLevelM2L;
		if (flopsPerLevelL2L) delete [] flopsPerLevelL2L;
	}
	
	
	
    void P2M(CellClass* const /* not needed */, const ContainerClass* const SourceParticles) override
	{
        flopsP2M += countFlopsP2M(int(SourceParticles->getNbParticles()));
	}



	void M2M(CellClass* const FRestrict /* not needed */,
					 const CellClass*const FRestrict *const FRestrict ChildCells,
                     const int TreeLevel) override
	{
        unsigned int flops = 0;
        for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex)
			if (ChildCells[ChildIndex])	flops += countFlopsM2MorL2L();
		flopsM2M += flops;
		flopsPerLevelM2M[TreeLevel] += flops;
	}




	void M2L(CellClass* const FRestrict /* not needed */,
                     const CellClass* SourceCells[],
                    const int positions[],
                     const int size,
                     const int TreeLevel) override
	{
        unsigned int flops = 0;
		// count how ofter each of the 16 interactions is used
		memset(countExp, 0, sizeof(int) * 343);
        for (int idx=0; idx<size; ++idx)
            countExp[SymHandler->pindices[positions[idx]]]++;
		// multiply (mat-mat-mul)
        for (int pidx=0; pidx<343; ++pidx)
			if (countExp[pidx])
				flops += countFlopsM2L(countExp[pidx], SymHandler->LowRank[pidx]) + countExp[pidx]*nnodes;
		flopsM2L += flops;
		flopsPerLevelM2L[TreeLevel] += flops;
	}


	void L2L(const CellClass* const FRestrict /* not needed */,
					 CellClass* FRestrict *const FRestrict ChildCells,
                     const int TreeLevel) override
	{
        unsigned int flops = 0;
        for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex)
			if (ChildCells[ChildIndex])	flops += countFlopsM2MorL2L() + nnodes;
		flopsL2L += flops;
		flopsPerLevelL2L[TreeLevel] += flops;
	}



	void L2P(const CellClass* const /* not needed */,
                     ContainerClass* const TargetParticles) override
	{
		//// 1.a) apply Sx
        //flopsL2P += countFlopsP2MorL2P(TargetParticlesParticles->getNbParticles()) + TargetParticles->getNbParticles();
		//// 1.b) apply Px (grad Sx)
        //flopsL2P += countFlopsL2PGradient(TargetParticlesParticles->getNbParticles()) + 3 * TargetParticles->getNbParticles();

		// or

		// 2) apply Sx and Px (grad Sx)
        flopsL2P += countFlopsL2PTotal(int(TargetParticles->getNbParticles())) + 4 * int(TargetParticles->getNbParticles());
	}



	void P2P(const FTreeCoordinate& /* LeafCellCoordinate */, // needed for periodic boundary conditions
					 ContainerClass* const FRestrict TargetParticles,
					 const ContainerClass* const FRestrict SourceParticles,
                     ContainerClass* const NeighborSourceParticles[],
                     const int neighborPositions[],
                     const int size) override
	{
		if (TargetParticles != SourceParticles) {
            flopsP2P += countFlopsP2P() * TargetParticles->getNbParticles() * SourceParticles->getNbParticles();
            for (int idx=0; idx<size; ++idx)
                    flopsP2P += countFlopsP2P() * TargetParticles->getNbParticles() * NeighborSourceParticles[idx]->getNbParticles();
		} else {
            flopsP2P += countFlopsP2Pmutual() * ((TargetParticles->getNbParticles()*TargetParticles->getNbParticles()+TargetParticles->getNbParticles()) / 2);
            for (int idx=0; idx < size && neighborPositions[idx]<=13; ++idx)
                    flopsP2P += countFlopsP2Pmutual() * TargetParticles->getNbParticles() * NeighborSourceParticles[idx]->getNbParticles();
		}
	}

    void P2POuter(const FTreeCoordinate& /* LeafCellCoordinate */, // needed for periodic boundary conditions
                     ContainerClass* const FRestrict TargetParticles,
                     ContainerClass* const NeighborSourceParticles[],
                     const int neighborPositions[],
                     const int size) override
    {
        for (int idx=0; idx < size && neighborPositions[idx]<=13; ++idx)
                flopsP2P += countFlopsP2Pmutual() * TargetParticles->getNbParticles() * NeighborSourceParticles[idx]->getNbParticles();
    }

};










/**
 * Handler to deal with all symmetries: Stores permutation indices and vectors
 * to reduce 343 different interactions to 16 only.
 */
template <class FReal,			class CellClass,
					class ContainerClass,
					class MatrixKernelClass,
					int ORDER>
struct FChebFlopsSymKernel<FReal, CellClass, ContainerClass, MatrixKernelClass, ORDER>
::SymmetryHandler
{
	// M2L operators
	FReal*    K[343];
	int LowRank[343];
		
	// permutation vectors and permutated indices
    unsigned int pvectors[343][nnodes];
    unsigned int pindices[343];


	// compute rank
    unsigned int getRank(const FReal singular_values[], const double eps)
	{
		FReal nrm2(0.);
        for (unsigned int k=0; k<nnodes; ++k)
			nrm2 += singular_values[k] * singular_values[k];
		
		FReal nrm2k(0.);
        for (unsigned int k=nnodes; k>0; --k) {
			nrm2k += singular_values[k-1] * singular_values[k-1];
			if (nrm2k > eps*eps * nrm2)	return k;
		}
        throw std::runtime_error("rank cannot be larger than nnodes");
		return 0;
	}
	

	/** Constructor */
	SymmetryHandler(const MatrixKernelClass *const MatrixKernel, const double Epsilon)
	{
		// init all 343 item to zero, because effectively only 16 exist
        for (unsigned int t=0; t<343; ++t) {
			K[t] = nullptr;
			LowRank[t] = 0;
		}
			
		// set permutation vector and indices
		const FInterpSymmetries<ORDER> Symmetries;
		for (int i=-3; i<=3; ++i)
			for (int j=-3; j<=3; ++j)
				for (int k=-3; k<=3; ++k)
					if (abs(i)>1 || abs(j)>1 || abs(k)>1) {
                        const unsigned int idx = ((i+3) * 7 + (j+3)) * 7 + (k+3);
						pindices[idx] = Symmetries.getPermutationArrayAndIndex(i,j,k, pvectors[idx]);
					}

		// precompute 16 M2L operators
		this->precomputeSVD(MatrixKernel, Epsilon);
	}



	/** Destructor */
	~SymmetryHandler()
	{
        for (unsigned int t=0; t<343; ++t)
			if (K[  t]!=nullptr) delete [] K[  t];
	}



private:
	void precomputeSVD(const MatrixKernelClass *const MatrixKernel, const double Epsilon)
	{
		// interpolation points of source (Y) and target (X) cell
        FPoint<FReal> X[nnodes], Y[nnodes];
		// set roots of target cell (X)
        FChebTensor<FReal,ORDER>::setRoots(FPoint<FReal>(0.,0.,0.), FReal(2.), X);
		// temporary matrix
		FReal* U = new FReal [nnodes*nnodes];

		// needed for the SVD
        const unsigned int LWORK = 2 * (3*nnodes + nnodes);
		FReal *const WORK = new FReal [LWORK];
		FReal *const VT = new FReal [nnodes*nnodes];
		FReal *const S = new FReal [nnodes];
		
        unsigned int counter = 0;
		for (int i=2; i<=3; ++i) {
			for (int j=0; j<=i; ++j) {
				for (int k=0; k<=j; ++k) {

					// assemble matrix
                    const FPoint<FReal> cy(FReal(2.*i), FReal(2.*j), FReal(2.*k));
					FChebTensor<FReal,ORDER>::setRoots(cy, FReal(2.), Y);
                    for (unsigned int n=0; n<nnodes; ++n)
                        for (unsigned int m=0; m<nnodes; ++m)
							U[n*nnodes + m] = MatrixKernel->evaluate(X[m], Y[n]);

					// applying weights ////////////////////////////////////////
					FReal weights[nnodes];
					FChebTensor<FReal,ORDER>::setRootOfWeights(weights);
                    for (unsigned int n=0; n<nnodes; ++n) {
						FBlas::scal(nnodes, weights[n], U + n,  nnodes); // scale rows
						FBlas::scal(nnodes, weights[n], U + n * nnodes); // scale cols
					}
					//////////////////////////////////////////////////////////		

					// truncated singular value decomposition of matrix
                    long long info	= FBlas::gesvd(nnodes, nnodes, U, S, VT, nnodes, LWORK, WORK);
                    if (info!=0) throw std::runtime_error("SVD did not converge with " + std::to_string(info));
                    const unsigned int rank = this->getRank(S, Epsilon);

					// store 
                    const unsigned int idx = (i+3)*7*7 + (j+3)*7 + (k+3);
					assert(K[idx]==nullptr);
					K[idx] = new FReal [2*rank*nnodes];
					LowRank[idx] = rank;
                    for (unsigned int r=0; r<rank; ++r)
						FBlas::scal(nnodes, S[r], U + r*nnodes);
					FBlas::copy(rank*nnodes, U,  K[idx]);
                    for (unsigned int r=0; r<rank; ++r)
						FBlas::copy(nnodes, VT + r, nnodes, K[idx] + rank*nnodes + r*nnodes, 1);

					// un-weighting ////////////////////////////////////////////
                    for (unsigned int n=0; n<nnodes; ++n) {
						FBlas::scal(rank, FReal(1.) / weights[n], K[idx] + n,               nnodes); // scale rows
						FBlas::scal(rank, FReal(1.) / weights[n], K[idx] + rank*nnodes + n, nnodes); // scale rows
					}
					//////////////////////////////////////////////////////////		

					std::cout << "(" << i << "," << j << "," << k << ") " << idx <<
						", low rank = " << rank << std::endl;

					counter++;
				}
			}
		}
		std::cout << "num interactions = " << counter << std::endl;
		delete [] U;
		delete [] WORK;
		delete [] VT;
		delete [] S;
	}

};











#endif //FCHEBSYMKERNELS_HPP

// [--END--]
