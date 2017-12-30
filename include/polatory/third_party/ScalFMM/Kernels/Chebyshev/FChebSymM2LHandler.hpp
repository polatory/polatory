// See LICENCE file at project root
#ifndef FCHEBSYMM2LHANDLER_HPP
#define FCHEBSYMM2LHANDLER_HPP

#include <climits>
#include <sstream>

#include "../../Utils/FBlas.hpp"


#include "FChebTensor.hpp"
#include "../Interpolation/FInterpSymmetries.hpp"

#include "FChebM2LHandler.hpp"

#include "../../Utils/FAca.hpp"


/**
 * @author Matthias Messner (matthias.matthias@inria.fr)
 * Please read the license
 */


/*!  Choose either \a FULLY_PIVOTED_ACASVD or \a PARTIALLY_PIVOTED_ACASVD or
    \a ONLY_SVD.
 */
//#define ONLY_SVD
//#define FULLY_PIVOTED_ACASVD
#define PARTIALLY_PIVOTED_ACASVD



/*!  Precomputes the 16 far-field interactions (due to symmetries in their
  arrangement all 316 far-field interactions can be represented by
  permutations of the 16 we compute in this function). Depending on whether
  FACASVD is defined or not, either ACA+SVD or only SVD is used to compress
  them. */
template <class FReal, int ORDER, typename MatrixKernelClass>
static void precompute(const MatrixKernelClass *const MatrixKernel, const FReal CellWidth,
        const FReal Epsilon, FReal* K[343], int LowRank[343])
{
    //  std::cout << "\nComputing 16 far-field interactions (l=" << ORDER << ", eps=" << Epsilon
    //                      << ") for cells of width w = " << CellWidth << std::endl;

    static const unsigned int nnodes = ORDER*ORDER*ORDER;

    // interpolation points of source (Y) and target (X) cell
    FPoint<FReal> X[nnodes], Y[nnodes];
    // set roots of target cell (X)
    FChebTensor<FReal, ORDER>::setRoots(FPoint<FReal>(0.,0.,0.), CellWidth, X);
    // temporary matrix
    FReal* U = new FReal [nnodes*nnodes];

    // needed for the SVD
     int INFO;
    const unsigned int LWORK = 2 * (3*nnodes + nnodes);
    FReal *const WORK = new FReal [LWORK];
    FReal *const VT = new FReal [nnodes*nnodes];
    FReal *const S = new FReal [nnodes];


    // initialize timer
    FTic time;
    double overall_time(0.);
    double elapsed_time(0.);

    // initialize rank counter
    unsigned int overall_rank = 0;

    unsigned int counter = 0;
    for (int i=2; i<=3; ++i) {
        for (int j=0; j<=i; ++j) {
            for (int k=0; k<=j; ++k) {

                // assemble matrix and apply weighting matrices
                const FPoint<FReal> cy(CellWidth*FReal(i), CellWidth*FReal(j), CellWidth*FReal(k));
                FChebTensor<FReal, ORDER>::setRoots(cy, CellWidth, Y);
                FReal weights[nnodes];
                FChebTensor<FReal, ORDER>::setRootOfWeights(weights);

                // now the entry-computer is responsible for weighting the matrix entries
                EntryComputer<FReal, MatrixKernelClass> Computer(MatrixKernel, nnodes, X, nnodes, Y, weights);

                // start timer
                time.tic();

#if (defined ONLY_SVD || defined FULLY_PIVOTED_ACASVD)
                Computer(0, nnodes, 0, nnodes, U);
#endif
                /*
                // applying weights ////////////////////////////////////////
                FReal weights[nnodes];
                FChebTensor<FReal,ORDER>::setRootOfWeights(weights);
                for (unsigned int n=0; n<nnodes; ++n) {
                    FBlas::scal(nnodes, weights[n], U + n,  nnodes); // scale rows
                    FBlas::scal(nnodes, weights[n], U + n * nnodes); // scale cols
                }
                 */

                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////
                // ALL PREPROC FLAGS ARE SET ON TOP OF THIS FILE !!! /////////
                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////



                //////////////////////////////////////////////////////////////
#if (defined FULLY_PIVOTED_ACASVD || defined PARTIALLY_PIVOTED_ACASVD) ////////////
                FReal *UU, *VV;
                unsigned int rank;

#ifdef FULLY_PIVOTED_ACASVD
                FAca::fACA(U,        nnodes, nnodes, Epsilon, UU, VV, rank);
#else
                FAca::pACA(Computer, nnodes, nnodes, Epsilon, UU, VV, rank);
#endif 

                // QR decomposition
                FReal* phi = new FReal [rank*rank];
                {
                    // QR of U and V
                    FReal* tauU = new FReal [rank];
                    INFO = FBlas::geqrf(nnodes, rank, UU, tauU, LWORK, WORK);
                    assert(INFO==0);
                    FReal* tauV = new FReal [rank];
                    INFO = FBlas::geqrf(nnodes, rank, VV, tauV, LWORK, WORK);
                    assert(INFO==0);
                    // phi = Ru Rv'
                    FReal* rU = new FReal [2 * rank*rank];
                    FReal* rV = rU + rank*rank;
                    FBlas::setzero(2 * rank*rank, rU);
                    for (unsigned int l=0; l<rank; ++l) {
                        FBlas::copy(l+1, UU + l*nnodes, rU + l*rank);
                        FBlas::copy(l+1, VV + l*nnodes, rV + l*rank);
                    }
                    FBlas::gemmt(rank, rank, rank, FReal(1.), rU, rank, rV, rank, phi, rank);
                    delete [] rU;
                    // get Qu and Qv
                    INFO = FBlas::orgqr(nnodes, rank, UU, tauU, LWORK, WORK);
                    assert(INFO==0);
                    INFO = FBlas::orgqr(nnodes, rank, VV, tauV, LWORK, WORK);
                    assert(INFO==0);
                    delete [] tauU;
                    delete [] tauV;
                }

                const unsigned int aca_rank = rank;

                // SVD
                {
                    INFO = FBlas::gesvd(aca_rank, aca_rank, phi, S, VT, aca_rank, LWORK, WORK);
                    if (INFO!=0){
                        std::stringstream stream;
                        stream << INFO;
                        throw std::runtime_error("SVD did not converge with " + stream.str());
                    }
                    rank = FSvd::getRank(S, aca_rank, Epsilon);
                }                   

                const unsigned int idx = (i+3)*7*7 + (j+3)*7 + (k+3);

                // store
                {
                    // allocate
                    assert(K[idx]==nullptr);
                    K[idx] = new FReal [2*rank*nnodes];

                    // set low rank
                    LowRank[idx] = static_cast<int>(rank);

                    // (U Sigma)
                    for (unsigned int r=0; r<rank; ++r)
                        FBlas::scal(aca_rank, S[r], phi + r*aca_rank);

                    // Qu (U Sigma) 
                    FBlas::gemm(nnodes, aca_rank, rank, FReal(1.), UU, nnodes, phi, aca_rank, K[idx], nnodes);
                    delete [] phi;

                    // Vt -> V and then Qu V
                    FReal *const V = new FReal [aca_rank * rank];
                    for (unsigned int r=0; r<rank; ++r)
                        FBlas::copy(aca_rank, VT + r, aca_rank, V + r*aca_rank, 1);
                    FBlas::gemm(nnodes, aca_rank, rank, FReal(1.), VV, nnodes, V, aca_rank, K[idx] + rank*nnodes, nnodes);
                    delete [] V;
                }

                //// store recompressed UV
                //const unsigned int idx = (i+3)*7*7 + (j+3)*7 + (k+3);
                //assert(K[idx]==NULL);
                //K[idx] = new FReal [2*rank*nnodes];
                //LowRank[idx] = rank;
                //FBlas::copy(rank*nnodes, UU,  K[idx]);
                //FBlas::copy(rank*nnodes, VV,  K[idx] + rank*nnodes);

                delete [] UU;
                delete [] VV;

                elapsed_time = time.tacAndElapsed(); 
                overall_time += elapsed_time;
                overall_rank += rank;
                // std::cout << "(" << i << "," << j << "," << k << ") " << idx <<
                //  ", low rank = " << rank << " (" << aca_rank << ") in " << elapsed_time << "s" << std::endl;

                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////
                // ALL PREPROC FLAGS ARE SET ON TOP OF THIS FILE !!! /////////
                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////

#elif defined ONLY_SVD
                // truncated singular value decomposition of matrix
                INFO = FBlas::gesvd(nnodes, nnodes, U, S, VT, nnodes, LWORK, WORK);
                if (INFO!=0){
                    std::stringstream stream;
                    stream << INFO;
                    throw std::runtime_error("SVD did not converge with " + stream.str());
                }
                const unsigned int rank = FSvd::getRank<ORDER>(S, Epsilon);

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

                elapsed_time = time.tacAndElapsed(); 
                overall_time += elapsed_time;
                overall_rank += rank;
                //              std::cout << "(" << i << "," << j << "," << k << ") " << idx <<
                //  ", low rank = " << rank << " in " << elapsed_time << "s" << std::endl;
#else
#error Either fully-, partially pivoted ACA or only SVD must be defined!
#endif ///////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////


                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////
                // ALL PREPROC FLAGS ARE SET ON TOP OF THIS FILE !!! /////////
                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////


                // un-weighting ////////////////////////////////////////////
                for (unsigned int n=0; n<nnodes; ++n) {
                    FBlas::scal(rank, FReal(1.) / weights[n], K[idx] + n,               nnodes); // scale rows
                    FBlas::scal(rank, FReal(1.) / weights[n], K[idx] + rank*nnodes + n, nnodes); // scale rows
                }
                //////////////////////////////////////////////////////////      

                ++counter;
            }
        }
    }
    //std::cout << "The approximation of the " << counter
    //      << " far-field interactions (overall rank " << overall_rank
    //      << " / " << 16*nnodes
    //      << " , sizeM2L= " << 2*overall_rank*nnodes*sizeof(FReal) << ""
    //      << " / " << 16*nnodes*nnodes*sizeof(FReal) << " B"
    //      << ") took " << overall_time << "s\n" << std::endl;

    std::cout << "Compressed and set M2L operators (" << 2*overall_rank*nnodes*sizeof(FReal) << " B) in " << overall_time << "sec." << std::endl;

    delete [] U;
    delete [] WORK;
    delete [] VT;
    delete [] S;
}









/*!  \class SymmetryHandler 

    \brief Deals with all the symmetries in the arrangement of the far-field interactions

    Stores permutation indices and permutation vectors to reduce 316 (7^3-3^3)
  different far-field interactions to 16 only. We use the number 343 (7^3)
  because it allows us to use to associate the far-field interactions based on
  the index \f$t = 7^2(i+3) + 7(j+3) + (k+3)\f$ where \f$(i,j,k)\f$ denotes
  the relative position of the source cell to the target cell. */
template <class FReal, int ORDER, KERNEL_FUNCTION_TYPE TYPE> class SymmetryHandler;

/*! Specialization for homogeneous kernel functions */
template <class FReal, int ORDER>
class SymmetryHandler<FReal, ORDER, HOMOGENEOUS>
{
    static const unsigned int nnodes = ORDER*ORDER*ORDER;

    // M2L operators
    FReal*    K[343];
    int LowRank[343];

public:

    // permutation vectors and permutated indices
    unsigned int pvectors[343][nnodes];
    unsigned int pindices[343];


    /** Constructor: with 16 small SVDs */
    template <typename MatrixKernelClass>
    SymmetryHandler(const MatrixKernelClass *const MatrixKernel, const FReal Epsilon,
                    const FReal, const unsigned int)
    {
        // init all 343 item to zero, because effectively only 16 exist
        for (unsigned int t=0; t<343; ++t) {
            K[t]            = nullptr;
            LowRank[t] = 0;
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

        // precompute 16 M2L operators
        const FReal ReferenceCellWidth = FReal(2.0);
        precompute<FReal, ORDER>(MatrixKernel, ReferenceCellWidth, Epsilon, K, LowRank);
    }



    /** Destructor */
    ~SymmetryHandler()
    {
        for (unsigned int t=0; t<343; ++t) if (K[t]!=nullptr) delete [] K[t];
    }


    /*! return the t-th approximated far-field interactions*/
    const FReal * getK(const  int, const unsigned int t) const
    {   return K[t]; }

    /*! return the t-th approximated far-field interactions*/
    int getLowRank(const int, const unsigned int t) const
    {   return LowRank[t]; }

};






/*! Specialization for non-homogeneous kernel functions */
template <class FReal, int ORDER>
class SymmetryHandler<FReal, ORDER, NON_HOMOGENEOUS>
{
    static const unsigned int nnodes = ORDER*ORDER*ORDER;

    // Height of octree; needed only in the case of non-homogeneous kernel functions
    const unsigned int TreeHeight;

    // M2L operators for all levels in the octree
    FReal***    K;
    int** LowRank;

public:

    // permutation vectors and permutated indices
    unsigned int pvectors[343][nnodes];
    unsigned int pindices[343];


    /** Constructor: with 16 small SVDs */
    template <typename MatrixKernelClass>
    SymmetryHandler(const MatrixKernelClass *const MatrixKernel, const double Epsilon,
                    const FReal RootCellWidth, const unsigned int inTreeHeight)
    : TreeHeight(inTreeHeight)
    {
        // init all 343 item to zero, because effectively only 16 exist
        K       = new FReal** [TreeHeight];
        LowRank = new int*    [TreeHeight];
        K[0]       = nullptr; K[1]       = nullptr;
        LowRank[0] = nullptr; LowRank[1] = nullptr;
        for (unsigned int l=2; l<TreeHeight; ++l) {
            K[l]       = new FReal* [343];
            LowRank[l] = new int    [343];
            for (unsigned int t=0; t<343; ++t) {
                K[l][t]       = nullptr;
                LowRank[l][t] = 0;
            }
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
            precompute<FReal,ORDER>(MatrixKernel, CellWidth, Epsilon, K[l], LowRank[l]);
            CellWidth /= FReal(2.);                    // at level l+1 
        }
    }



    /** Destructor */
    ~SymmetryHandler()
    {
        for (unsigned int l=0; l<TreeHeight; ++l) {
            if (K[l]!=nullptr) {
                for (unsigned int t=0; t<343; ++t) if (K[l][t]!=nullptr) delete [] K[l][t];
                delete [] K[l];
            }
            if (LowRank[l]!=nullptr)    delete [] LowRank[l];
        }
        delete [] K;
        delete [] LowRank;
    }

    /*! return the t-th approximated far-field interactions*/
    const FReal * getK(const  int l, const unsigned int t) const
    {   return K[l][t]; }

    /*! return the t-th approximated far-field interactions*/
    int getLowRank(const  int l, const unsigned int t) const
    {   return LowRank[l][t]; }

};








#include <fstream>
#include <sstream>


/**
 * Computes, compresses and stores the 16 M2L kernels in a binary file.
 */
template <class FReal, int ORDER, typename MatrixKernelClass>
static void ComputeAndCompressAndStoreInBinaryFile(const MatrixKernelClass *const MatrixKernel, const FReal Epsilon)
{
    static const unsigned int nnodes = ORDER*ORDER*ORDER;

    // compute and compress ////////////
    FReal* K[343];
    int LowRank[343];
    for (unsigned int idx=0; idx<343; ++idx) { K[idx] = nullptr; LowRank[idx] = 0;  }
    precompute<FReal,ORDER>(MatrixKernel, FReal(2.), Epsilon, K, LowRank);

    // write to binary file ////////////
    FTic time; time.tic();
    // start computing process
    const char precision = (typeid(FReal)==typeid(double) ? 'd' : 'f');
    std::stringstream sstream;
    sstream << "sym2l_" << precision << "_o" << ORDER << "_e" << Epsilon << ".bin";
    const std::string filename(sstream.str());
    std::ofstream stream(filename.c_str(),
            std::ios::out | std::ios::binary | std::ios::trunc);
    if (stream.good()) {
        stream.seekp(0);
        for (unsigned int idx=0; idx<343; ++idx)
            if (K[idx]!=nullptr) {
                // 1) write index
                stream.write(reinterpret_cast<char*>(&idx), sizeof(int));
                // 2) write low rank (int)
                int rank = LowRank[idx];
                stream.write(reinterpret_cast<char*>(&rank), sizeof(int));
                // 3) write U and V (both: rank*nnodes * FReal)
                FReal *const U = K[idx];
                FReal *const V = K[idx] + rank*nnodes;
                stream.write(reinterpret_cast<char*>(U), sizeof(FReal)*rank*nnodes);
                stream.write(reinterpret_cast<char*>(V), sizeof(FReal)*rank*nnodes);
            }
    } else throw std::runtime_error("File could not be opened to write");
    stream.close();
    // write info
    //  std::cout << "Compressed M2L operators stored in binary file " << filename
    //                  << " in " << time.tacAndElapsed() << "sec." << std::endl;

    // free memory /////////////////////
    for (unsigned int t=0; t<343; ++t) if (K[t]!=nullptr) delete [] K[t];
}


/**
 * Reads the 16 compressed M2L kernels from the binary files and writes them
 * in K and the respective low-rank in LowRank.
 */
template <class FReal, int ORDER>
void ReadFromBinaryFile(const FReal Epsilon, FReal* K[343], int LowRank[343])
{
    // compile time constants
    const unsigned int nnodes = ORDER*ORDER*ORDER;

    // find filename
    const char precision = (typeid(FReal)==typeid(double) ? 'd' : 'f');
    std::stringstream sstream;
    sstream << "sym2l_" << precision << "_o" << ORDER << "_e" << Epsilon << ".bin";
    const std::string filename(sstream.str());

    // read binary file
    std::ifstream istream(filename.c_str(),
            std::ios::in | std::ios::binary | std::ios::ate);
    const std::ifstream::pos_type size = istream.tellg();
    if (size<=0) throw std::runtime_error("The requested binary file does not yet exist. Exit.");

    if (istream.good()) {
        istream.seekg(0);
        // 1) read index (int)
        int _idx;
        istream.read(reinterpret_cast<char*>(&_idx), sizeof(int));
        // loop to find 16 compressed m2l operators
        for (int idx=0; idx<343; ++idx) {
            K[idx] = nullptr;
            LowRank[idx] = 0;
            // if it exists
            if (idx == _idx) {
                // 2) read low rank (int)
                int rank;
                istream.read(reinterpret_cast<char*>(&rank), sizeof(int));
                LowRank[idx] = rank;
                // 3) read U and V (both: rank*nnodes * FReal)
                K[idx] = new FReal [2*rank*nnodes];
                FReal *const U = K[idx];
                FReal *const V = K[idx] + rank*nnodes;
                istream.read(reinterpret_cast<char*>(U), sizeof(FReal)*rank*nnodes);
                istream.read(reinterpret_cast<char*>(V), sizeof(FReal)*rank*nnodes);

                // 1) read next index
                istream.read(reinterpret_cast<char*>(&_idx), sizeof(int));
            }
        }
    }   else throw std::runtime_error("File could not be opened to read");
    istream.close();
}





#endif
