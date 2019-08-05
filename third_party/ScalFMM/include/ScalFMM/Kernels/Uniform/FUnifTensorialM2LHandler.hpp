// See LICENCE file at project root
// Keep in private GIT

#ifndef FUNIFTENSORIALM2LHANDLER_HPP
#define FUNIFTENSORIALM2LHANDLER_HPP

#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>
#include <typeinfo>

#include "Utils/FBlas.hpp"
#include "Utils/FTic.hpp"
#include "Utils/FDft.hpp"

#include "Utils/FComplex.hpp"


#include "FUnifTensor.hpp"


/*!  Precomputation of the 316 interactions by evaluation of the matrix kernel on the uniform grid and transformation into Fourier space. These interactions can be tensorial (of size ncmp) and are computed blockwise
PB: Compute() does not belong to the M2LHandler like it does in the Chebyshev kernel. This allows much nicer specialization of the M2LHandler class with respect to the homogeneity of the kernel of interaction like in the ChebyshevSym kernel.*/
template < class FReal, int ORDER, class MatrixKernelClass>
static void Compute(const MatrixKernelClass *const MatrixKernel, 
                    const FReal CellWidth, 
                    const FReal CellWidthExtension, 
                    FComplex<FReal>** &FC, 
                    const int SeparationCriterion = 1)
{
    // dimensions of operators
    const unsigned int order = ORDER;
    const unsigned int nnodes = TensorTraits<ORDER>::nnodes;
    const unsigned int ninteractions = 316+26*(SeparationCriterion<1 ? 1 : 0) + 1*(SeparationCriterion<0 ? 1 : 0);
    const unsigned int ncmp = MatrixKernelClass::NCMP;

    // utils
    typedef FUnifTensor<FReal,ORDER> TensorType;

    // allocate memory and store compressed M2L operators
    for (unsigned int d=0; d<ncmp; ++d) 
    if (FC[d]) throw std::runtime_error("M2L operators are already set");

    // interpolation points of source (Y) and target (X) cell
    FPoint<FReal> X[nnodes], Y[nnodes];
    // set roots of target cell (X)
    const FReal ExtendedCellWidth(CellWidth+CellWidthExtension);
    FUnifTensor<FReal,order>::setRoots(FPoint<FReal>(0.,0.,0.), ExtendedCellWidth, X);

    // allocate memory and compute 316 m2l operators
    FReal** _C;
    FComplex<FReal>** _FC;

    // reduce storage from nnodes^2=order^6 to (2order-1)^3
    const unsigned int rc = (2*order-1)*(2*order-1)*(2*order-1);
    _C  = new FReal* [ncmp];
    _FC = new FComplex<FReal>* [ncmp];
    for (unsigned int d=0; d<ncmp; ++d) 
    _C[d]  = new FReal[rc];
    for (unsigned int d=0; d<ncmp; ++d) 
    _FC[d] = new FComplex<FReal>[rc * ninteractions];

    // initialize root node ids pairs
    unsigned int node_ids_pairs[rc][2];
    TensorType::setNodeIdsPairs(node_ids_pairs);

    // init Discrete Fourier Transformator
    const int dimfft = 1; // unidim FFT since fully circulant embedding
    FFftw<FReal,FComplex<FReal>,dimfft> Dft(rc);

    // get first column of K via permutation
    unsigned int perm[rc];
    TensorType::setStoragePermutation(perm);

    // Allocate intermediate interaction block
    FReal* block = new FReal[ncmp]; 

    unsigned int counter = 0;
    for (int i=-3; i<=3; ++i) {
        for (int j=-3; j<=3; ++j) {
            for (int k=-3; k<=3; ++k) {
                if (abs(i)>SeparationCriterion || abs(j)>SeparationCriterion || abs(k)>SeparationCriterion) {
                    // set roots of source cell (Y)
                    const FPoint<FReal> cy(CellWidth*FReal(i), CellWidth*FReal(j), CellWidth*FReal(k));
                    FUnifTensor<FReal,order>::setRoots(cy, ExtendedCellWidth, Y);
                    // evaluate m2l operator
                    unsigned int ido=0;
                    for(unsigned int l=0; l<2*order-1; ++l)
                        for(unsigned int m=0; m<2*order-1; ++m)
                            for(unsigned int n=0; n<2*order-1; ++n){   
                                // Compute current M2L interaction (block matrix)
                                MatrixKernel->evaluateBlock(X[node_ids_pairs[ido][0]], 
                                                            Y[node_ids_pairs[ido][1]],
                                                            block);
                                // store value at current position in C
                                // use permutation if DFT is used because 
                                // the storage of the first column is required
                                // i.e. C[0] C[rc-1] C[rc-2] .. C[1] < WRONG!
                                // i.e. C[rc-1] C[0] C[1] .. C[rc-2] < RIGHT!
                                //                _C[counter*rc + ido]
                                for (unsigned int d=0; d<ncmp; ++d) 
                                    _C[d][perm[ido]] = block[d];
                                ido++;
                            }

                    // Apply Discrete Fourier Transformation
                    for (unsigned int d=0; d<ncmp; ++d) 
                        Dft.applyDFT(_C[d],_FC[d]+counter*rc);

                    // increment interaction counter
                    counter++;
                }
            }
        }
    }
    if (counter != ninteractions)
        throw std::runtime_error("Number of interactions must correspond to 316");

    // Free block
    delete [] block;

    // Free _C
    for (unsigned int d=0; d<ncmp; ++d) 
    delete [] _C[d];

    // store FC
    counter = 0;
    // reduce storage if real valued kernel
    const unsigned int opt_rc = rc/2+1;
    // allocate M2L
    for (unsigned int d=0; d<ncmp; ++d) 
    FC[d] = new FComplex<FReal>[343 * opt_rc];

    for (int i=-3; i<=3; ++i)
        for (int j=-3; j<=3; ++j)
            for (int k=-3; k<=3; ++k) {
                const unsigned int idx = (i+3)*7*7 + (j+3)*7 + (k+3);
                if (abs(i)>SeparationCriterion || abs(j)>SeparationCriterion || abs(k)>SeparationCriterion) {
                    for (unsigned int d=0; d<ncmp; ++d) 
                        FBlas::c_copy(opt_rc, reinterpret_cast<FReal*>(_FC[d] + counter*rc), 
                                      reinterpret_cast<FReal*>(FC[d] + idx*opt_rc));
                    counter++;
                } else { 
                    for (unsigned int d=0; d<ncmp; ++d) 
                        FBlas::c_setzero(opt_rc, reinterpret_cast<FReal*>(FC[d] + idx*opt_rc));
                }
            }

    if (counter != ninteractions)
        throw std::runtime_error("Number of interactions must correspond to 316");
    for (unsigned int d=0; d<ncmp; ++d) 
        delete [] _FC[d];   
}

/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FUnifTensorialM2LHandler
 * Please read the license
 *
 * This class precomputes and efficiently stores the M2L operators
 * \f$[K_1,\dots,K_{316}]\f$ for all (\f$7^3-3^3 = 316\f$ possible interacting
 * cells in the far-field) interactions for the Lagrange interpolation
 * approach. The resulting Lagrange operators have a Circulant Toeplitz 
 * structure and can be applied efficiently in Fourier Space. Hence, the
 * originally \f$K_t\f$ of size \f$\ell^3\times\ell^3\f$ times \f$316\f$ for
 * all interactions is reduced to \f$316\f$ \f$C_t\f$, each of size \f$2\ell-1\f$.
 *
 * @tparam ORDER interpolation order \f$\ell\f$
 */
template <class FReal, int ORDER, class MatrixKernelClass, KERNEL_FUNCTION_TYPE TYPE> class FUnifTensorialM2LHandler;

template <class FReal, int ORDER, class MatrixKernelClass>
class FUnifTensorialM2LHandler<FReal, ORDER,MatrixKernelClass,HOMOGENEOUS>
{
    enum {order = ORDER,
          nnodes = TensorTraits<ORDER>::nnodes,
          ninteractions = 316, // 7^3 - 3^3 (max num cells in far-field)
          rc = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1),
          ncmp = MatrixKernelClass::NCMP};

    /// M2L Operators (stored in Fourier space for each component of tensor)
    FSmartPointer< FComplex<FReal>*,FSmartArrayMemory> FC;

    const FReal CellWidthExtension; //<! extension of cells width

    /// Utils
    typedef FUnifTensor<FReal,ORDER> TensorType;
    unsigned int node_diff[nnodes*nnodes];

    /// DFT specific
    static const int dimfft = 1; // unidim FFT since fully circulant embedding
    typedef FFftw<FReal,FComplex<FReal>,dimfft> DftClass; // Fast Discrete Fourier Transformator
    DftClass Dft;
    const unsigned int opt_rc; // specific to real valued kernel

    /// Leaf level separation criterion
    const int LeafLevelSeparationCriterion;

    static const std::string getFileName()
    {
        const char precision_type = (typeid(FReal)==typeid(double) ? 'd' : 'f');
        std::stringstream stream;
        stream << "m2l_k"/*<< MatrixKernelClass::getID()*/ << "_" << precision_type
               << "_o" << order << ".bin";
        return stream.str();
    }

    
public:
    FUnifTensorialM2LHandler(const MatrixKernelClass *const MatrixKernel, const unsigned int, const FReal, const FReal inCellWidthExtension, const int inLeafLevelSeparationCriterion = 1)
        :  CellWidthExtension(inCellWidthExtension), 
           Dft(rc), opt_rc(rc/2+1), LeafLevelSeparationCriterion(inLeafLevelSeparationCriterion)
    {

        // allocate FC
        FC = new FComplex<FReal>*[ncmp];
        for (unsigned int d=0; d<ncmp; ++d)
            FC[d]=nullptr;

        // initialize root node ids
        TensorType::setNodeIdsDiff(node_diff);

        // Compute and Set M2L Operators
        ComputeAndSet(MatrixKernel);
    }

  /*
   * Copy constructor
   */
  FUnifTensorialM2LHandler(const FUnifTensorialM2LHandler& other)
    : FC(other.FC),
      CellWidthExtension(other.CellWidthExtension),
      Dft(other.Dft), opt_rc(other.opt_rc), LeafLevelSeparationCriterion(other.LeafLevelSeparationCriterion)
    {
        // copy node_diff
        memcpy(node_diff,other.node_diff,sizeof(unsigned int)*nnodes*nnodes);
    }

    ~FUnifTensorialM2LHandler()
    {
        for (unsigned int d=0; d<ncmp; ++d)
            if (FC[d] != nullptr) delete [] FC[d];
    }

    /**
     * Computes and sets the matrix \f$C_t\f$
     */
    void ComputeAndSet(const MatrixKernelClass *const MatrixKernel)
    {
        // measure time
        FTic time; time.tic();

        // check if aready set
        for (unsigned int d=0; d<ncmp; ++d)
            if (FC[d]) throw std::runtime_error("M2L operator already set");

        // Compute matrix of interactions
        // reference cell width is arbitrarly set to 2. 
        // but it NEEDS to match the numerator of the scale factor in matrix kernel!
        // Therefore box width extension is not yet supported for homog kernels
        const FReal ReferenceCellWidth = FReal(2.);
        Compute<FReal,order>(MatrixKernel,ReferenceCellWidth, 0., FC, LeafLevelSeparationCriterion);
        
        // Compute memory usage
        unsigned long sizeM2L = 343*ncmp*opt_rc*sizeof(FComplex<FReal>);

        // write info
        std::cout << "Compute and set M2L operators ("<< long(sizeM2L/**1e-6*/) <<" B) in "
                  << time.tacAndElapsed() << "sec."   << std::endl;
    }

    unsigned long long getMemory() const {
        return 343*ncmp*opt_rc*sizeof(FComplex<FReal>);
    }        

    /**
     * Expands potentials \f$x+=IDFT(X)\f$ of a target cell. This operation can be
     * seen as part of the L2L operation.
     *
     * @param[in] X transformed local expansion of size \f$r\f$
     * @param[out] x local expansion of size \f$\ell^3\f$
     */
    void unapplyZeroPaddingAndDFT(const FComplex<FReal> *const FX, FReal *const x) const
    { 
        FReal Px[rc];
        FBlas::setzero(rc,Px);
        // Apply forward Discrete Fourier Transform
        Dft.applyIDFTNorm(FX,Px);
        // Unapply Zero Padding
        for (unsigned int j=0; j<nnodes; ++j)
            x[j]+=Px[node_diff[nnodes-j-1]];
    }

    /**
     * The M2L operation \f$X+=C_tY\f$ is performed in Fourier space by 
     * application of the convolution theorem (i.e. \f$FX+=FC_t:FY\f$), 
     * where \f$FY\f$ is the transformed multipole expansion and 
     * \f$FX\f$ is the transformed local expansion, both padded with zeros and
     * of size \f$r_c=(2\times ORDER-1)^3\f$ or \f$r_c^{opt}=\frac{r_c}{2}+1\f$ 
     * for real valued kernels. The index \f$t\f$ denotes the transfer vector 
     * of the target cell to the source cell.
     *
     * @param[in] transfer transfer vector
     * @param[in] FY transformed multipole expansion
     * @param[out] FX transformed local expansion
     * @param[in] CellWidth needed for the scaling of the compressed M2L operators which are based on a homogeneous matrix kernel computed for the reference cell width \f$w=2\f$, ie in \f$[-1,1]^3\f$.
     */
    void applyFC(const unsigned int idx, const unsigned int , 
                 const FReal scale, const unsigned int d,
                 const FComplex<FReal> *const FY, FComplex<FReal> *const FX) const
    {
        // Perform entrywise product manually
        for (unsigned int j=0; j<opt_rc; ++j){
            FX[j].addMul(FComplex<FReal>(scale*FC[d][idx*opt_rc + j].getReal(),
                                  scale*FC[d][idx*opt_rc + j].getImag()), 
                         FY[j]);
        }
    
    }


    /**
     * Transform densities \f$Y= DFT(y)\f$ of a source cell. This operation
     * can be seen as part of the M2M operation.
     *
     * @param[in] y multipole expansion of size \f$\ell^3\f$
     * @param[out] Y transformed multipole expansion of size \f$r\f$
     */
    void applyZeroPaddingAndDFT(FReal *const y, FComplex<FReal> *const FY) const
    {
        FReal Py[rc];
        FBlas::setzero(rc,Py);
        // Apply Zero Padding
        for (unsigned int i=0; i<nnodes; ++i)
            Py[node_diff[i*nnodes]]=y[i];
        // Apply forward Discrete Fourier Transform
        Dft.applyDFT(Py,FY);
    }


};



template <class FReal, int ORDER, class MatrixKernelClass>
class FUnifTensorialM2LHandler<FReal,ORDER,MatrixKernelClass,NON_HOMOGENEOUS>
{
    enum {order = ORDER,
          nnodes = TensorTraits<ORDER>::nnodes,
          ninteractions = 316, // 7^3 - 3^3 (max num cells in far-field)
          rc = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1),
          ncmp = MatrixKernelClass::NCMP};


    /// M2L Operators (stored in Fourier space for each component and each level)
    FSmartPointer< FComplex<FReal>**,FSmartArrayMemory> FC;

    /// Homogeneity specific variables
    const unsigned int TreeHeight;
    const FReal RootCellWidth;
    const FReal CellWidthExtension; //<! extension of cells width

    /// Utils
    typedef FUnifTensor<FReal,ORDER> TensorType;
    unsigned int node_diff[nnodes*nnodes];

    /// DFT specific
    static const int dimfft = 1; // unidim FFT since fully circulant embedding
    typedef FFftw<FReal,FComplex<FReal>,dimfft> DftClass; // Fast Discrete Fourier Transformator
    DftClass Dft;
    const unsigned int opt_rc; // specific to real valued kernel

    /// Leaf level separation criterion
    const int LeafLevelSeparationCriterion;

    static const std::string getFileName()
    {
        const char precision_type = (typeid(FReal)==typeid(double) ? 'd' : 'f');
        std::stringstream stream;
        stream << "m2l_k"/*<< MatrixKernelClass::getID*/ << "_" << precision_type
                     << "_o" << order << ".bin";
        return stream.str();
    }

    
public:
    FUnifTensorialM2LHandler(const MatrixKernelClass *const MatrixKernel, const unsigned int inTreeHeight, const FReal inRootCellWidth, const FReal inCellWidthExtension, const int inLeafLevelSeparationCriterion = 1)
        : TreeHeight(inTreeHeight),
          RootCellWidth(inRootCellWidth), 
          CellWidthExtension(inCellWidthExtension),
          Dft(rc), opt_rc(rc/2+1), LeafLevelSeparationCriterion(inLeafLevelSeparationCriterion)
    {

        // allocate FC
        FC = new FComplex<FReal>**[TreeHeight];
        for (unsigned int l=0; l<TreeHeight; ++l){ 
            FC[l] = new FComplex<FReal>*[ncmp];
            for (unsigned int d=0; d<ncmp; ++d)
                FC[l][d]=nullptr;
        }

        // initialize root node ids
        TensorType::setNodeIdsDiff(node_diff);

        // Compute and Set M2L Operators
        ComputeAndSet(MatrixKernel);
    }

    /*
    * Copy constructor
    */
    FUnifTensorialM2LHandler(const FUnifTensorialM2LHandler& other)
    : FC(other.FC),
      TreeHeight(other.TreeHeight),
      RootCellWidth(other.RootCellWidth), 
      CellWidthExtension(other.CellWidthExtension),
      Dft(other.Dft), opt_rc(other.opt_rc), LeafLevelSeparationCriterion(other.LeafLevelSeparationCriterion)
    {    
        // copy node_diff
        memcpy(node_diff,other.node_diff,sizeof(unsigned int)*nnodes*nnodes);
    }


    ~FUnifTensorialM2LHandler()
   {
      if(FC.isLast()){
        for (unsigned int l=0; l<TreeHeight; ++l) 
	  for (unsigned int d=0; d<ncmp; ++d)
	    if (FC[l][d] != nullptr) delete [] FC[l][d];
      }
    }
    /**
     * Computes and sets the matrix \f$C_t\f$
     */
    void ComputeAndSet(const MatrixKernelClass *const MatrixKernel)
    {
        // measure time
        FTic time; time.tic();
    
        // Compute matrix of interactions at each level !! (since non homog)
        FReal CellWidth = RootCellWidth / FReal(2.); // at level 1
        CellWidth /= FReal(2.);                      // at level 2
        for (unsigned int l=2; l<TreeHeight; ++l) {
            // Determine separation criteria wrt level
            const int SeparationCriterion = (l != TreeHeight-1 ? 1 : LeafLevelSeparationCriterion);
            // check if already set
            for (unsigned int d=0; d<ncmp; ++d)
                if (FC[l][d]) throw std::runtime_error("M2L operator already set");
            Compute<FReal,order>(MatrixKernel,CellWidth,CellWidthExtension,FC[l],SeparationCriterion);
            CellWidth /= FReal(2.);                    // at level l+1 
        }

        // Compute memory usage
        unsigned long sizeM2L = (TreeHeight-2)*343*ncmp*opt_rc*sizeof(FComplex<FReal>);

        // write info
        std::cout << "Compute and set M2L operators ("<< long(sizeM2L/**1e-6*/) <<" B) in "
                  << time.tacAndElapsed() << "sec."   << std::endl;
    }

    unsigned long long getMemory() const {
        return (TreeHeight-2)*343*ncmp*opt_rc*sizeof(FComplex<FReal>);
    }        

    /**
     * Expands potentials \f$x+=IDFT(X)\f$ of a target cell. This operation can be
     * seen as part of the L2L operation.
     *
     * @param[in] X transformed local expansion of size \f$r\f$
     * @param[out] x local expansion of size \f$\ell^3\f$
     */
    void unapplyZeroPaddingAndDFT(const FComplex<FReal> *const FX, FReal *const x) const
    { 
        FReal Px[rc];
        FBlas::setzero(rc,Px);
        // Apply forward Discrete Fourier Transform
        Dft.applyIDFTNorm(FX,Px);
        // Unapply Zero Padding
        for (unsigned int j=0; j<nnodes; ++j)
            x[j]+=Px[node_diff[nnodes-j-1]];
    }

    /**
     * The M2L operation \f$X+=C_tY\f$ is performed in Fourier space by 
     * application of the convolution theorem (i.e. \f$FX+=FC_t:FY\f$), 
     * where \f$FY\f$ is the transformed multipole expansion and 
     * \f$FX\f$ is the transformed local expansion, both padded with zeros and
     * of size \f$r_c=(2\times ORDER-1)^3\f$ or \f$r_c^{opt}=\frac{r_c}{2}+1\f$ 
     * for real valued kernels. The index \f$t\f$ denotes the transfer vector 
     * of the target cell to the source cell.
     *
     * @param[in] transfer transfer vector
     * @param[in] FY transformed multipole expansion
     * @param[out] FX transformed local expansion
     * @param[in] CellWidth needed for the scaling of the compressed M2L operators which are based on a homogeneous matrix kernel computed for the reference cell width \f$w=2\f$, ie in \f$[-1,1]^3\f$.
     */
    void applyFC(const unsigned int idx, const unsigned int TreeLevel, 
                 const FReal, const unsigned int d,
                 const FComplex<FReal> *const FY, FComplex<FReal> *const FX) const
    {
        // Perform entrywise product manually
        for (unsigned int j=0; j<opt_rc; ++j){
            FX[j].addMul(FC[TreeLevel][d][idx*opt_rc + j], FY[j]);
        }

    }


    /**
     * Transform densities \f$Y= DFT(y)\f$ of a source cell. This operation
     * can be seen as part of the M2M operation.
     *
     * @param[in] y multipole expansion of size \f$\ell^3\f$
     * @param[out] Y transformed multipole expansion of size \f$r\f$
     */
    void applyZeroPaddingAndDFT(FReal *const y, FComplex<FReal> *const FY) const
    {
        FReal Py[rc];
        FBlas::setzero(rc,Py);
        // Apply Zero Padding
        for (unsigned int i=0; i<nnodes; ++i)
            Py[node_diff[i*nnodes]]=y[i];
        // Apply forward Discrete Fourier Transform
        Dft.applyDFT(Py,FY);
    }


};


#endif // FUNIFTENSORIALM2LHANDLER_HPP
 
// [--END--]
