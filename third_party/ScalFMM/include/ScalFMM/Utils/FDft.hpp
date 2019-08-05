// See LICENCE file at project root
#ifndef FDFT_HPP
#define FDFT_HPP

#include "FGlobal.hpp"
#ifndef SCALFMM_USE_FFT
#error The FFT header is included while SCALFMM_USE_FFT is turned OFF
#endif


#ifdef SCALFMM_USE_FFT
#include <iostream>
#include <stdlib.h>

#ifdef SCALFMM_USE_ESSL_AS_FFTW
#include <fftw3_essl.h>
#else
#include <fftw3.h>
#endif

// for @class FDft only
#include "FBlas.hpp"

#include "FComplex.hpp"


/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FDft, @class FFftw
 * Please read the license
 *
 * These classes handle the forward and backward Discete Fourier Transform
 * (DFT).
 * @class FDft implements a direct method while @class FFftw uses the Fast
 * Fourier Transform (FFT). The FFT algorithm can either be provided by the
 * FFTW(3) library itself or a version that is wrapped in Intel MKL.
 *
 * The @class FDft is templatized with the input value type (FReal or FComplex<FReal>),
 * while @class FFftw is templatized with input and output value types and the dimension.
 *
 * The aim of writing these specific classes is to allow further customization
 * of the DFT such as implementing a tensorial variant, a weighted variant
 * or any other feature.
 *
 */


/**
 * @class FDft
 *
 * @tparam size_ number of sampled values \f$N\f$
 */
template< class FReal, typename ValueType = FReal>
class FDft
{

    ValueType* data_;  //< data in physical space
    FComplex<FReal>* dataF_; //< data in Fourier space

    FReal *cosRS_, *sinRS_;

private:
    unsigned int size_; //< number of steps

    void initDFT()
    {
        // allocate arrays
        data_  = new ValueType[size_];
        dataF_ = new FComplex<FReal>[size_];

        // Beware this is extremely HEAVY to store!!! => Use FDft only for debug!
        cosRS_ = new FReal[size_*size_];
        sinRS_ = new FReal[size_*size_];

        for(unsigned int r=0; r<size_; ++r)
            for(unsigned int s=0; s<size_; ++s){
                FReal thetaRS = 2*M_PI*r*s/size_;
                cosRS_[r*size_+s]=FMath::Cos(thetaRS);
                sinRS_[r*size_+s]=FMath::Sin(thetaRS);
            }
    }

public:

    FDft(const unsigned int size)
        : size_(size)
    {
        // init DFT
        initDFT();
    }

    FDft(const FDft& other)
        : size_(other.size_)
    {
        // init DFT
        initDFT();
    }

    virtual ~FDft()
    {
        delete [] data_;
        delete [] dataF_;
        delete [] cosRS_;
        delete [] sinRS_;
    }

    /// Forward DFT
    // Real valued DFT
    void applyDFT(const FReal* sampledData,
                  FComplex<FReal>* transformedData) const
    {
        // read sampled data
        FBlas::c_setzero(size_,reinterpret_cast<FReal*>(dataF_));
        FBlas::copy(size_, sampledData,data_);

        // perform direct forward transformation
        for(unsigned int r=0; r<size_; ++r)
            for(unsigned int s=0; s<size_; ++s){
                dataF_[r] += FComplex<FReal>(data_[s]*cosRS_[r*size_+s],
                        -data_[s]*sinRS_[r*size_+s]);
            }

        // write transformed data
        FBlas::c_copy(size_,reinterpret_cast<FReal*>(dataF_),
                      reinterpret_cast<FReal*>(transformedData));
    }
    // Complexe valued DFT
    void applyDFT(const FComplex<FReal>* sampledData,
                  FComplex<FReal>* transformedData) const
    {
        // read sampled data
        FBlas::c_setzero(size_,reinterpret_cast<FReal*>(dataF_));
        FBlas::c_copy(size_,reinterpret_cast<const FReal*>(sampledData),
                      reinterpret_cast<FReal*>(data_));

        // perform direct forward transformation
        for(unsigned int r=0; r<size_; ++r)
            for(unsigned int s=0; s<size_; ++s){
                dataF_[r] += FComplex<FReal>(data_[s].getReal()*cosRS_[r*size_+s]
                        + data_[s].getImag()*sinRS_[r*size_+s],
                        data_[s].getImag()*cosRS_[r*size_+s]
                        - data_[s].getReal()*sinRS_[r*size_+s]);
            }

        // write transformed data
        FBlas::c_copy(size_,reinterpret_cast<FReal*>(dataF_),
                      reinterpret_cast<FReal*>(transformedData));
    }

    /// Backward DFT
    // Real valued IDFT
    void applyIDFT(const FComplex<FReal>* transformedData,
                   FReal* sampledData) const
    {
        // read transformed data
        FBlas::setzero(size_,data_);
        FBlas::c_copy(size_,reinterpret_cast<const FReal*>(transformedData),
                      reinterpret_cast<FReal*>(dataF_));

        // perform direct backward transformation
        for(unsigned int r=0; r<size_; ++r){
            for(unsigned int s=0; s<size_; ++s){
                data_[r] += dataF_[s].getReal()*cosRS_[r*size_+s]
                        + dataF_[s].getImag()*sinRS_[r*size_+s];
            }
            data_[r]*=1./size_;
        }

        // write sampled data
        FBlas::copy(size_,data_,sampledData);
    }

    // Complexe valued IDFT
    void applyIDFT(const FComplex<FReal>* transformedData,
                   FComplex<FReal>* sampledData) const
    {
        // read transformed data
        FBlas::c_setzero(size_,reinterpret_cast<FReal*>(data_));
        FBlas::c_copy(size_,reinterpret_cast<const FReal*>(transformedData),
                      reinterpret_cast<FReal*>(dataF_));

        // perform direct backward transformation
        for(unsigned int r=0; r<size_; ++r){
            for(unsigned int s=0; s<size_; ++s){
                data_[r] += FComplex<FReal>(dataF_[s].getReal()*cosRS_[r*size_+s]
                        - dataF_[s].getImag()*sinRS_[r*size_+s],
                        dataF_[s].getImag()*cosRS_[r*size_+s]
                        + dataF_[s].getReal()*sinRS_[r*size_+s]);
            }
            data_[r]*=1./size_;
        }

        // write sampled data
        FBlas::c_copy(size_,reinterpret_cast<FReal*>(data_),
                      reinterpret_cast<FReal*>(sampledData));
    }

};






/**
 * @class FFftw
 * 
 * @brief This class is a wrapper to FFTW. It is important to enable float if FFftw<float> is used.
 *
 *
 * @tparam ValueClassSrc Source value type
 * @tparam ValueClassDest Destination value type
 * @tparam DIM dimension \f$d\f$ of the Discrete Fourier Transform
 * @param nbPointsPerDim number of sampled values \f$N\f$ per dimension
 * @param nbPoints total number of sampled values \f$N^d\f$
 * 
 * 
 */
template <class ValueClassSrc, class ValueClassDest, class ValueClassSrcFftw, class ValueClassDestFftw, class PlanClassFftw, int DIM >
class FFftwCore {
    enum{dim = DIM};
protected:
    static void PlusEqual(double dest[], const double src[], const int nbElements){
        for(int idxVal = 0 ; idxVal < nbElements ; ++idxVal){
            dest[idxVal] += src[idxVal];
        }
    }
    static void PlusEqual(FComplex<double> dest[], const fftw_complex src[], const int nbElements){
        const double* ptrSrc = reinterpret_cast<const double*>(src);
        for(int idxVal = 0 ; idxVal < nbElements ; ++idxVal){
            dest[idxVal] += FComplex<double>(ptrSrc[2*idxVal], ptrSrc[2*idxVal+1]);
        }
    }
    static void Equal(double dest[], const double src[], const int nbElements){
        for(int idxVal = 0 ; idxVal < nbElements ; ++idxVal){
            dest[idxVal] = src[idxVal];
        }
    }
    static void Equal(FComplex<double> dest[], const fftw_complex src[], const int nbElements){
        const double* ptrSrc = reinterpret_cast<const double*>(src);
        for(int idxVal = 0 ; idxVal < nbElements ; ++idxVal){
            dest[idxVal] = FComplex<double>(ptrSrc[2*idxVal], ptrSrc[2*idxVal+1]);
        }
    }
    static void Bind_fftw_alloc(double** ptr, const int size){
        *ptr = (double*) fftw_malloc(sizeof(double) * size);
        //SpErrAEH(*ptr);
    }
    static void Bind_fftw_alloc(fftw_complex** ptr, const int size){
        *ptr = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
        //SpErrAEH(*ptr);
    }
    static void Bind_fftw_destroy_plan(fftw_plan plan){
        //#pragma omp critical(SF_KEEP_FFTW)
        //{
            fftw_destroy_plan(plan);
        //}
    }
    static fftw_plan Bind_fftw_plan_dft(int d, int *n0, fftw_complex *in, double *out, int sign, unsigned flags){
        //SpErrAE(sign == 1);
        fftw_plan plan;
        //#pragma omp critical(SF_KEEP_FFTW)
        //{
            plan = fftw_plan_dft_c2r(d, n0, in, out, flags);
        //}
        return plan;
    }
    static fftw_plan Bind_fftw_plan_dft(int d, int *n0, double *in, fftw_complex *out, int sign, unsigned flags){
        //SpErrAE(sign == -1);
        fftw_plan plan;
        //#pragma omp critical(SF_KEEP_FFTW)
        //{
            plan = fftw_plan_dft_r2c(d, n0, in, out, flags);
        //}
        return plan;
    }
    static fftw_plan Bind_fftw_plan_dft(int d, int *n0, fftw_complex *in, fftw_complex *out, int sign, unsigned flags){
        fftw_plan plan;
        //#pragma omp critical(SF_KEEP_FFTW)
        //{
            plan = fftw_plan_dft(d, n0, in, out, sign, flags);
        //}
        return plan;
    }
    static void Bind_fftw_execute(fftw_plan plan){
        fftw_execute(plan);
    }

    static void PlusEqual(float dest[], const float src[], const int nbElements){
        for(int idxVal = 0 ; idxVal < nbElements ; ++idxVal){
            dest[idxVal] += src[idxVal];
        }
    }
    static void PlusEqual(FComplex<float> dest[], const fftwf_complex src[], const int nbElements){
        const float* ptrSrc = reinterpret_cast<const float*>(src);
        for(int idxVal = 0 ; idxVal < nbElements ; ++idxVal){
            dest[idxVal] += FComplex<float>(ptrSrc[2*idxVal], ptrSrc[2*idxVal+1]);
        }
    }
    static void Equal(float dest[], const float src[], const int nbElements){
        for(int idxVal = 0 ; idxVal < nbElements ; ++idxVal){
            dest[idxVal] = src[idxVal];
        }
    }
    static void Equal(FComplex<float> dest[], const fftwf_complex src[], const int nbElements){
        const float* ptrSrc = reinterpret_cast<const float*>(src);
        for(int idxVal = 0 ; idxVal < nbElements ; ++idxVal){
            dest[idxVal] = FComplex<float>(ptrSrc[2*idxVal], ptrSrc[2*idxVal+1]);
        }
    }    
    static void Bind_fftw_alloc(float** ptr, const int size){
        *ptr = (float*) fftwf_malloc(sizeof(float) * size);
        //SpErrAEH(*ptr);
    }
    static void Bind_fftw_alloc(fftwf_complex** ptr, const int size){
        *ptr = (fftwf_complex*) fftw_malloc(sizeof(fftwf_complex) * size);
        //SpErrAEH(*ptr);
    }
    static void Bind_fftw_destroy_plan(fftwf_plan plan){
        //#pragma omp critical(SF_KEEP_FFTW)
        //{
            fftwf_destroy_plan(plan);
        //}
    }
    static fftwf_plan Bind_fftw_plan_dft(int d, int *n0, fftwf_complex *in, float *out, int /*sign*/, unsigned flags){
        fftwf_plan plan;
        //#pragma omp critical(SF_KEEP_FFTW)
        //{
            plan = fftwf_plan_dft_c2r(d, n0, in, out, flags);
        //}
        return plan;
    }
    static fftwf_plan Bind_fftw_plan_dft(int d, int *n0, float *in, fftwf_complex *out, int /*sign*/, unsigned flags){
        fftwf_plan plan;
        //#pragma omp critical(SF_KEEP_FFTW)
        //{
            plan = fftwf_plan_dft_r2c(d, n0, in, out, flags);
        //}
        return plan;
    }
    static fftwf_plan Bind_fftw_plan_dft(int d, int *n0, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags){
        fftwf_plan plan;
        //#pragma omp critical(SF_KEEP_FFTW)
        //{
            plan = fftwf_plan_dft(d, n0, in, out, sign, flags);
        //}
        return plan;
    }
    static void Bind_fftw_execute(fftwf_plan plan){
        fftwf_execute(plan);
    }
    int nbPointsPerDim; //< Number of discrete points per dimension
    int nbPoints; //< Total number of discrete points
    ValueClassSrcFftw* timeSignal; //< FFTW array for time values
    ValueClassDestFftw* freqSignal; //< FFTW array for freq values
    PlanClassFftw plan_s2d; //< backward FFT plan
    PlanClassFftw plan_d2s; //< forward FFT plan
    /** Free allocated data and set attributes to 0 */
    void releaseData(){
        nbPointsPerDim = 0;
        nbPoints       = 0;
        fftw_free(timeSignal);
        timeSignal = nullptr;
        fftw_free(freqSignal);
        freqSignal = nullptr;
        Bind_fftw_destroy_plan(plan_s2d);
        Bind_fftw_destroy_plan(plan_d2s);
    }
    /** Allocate data and set attributes values, ptr should be = nullptr */
    void allocData(const int inNbTemporalPoints){
        nbPointsPerDim = inNbTemporalPoints;
        nbPoints       = 1;
        int nbPointsArray[dim];
        for(int d = 0; d<dim;++d) {
            nbPointsArray[d] = nbPointsPerDim;
            nbPoints *= inNbTemporalPoints;
        }
        Bind_fftw_alloc(&timeSignal, nbPoints);
        Bind_fftw_alloc(&freqSignal, nbPoints);
        plan_s2d = Bind_fftw_plan_dft(dim,nbPointsArray,freqSignal,timeSignal,1,FFTW_ESTIMATE | FFTW_UNALIGNED);
        plan_d2s = Bind_fftw_plan_dft(dim,nbPointsArray,timeSignal,freqSignal,-1,FFTW_ESTIMATE | FFTW_UNALIGNED);
    }
public:
    /** Constructor with the number of discrete points in parameter */
    explicit FFftwCore(const int inNbTemporalPoints = 0)
    : nbPoints(0), timeSignal(nullptr), freqSignal(nullptr) {
        allocData(inNbTemporalPoints);
    }
    /** Copy constructor (values of buffer are also copied) */
    FFftwCore(const FFftwCore& other)
    : nbPoints(0), timeSignal(nullptr), freqSignal(nullptr) {
        allocData(other.nbPointsPerDim);
        memcpy(timeSignal, other.timeSignal, sizeof(ValueClassSrcFftw) * nbPoints);
        memcpy(freqSignal, other.freqSignal, sizeof(ValueClassDestFftw) * nbPoints);
    }
    /** Copy operator, current object may be resized (values of buffer are also copied) */
    FFftwCore& operator=(const FFftwCore& other){
        if(nbPoints != other.nbPoints){
            releaseData();
            allocData(other.nbPointsPerDim);
        }
        memcpy(timeSignal, other.timeSignal, sizeof(ValueClassSrcFftw) * nbPoints);
        memcpy(freqSignal, other.freqSignal, sizeof(ValueClassDestFftw) * nbPoints);

    }
    /** Copy r-constructor move data from given parameter object */
    FFftwCore(FFftwCore&& other)
    : nbPoints(0), timeSignal(nullptr), freqSignal(nullptr) {
        nbPointsPerDim = other.nbPointsPerDim;
        nbPoints       = other.nbPoints;
        timeSignal = other.timeSignal;
        freqSignal = other.freqSignal;
        memcpy(plan_s2d, other.plan_s2d, sizeof(plan_s2d));
        memcpy(plan_d2s, other.plan_d2s, sizeof(plan_d2s));
        other.nbPointsPerDim = 0;
        other.nbPoints       = 0;
        other.timeSignal = nullptr;
        other.freqSignal = nullptr;
        memset(other.plan_s2d, 0, sizeof(plan_s2d));
        memset(other.plan_d2s, 0, sizeof(plan_d2s));
    }
    /** Copy r-operator move data from given parameter object */
    FFftwCore& operator=(FFftwCore&& other){
        releaseData();
        nbPointsPerDim = other.nbPointsPerDim;
        nbPoints       = other.nbPoints;
        timeSignal = other.timeSignal;
        freqSignal = other.freqSignal;
        memcpy(plan_s2d, other.plan_s2d, sizeof(plan_s2d));
        memcpy(plan_d2s, other.plan_d2s, sizeof(plan_d2s));
        other.nbPointsPerDim = 0;
        other.nbPoints       = 0;
        other.timeSignal = nullptr;
        other.freqSignal = nullptr;
        memset(other.plan_s2d, 0, sizeof(plan_s2d));
        memset(other.plan_d2s, 0, sizeof(plan_d2s));
    }
    /** Release all data */
    ~FFftwCore(){
        releaseData();
    }
    /** Dealloc and realloc to the desired size */
    void resize(const int inNbTemporalPoints){
        if(nbPointsPerDim != inNbTemporalPoints){
            releaseData();
            allocData(inNbTemporalPoints);
        }
    }
    /** Compute the DFT using signalToTransform temporal values
    * The result is equal (=) to resultSignal
    */
    void applyDFT(const ValueClassSrc signalToTransform[], ValueClassDest resultSignal[]) const {
        memcpy(timeSignal, signalToTransform, sizeof(ValueClassSrc) * nbPoints);
        memset(freqSignal,0,sizeof(ValueClassDestFftw) * nbPoints);
        Bind_fftw_execute( plan_d2s );
        Equal(resultSignal, freqSignal, nbPoints);
    }
    /** Compute the inverse DFT using signalToTransform frequency values
    * The result is equal (=) to resultSignal
    */
    void applyIDFT(const ValueClassDest signalToTransform[], ValueClassSrc resultSignal[]) const {
        memcpy(freqSignal, signalToTransform, sizeof(ValueClassDest) * nbPoints);
        memset(timeSignal,0,sizeof(ValueClassSrcFftw) * nbPoints);
        Bind_fftw_execute( plan_s2d );
        Equal(resultSignal, timeSignal, nbPoints);
    }
    /** Compute the inverse DFT using signalToTransform frequency values
    * The result is equal (=) to resultSignal
    */
    void applyIDFTNorm(const ValueClassDest signalToTransform[], ValueClassSrc resultSignal[]) const {
        memcpy(freqSignal, signalToTransform, sizeof(ValueClassDest) * nbPoints);
        memset(timeSignal,0,sizeof(ValueClassSrcFftw) * nbPoints);
        Bind_fftw_execute( plan_s2d );
        Equal(resultSignal, timeSignal, nbPoints);
        normalize(resultSignal);
    }
    void applyIDFTNormConj(const ValueClassDest signalToTransform[], ValueClassSrc resultSignal[]) const {
        freqSignal[0][0] = signalToTransform[0].getReal();
        freqSignal[0][1] = signalToTransform[0].getImag();
        for(int idx = 1 ; idx <= nbPoints/2 ; ++idx){
            freqSignal[idx][0] = signalToTransform[idx].getReal();
            freqSignal[idx][1] = signalToTransform[idx].getImag();
            freqSignal[nbPoints-idx][0] =  signalToTransform[idx].getReal();
            freqSignal[nbPoints-idx][1] = -signalToTransform[idx].getImag();
        }
        memset(timeSignal,0,sizeof(ValueClassSrcFftw) * nbPoints);
        Bind_fftw_execute( plan_s2d );
        Equal(resultSignal, timeSignal, nbPoints);
        normalize(resultSignal);
    }
    /** Compute the DFT using signalToTransform temporal values
    * The result is added (+=) to resultSignal
    */
    void applyDFTAdd(const ValueClassSrc signalToTransform[], ValueClassDest resultSignal[]) const {
        memcpy(timeSignal, signalToTransform, sizeof(ValueClassSrc) * nbPoints);
        memset(freqSignal,0,sizeof(ValueClassDestFftw) * nbPoints);
        Bind_fftw_execute( plan_d2s );
        PlusEqual(resultSignal, freqSignal, nbPoints);
    }
    /** Compute the inverse DFT using signalToTransform frequency values
    * The result is added (+=) to resultSignal
    */
    void applyIDFTAdd(const ValueClassDest signalToTransform[], ValueClassSrc resultSignal[]) const {
        memcpy(freqSignal, signalToTransform, sizeof(ValueClassDest) * nbPoints);
        memset(timeSignal,0,sizeof(ValueClassSrcFftw) * nbPoints);
        Bind_fftw_execute( plan_s2d );
        PlusEqual(resultSignal, timeSignal, nbPoints);
    }
    /** Compute the inverse DFT using signalToTransform frequency values
    * The result is added (+=) to resultSignal
    */
    void applyIDFTAddNorm(const ValueClassDest signalToTransform[], ValueClassSrc resultSignal[]) const {
        memcpy(freqSignal, signalToTransform, sizeof(ValueClassDest) * nbPoints);
        memset(timeSignal,0,sizeof(ValueClassSrcFftw) * nbPoints);
        Bind_fftw_execute( plan_s2d );
        PlusEqual(resultSignal, timeSignal, nbPoints);
        normalize(resultSignal);
    }
    void applyIDFTAddNormConj(const ValueClassDest signalToTransform[], ValueClassSrc resultSignal[]) const {
        freqSignal[0][0] = signalToTransform[0].getReal();
        freqSignal[0][1] = signalToTransform[0].getImag();
        for(int idx = 1 ; idx <= nbPoints/2 ; ++idx){
            freqSignal[idx][0] = signalToTransform[idx].getReal();
            freqSignal[idx][1] = signalToTransform[idx].getImag();
            freqSignal[nbPoints-idx][0] =  signalToTransform[idx].getReal();
            freqSignal[nbPoints-idx][1] = -signalToTransform[idx].getImag();
        }
        memset(timeSignal,0,sizeof(ValueClassSrcFftw) * nbPoints);
        Bind_fftw_execute( plan_s2d );
        PlusEqual(resultSignal, timeSignal, nbPoints);
        normalize(resultSignal);
    }

    void normalize(double* resultSignal) const {
        const double realNbPoints = static_cast<double>(nbPoints);
        for(int idxVal = 0 ; idxVal < nbPoints ; ++idxVal){
            resultSignal[idxVal] /= realNbPoints;
        }
    }
    void normalize(FComplex<double>* resultSignal) const {
        const double realNbPoints = static_cast<double>(nbPoints);
        for(int idxVal = 0 ; idxVal < nbPoints ; ++idxVal){
            resultSignal[idxVal] /= realNbPoints;
        }
    }
    void normalize(float* resultSignal) const {
        const float realNbPoints = static_cast<float>(nbPoints);
        for(int idxVal = 0 ; idxVal < nbPoints ; ++idxVal){
            resultSignal[idxVal] /= realNbPoints;
        }
    }
    void normalize(FComplex<float>* resultSignal) const {
        const float realNbPoints = static_cast<float>(nbPoints);
        for(int idxVal = 0 ; idxVal < nbPoints ; ++idxVal){
            resultSignal[idxVal] /= realNbPoints;
        }
    }

    /** To know if it is the real fftw */
    bool isTrueFftw() const{
        return true;
    }
};





template <class ValueClassSrc, class ValueClassDest, int DIM = 1>
class FFftw;
//////////////////////////////////////////////////////////////////////////////
/// Double precision
//////////////////////////////////////////////////////////////////////////////
template <int DIM>
class FFftw <double, FComplex<double>, DIM> : public FFftwCore<double, FComplex<double>, double, fftw_complex, fftw_plan, DIM>{
    typedef FFftwCore<double, FComplex<double>, double, fftw_complex, fftw_plan, DIM> ParentClass;
public:
    using ParentClass::ParentClass;
};
template <int DIM>
class FFftw <FComplex<double>, FComplex<double>, DIM> : public FFftwCore<FComplex<double>, FComplex<double>, fftw_complex, fftw_complex, fftw_plan, DIM>{
    typedef FFftwCore<FComplex<double>, FComplex<double>, fftw_complex, fftw_complex, fftw_plan, DIM> ParentClass;
public:
    using ParentClass::ParentClass;
};
//////////////////////////////////////////////////////////////////////////////
/// Single precision
//////////////////////////////////////////////////////////////////////////////
template <int DIM>
class FFftw <float, FComplex<float>, DIM> : public FFftwCore<float, FComplex<float>, float, fftwf_complex, fftwf_plan, DIM>{
    typedef FFftwCore<float, FComplex<float>, float, fftwf_complex, fftwf_plan, DIM> ParentClass;
public:
    using ParentClass::ParentClass;
};
template <int DIM>
class FFftw <FComplex<float>, FComplex<float>, DIM> : public FFftwCore<FComplex<float>, FComplex<float>, fftwf_complex, fftwf_complex, fftwf_plan, DIM>{
    typedef FFftwCore<FComplex<float>, FComplex<float>, fftwf_complex, fftwf_complex, fftwf_plan, DIM> ParentClass;
public:
    using ParentClass::ParentClass;
};

#endif /*SCALFMM_USE_FFT*/
#endif /* FDFT_HPP */

