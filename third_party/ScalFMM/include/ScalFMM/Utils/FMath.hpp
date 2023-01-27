// See LICENCE file at project root
#ifndef FMATH_HPP
#define FMATH_HPP


#include <cmath>
#include <limits>
#include <numbers>

#include "FGlobal.hpp"

#ifdef SCALFMM_USE_SSE
#include "FSse.hpp"
#endif

#ifdef SCALFMM_USE_AVX
#include "FAvx.hpp"
#endif


/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * @class
 * Please read the license
 *
 * Propose basic math functions or indirections to std math.
 */
struct FMath{
    template <class FReal>
    constexpr static FReal FPi(){ return std::numbers::pi_v<FReal>; }
    template <class FReal>
    constexpr static FReal FTwoPi(){ return FReal(2.0) * std::numbers::pi_v<FReal>; }
    template <class FReal>
    constexpr static FReal FPiDiv2(){ return std::numbers::pi_v<FReal> / FReal(2.0); }
    template <class FReal>
    constexpr static FReal Epsilon(){ return FReal(0.00000000000000000001); }

    /** To get absolute value */
    template <class NumType>
    static NumType Abs(const NumType inV){
        return (inV < 0 ? -inV : inV);
    }

#ifdef SCALFMM_USE_SSE
    static __m128 Abs(const __m128 inV){
        return _mm_max_ps(_mm_sub_ps(_mm_setzero_ps(), inV), inV);
    }

    static __m128d Abs(const __m128d inV){
        return _mm_max_pd(_mm_sub_pd(_mm_setzero_pd(), inV), inV);
    }
#endif
#ifdef SCALFMM_USE_AVX
    static __m256 Abs(const __m256 inV){
        return _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), inV), inV);
    }

    static __m256d Abs(const __m256d inV){
        return _mm256_max_pd(_mm256_sub_pd(_mm256_setzero_pd(), inV), inV);
    }
#endif
#ifdef SCALFMM_USE_AVX2
#ifdef __MIC__
    static __m512 Abs(const __m512 inV){
        return _mm512_max_ps(_mm512_sub_ps(_mm512_setzero_ps(), inV), inV);
    }

    static __m512d Abs(const __m512d inV){
        return _mm512_max_pd(_mm512_sub_pd(_mm512_setzero_pd(), inV), inV);
    }
#endif
#endif

    /** To get max between 2 values */
    template <class NumType>
    static NumType Max(const NumType inV1, const NumType inV2){
        return (inV1 > inV2 ? inV1 : inV2);
    }

    /** To get min between 2 values */
    template <class NumType>
    static NumType Min(const NumType inV1, const NumType inV2){
        return (inV1 < inV2 ? inV1 : inV2);
    }

#ifdef SCALFMM_USE_SSE
    static __m128 Max(const __m128 inV1, const __m128 inV2){
        return _mm_max_ps(inV1, inV2);
    }
    static __m128 Min(const __m128 inV1, const __m128 inV2){
        return _mm_min_ps(inV1, inV2);
    }

    static __m128d Max(const __m128d inV1, const __m128d inV2){
        return _mm_max_pd(inV1, inV2);
    }
    static __m128d Min(const __m128d inV1, const __m128d inV2){
        return _mm_min_pd(inV1, inV2);
    }
#endif
#ifdef SCALFMM_USE_AVX
    static __m256 Max(const __m256 inV1, const __m256 inV2){
        return _mm256_max_ps(inV1, inV2);
    }
    static __m256 Min(const __m256 inV1, const __m256 inV2){
        return _mm256_min_ps(inV1, inV2);
    }

    static __m256d Max(const __m256d inV1, const __m256d inV2){
        return _mm256_max_pd(inV1, inV2);
    }
    static __m256d Min(const __m256d inV1, const __m256d inV2){
        return _mm256_min_pd(inV1, inV2);
    }
#endif
#ifdef SCALFMM_USE_AVX2
#ifdef __MIC__
    static __m512 Max(const __m512 inV1, const __m512 inV2){
        return _mm512_max_ps(inV1, inV2);
    }
    static __m512 Min(const __m512 inV1, const __m512 inV2){
        return _mm512_min_ps(inV1, inV2);
    }

    static __m512d Max(const __m512d inV1, const __m512d inV2){
        return _mm512_max_pd(inV1, inV2);
    }
    static __m512d Min(const __m512d inV1, const __m512d inV2){
        return _mm512_min_pd(inV1, inV2);
    }
#endif
#endif
    /** To know if 2 values seems to be equal */
    template <class NumType>
    static bool LookEqual(const NumType inV1, const NumType inV2){
        return (Abs(inV1-inV2) < std::numeric_limits<NumType>::epsilon());
        //const FReal relTol = FReal(0.00001);
        //const FReal absTol = FReal(0.00001);
        //return (Abs(inV1 - inV2) <= Max(absTol, relTol * Max(Abs(inV1), Abs(inV2))));
    }

    /** To know if 2 values seems to be equal */
    template <class NumType>
    static NumType RelatifDiff(const NumType inV1, const NumType inV2){
        return Abs(inV1 - inV2)*Abs(inV1 - inV2)/Max(Abs(inV1*inV1), Abs(inV2*inV2));
    }

    /** To get floor of a FReal */
    static float dfloor(const float inValue){
        return floorf(inValue);
    }
    static double dfloor(const double inValue){
        return floor(inValue);
    }

    /** To get ceil of a FReal */
    static float Ceil(const float inValue){
        return ceilf(inValue);
    }
    static double Ceil(const double inValue){
        return ceil(inValue);
    }

#if  defined(SCALFMM_USE_SSE ) && defined(__SSSE4_1__)
    static __m128 dfloor(const __m128 inV){
        return _mm_floor_ps(inV);
    }

    static __m128d dfloor(const __m128d inV){
        return _mm_floor_pd(inV);
    }

    static __m128 Ceil(const __m128 inV){
        return _mm_ceil_ps(inV);
    }

    static __m128d Ceil(const __m128d inV){
        return _mm_ceil_pd(inV);
    }
#endif
#ifdef SCALFMM_USE_AVX
    static __m256 dfloor(const __m256 inV){
        return _mm256_floor_ps(inV);
    }

    static __m256d dfloor(const __m256d inV){
        return _mm256_floor_pd(inV);
    }

    static __m256 Ceil(const __m256 inV){
        return _mm256_ceil_ps(inV);
    }

    static __m256d Ceil(const __m256d inV){
        return _mm256_ceil_pd(inV);
    }
#endif
#ifdef SCALFMM_USE_AVX2
#ifdef __MIC__
    static __m512 dfloor(const __m512 inV){
        return _mm512_floor_ps(inV);
    }

    static __m512d dfloor(const __m512d inV){
        return _mm512_floor_pd(inV);
    }

    static __m512 Ceil(const __m512 inV){
        return _mm512_ceil_ps(inV);
    }

    static __m512d Ceil(const __m512d inV){
        return _mm512_ceil_pd(inV);
    }
#endif
#endif
    /** To get pow */
    static double pow(double x, double y){
        return ::pow(x,y);
    }
    static float pow(float x, float y){
        return ::powf(x,y);
    }
    template <class NumType>
    static NumType pow(const NumType inValue, int power){
        NumType result = 1;
        while(power-- > 0) result *= inValue;
        return result;
    }

    /** To get pow of 2 */
    static int pow2(const int power){
        return (1 << power);
    }

    /** To get factorial */
    template <class NumType>
    static double factorial(int inValue){
        if(inValue==0) return NumType(1.);
        else {
            NumType result = NumType(inValue);
            while(--inValue > 1) result *= NumType(inValue);
            return result;
        }
    }

    /** To get exponential */
    static double Exp(double x){
        return ::exp(x);
    }
    static float Exp(float x){
        return ::expf(x);
    }

    /** To know if a value is between two others */
    template <class NumType>
    static bool Between(const NumType inValue, const NumType inMin, const NumType inMax){
        return ( inMin <= inValue && inValue < inMax );
    }
    /** To compute fmadd operations **/
    template <class NumType>
    static NumType FMAdd(const NumType a, const NumType b, const NumType c){
	return a * b + c;
    }


#if  defined(SCALFMM_USE_SSE ) && defined(__SSSE4_1__)
    static __m128 FMAdd(const __m128 inV1, const __m128 inV2, const __m128 inV3){
        return _mm_add_ps( _mm_mul_ps(inV1,inV2), inV3);
    }

    static __m128d FMAdd(const __m128d inV1, const __m128d inV2, const __m128d inV3){
        return _mm_add_pd( _mm_mul_pd(inV1,inV2), inV3);
    }

#endif
#ifdef SCALFMM_USE_AVX
    static __m256 FMAdd(const __m256 inV1, const __m256 inV2, const __m256 inV3){
        return _mm256_add_ps( _mm256_mul_ps(inV1,inV2), inV3);
    }

    static __m256d FMAdd(const __m256d inV1, const __m256d inV2, const __m256d inV3){
        return _mm256_add_pd( _mm256_mul_pd(inV1,inV2), inV3);
    }

#endif
#ifdef SCALFMM_USE_AVX2
#ifdef __MIC__
    static __m512 FMAdd(const __m512 inV1, const __m512 inV2, const __m512 inV3){
        //return _mm512_add_ps( _mm512_mul_ps(inV1,inV2), inV3);
        return _mm512_fmadd_ps(inV1, inV2, inV3);
    }

    static __m512d FMAdd(const __m512d inV1, const __m512d inV2, const __m512d inV3){
        //return _mm512_add_pd( _mm512_mul_pd(inV1,inV2), inV3);
        return _mm512_fmadd_pd(inV1, inV2, inV3);
    }
#endif
#endif
    /** To get sqrt of a FReal */
    static float Sqrt(const float inValue){
        return sqrtf(inValue);
    }
    static double Sqrt(const double inValue){
        return sqrt(inValue);
    }
    static float Rsqrt(const float inValue){
        return float(1.0)/sqrtf(inValue);
    }
    static double Rsqrt(const double inValue){
        return 1.0/sqrt(inValue);
    }
#ifdef SCALFMM_USE_SSE
    static __m128 Exp(const __m128 inV){
         float ptr[4];
        _mm_storeu_ps(ptr, inV);
        for(int idx = 0 ; idx < 4 ; ++idx){
            ptr[idx] = std::exp(ptr[idx]);
        }
        return _mm_loadu_ps(ptr);
    }

    static __m128d Exp(const __m128d inV){
         double ptr[2];
        _mm_storeu_pd(ptr, inV);
        for(int idx = 0 ; idx < 2 ; ++idx){
            ptr[idx] = std::exp(ptr[idx]);
        }
        return _mm_loadu_pd(ptr);
    }

    static __m128 Sqrt(const __m128 inV){
        return _mm_sqrt_ps(inV);
    }

    static __m128d Sqrt(const __m128d inV){
        return _mm_sqrt_pd(inV);
    }

    static __m128 Rsqrt(const __m128 inV){
        return _mm_rsqrt_ps(inV);
    }

    static __m128d Rsqrt(const __m128d inV){
        return _mm_set_pd1(1.0) / _mm_sqrt_pd(inV);
    }
#endif
#ifdef SCALFMM_USE_AVX
    static __m256 Exp(const __m256 inV){
         float ptr[8];
        _mm256_storeu_ps(ptr, inV);
        for(int idx = 0 ; idx < 8 ; ++idx){
            ptr[idx] = std::exp(ptr[idx]);
        }
        return _mm256_loadu_ps(ptr);
    }

    static __m256d Exp(const __m256d inV){
         double ptr[4];
        _mm256_storeu_pd(ptr, inV);
        for(int idx = 0 ; idx < 4 ; ++idx){
            ptr[idx] = std::exp(ptr[idx]);
        }
        return _mm256_loadu_pd(ptr);
    }

    static __m256 Sqrt(const __m256 inV){
        return _mm256_sqrt_ps(inV);
    }

    static __m256d Sqrt(const __m256d inV){
        return _mm256_sqrt_pd(inV);
    }

    static __m256 Rsqrt(const __m256 inV){
        return _mm256_rsqrt_ps(inV);
    }

    static __m256d Rsqrt(const __m256d inV){
        return _mm256_set1_pd(1.0) / _mm256_sqrt_pd(inV);
    }
#endif
#ifdef SCALFMM_USE_AVX2
#ifdef __MIC__
    static __m512 Exp(const __m512 inV){
         float ptr[16];
        _mm512_storeu_ps(ptr, inV);
        for(int idx = 0 ; idx < 16 ; ++idx){
            ptr[idx] = std::exp(ptr[idx]);
        }
        return _mm512_loadu_ps(ptr);
    }

    static __m512d Exp(const __m512d inV){
         double ptr[8];
        _mm512_storeu_pd(ptr, inV);
        for(int idx = 0 ; idx < 8 ; ++idx){
            ptr[idx] = std::exp(ptr[idx]);
        }
        return _mm512_loadu_pd(ptr);
    }

    static __m512d Exp(const __m512d inV){
        return _mm512_sqrt_pd(inV);
    }

    static __m512 Sqrt(const __m512 inV){
        return _mm512_sqrt_ps(inV);
    }

    static __m512d Sqrt(const __m512d inV){
        return _mm512_sqrt_pd(inV);
    }

    static __m512 Rsqrt(const __m512 inV){
        return _mm512_rsqrt_ps(inV);
    }

    static __m512d Rsqrt(const __m512d inV){
        return _mm512_set1_pd(1.0) / _mm512_sqrt_pd(inV);
    }
#endif
#endif
    /** To get Log of a FReal */
    static float Log(const float inValue){
        return logf(inValue);
    }
    static double Log(const double inValue){
        return log(inValue);
    }

    /** To get Log2 of a FReal */
    static float Log2(const float inValue){
        return log2f(inValue);
    }
    static double Log2(const double inValue){
        return log2(inValue);
    }

    /** To get atan2 of a 2 FReal,  The return value is given in radians and is in the
      range -pi to pi, inclusive.  */
    static float Atan2(const float inValue1,const float inValue2){
        return atan2f(inValue1,inValue2);
    }
    static double Atan2(const double inValue1,const double inValue2){
        return atan2(inValue1,inValue2);
    }

    /** To get sin of a FReal */
    static float Sin(const float inValue){
        return sinf(inValue);
    }
    static double Sin(const double inValue){
        return sin(inValue);
    }

    /** To get asinf of a float. The result is in the range [0, pi]*/
    static float ASin(const float inValue){
        return asinf(inValue);
    }
    /** To get asinf of a double. The result is in the range [0, pi]*/
    static double ASin(const double inValue){
        return asin(inValue);
    }

    /** To get cos of a FReal */
    static float Cos(const float inValue){
        return cosf(inValue);
    }
    static double Cos(const double inValue){
        return cos(inValue);
    }

    /** To get arccos of a float. The result is in the range [0, pi]*/
    static float ACos(const float inValue){
        return acosf(inValue);
    }
    /** To get arccos of a double. The result is in the range [0, pi]*/
    static double ACos(const double inValue){
        return acos(inValue);
    }

    /** To get atan2 of a 2 FReal */
    static float Fmod(const float inValue1,const float inValue2){
        return fmodf(inValue1,inValue2);
    }
    /** return the floating-point remainder of inValue1  / inValue2 */
    static double Fmod(const double inValue1,const double inValue2){
        return fmod(inValue1,inValue2);
    }

    /** To know if a variable is nan, based on the C++0x */
    template <class TestClass>
    static bool IsNan(const TestClass& value){
        //volatile const TestClass* const pvalue = &value;
        //return (*pvalue) != value;
        return std::isnan(value);
    }

    /** To know if a variable is not inf, based on the C++0x */
    template <class TestClass>
    static bool IsFinite(const TestClass& value){
        // return !(value <= std::numeric_limits<T>::min()) && !(std::numeric_limits<T>::max() <= value);
        return std::isfinite(value);
    }

    template <class NumType>
    static NumType Zero();

    template <class NumType>
    static NumType One();

    template <class DestType, class SrcType>
    static DestType ConvertTo(const SrcType val);


    /** A class to compute accuracy */
    template <class FReal, class IndexType = FSize>
    class FAccurater {
        IndexType    nbElements;
        FReal l2Dot;
        FReal l2Diff;
        FReal max;
        FReal maxDiff;
    public:
        FAccurater() : nbElements(0),l2Dot(0.0), l2Diff(0.0), max(0.0), maxDiff(0.0) {
        }
        /** with inital values */
        FAccurater(const FReal inGood[], const FReal inBad[], const IndexType nbValues)
            :  nbElements(0),l2Dot(0.0), l2Diff(0.0), max(0.0), maxDiff(0.0)  {
            add(inGood, inBad, nbValues);
        }


        /** Add value to the current list */
        void add(const FReal inGood, const FReal inBad){
            l2Diff          += (inBad - inGood) * (inBad - inGood);
            l2Dot          += inGood * inGood;
            max               = Max(max , Abs(inGood));
            maxDiff         = Max(maxDiff, Abs(inGood-inBad));
            nbElements += 1 ;
        }
        /** Add array of values */
        void add(const FReal inGood[], const FReal inBad[], const IndexType nbValues){
            for(IndexType idx = 0 ; idx < nbValues ; ++idx){
                add(inGood[idx],inBad[idx]);
            }
            nbElements += nbValues ;
        }

        /** Add an accurater*/
        void add(const FAccurater& inAcc){
            l2Diff += inAcc.getl2Diff();
            l2Dot +=  inAcc.getl2Dot();
            max = Max(max,inAcc.getmax());
            maxDiff = Max(maxDiff,inAcc.getInfNorm());
            nbElements += inAcc.getNbElements();
        }

        FReal getl2Diff() const{
            return l2Diff;
        }
        FReal getl2Dot() const{
            return l2Dot;
        }
        FReal getmax() const{
            return max;
        }
        IndexType getNbElements() const{
            return nbElements;
        }
        void  setNbElements(const IndexType & n) {
            nbElements = n;
        }

        /** Get the root mean squared error*/
        FReal getL2Norm() const{
            return Sqrt(l2Diff );
        }
        /** Get the L2 norm */
        FReal getRMSError() const{
            return Sqrt(l2Diff /static_cast<FReal>(nbElements));
        }

        /** Get the inf norm */
        FReal getInfNorm() const{
            return maxDiff;
        }
        /** Get the L2 norm */
        FReal getRelativeL2Norm() const{
            return Sqrt(l2Diff / l2Dot);
        }
        /** Get the inf norm */
        FReal getRelativeInfNorm() const{
            return maxDiff / max;
        }
        /** Print */
        template <class StreamClass>
        friend StreamClass& operator<<(StreamClass& output, const FAccurater& inAccurater){
            output << "[Error] Relative L2Norm = " << inAccurater.getRelativeL2Norm() << " \t RMS Norm = " << inAccurater.getRMSError() << " \t Relative Inf = " << inAccurater.getRelativeInfNorm();
            return output;
        }

        void reset()
        {
            l2Dot          = FReal(0);
            l2Diff          = FReal(0);;
            max            = FReal(0);
            maxDiff      = FReal(0);
            nbElements = 0 ;
        }
    };
};

template <>
inline float FMath::Zero<float>(){
    return float(0.0);
}

template <>
inline double FMath::Zero<double>(){
    return double(0.0);
}

template <>
inline float FMath::One<float>(){
    return float(1.0);
}

template <>
inline double FMath::One<double>(){
    return double(1.0);
}

template <>
inline float FMath::ConvertTo<float,float>(const float val){
    return val;
}

template <>
inline double FMath::ConvertTo<double,double>(const double val){
    return val;
}

template <>
inline float FMath::ConvertTo<float,const float*>(const float* val){
    return *val;
}

template <>
inline double FMath::ConvertTo<double,const double*>(const double* val){
    return *val;
}

#ifdef SCALFMM_USE_SSE
template <>
inline __m128 FMath::One<__m128>(){
    return _mm_set_ps1(1.0);
}

template <>
inline __m128d FMath::One<__m128d>(){
    return _mm_set_pd1(1.0);
}

template <>
inline __m128 FMath::Zero<__m128>(){
    return _mm_setzero_ps();
}

template <>
inline __m128d FMath::Zero<__m128d>(){
    return _mm_setzero_pd();
}

template <>
inline __m128 FMath::ConvertTo<__m128,float>(const float val){
    return _mm_set_ps1(val);
}

template <>
inline __m128d FMath::ConvertTo<__m128d,double>(const double val){
    return _mm_set_pd1(val);
}

template <>
inline __m128 FMath::ConvertTo<__m128,const float*>(const float* val){
    return _mm_load1_ps(val);
}

template <>
inline __m128d FMath::ConvertTo<__m128d,const double*>(const double* val){
    return _mm_load1_pd(val);
}

template <>
inline float FMath::ConvertTo<float,__m128>(const __m128 val){
    __attribute__((aligned(16))) float buffer[4];
    _mm_store_ps(buffer, val);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3];
}

template <>
inline double FMath::ConvertTo<double,__m128d>(const __m128d val){
    __attribute__((aligned(16))) double buffer[2];
    _mm_store_pd(buffer, val);
    return buffer[0] + buffer[1];
}
#endif

#ifdef SCALFMM_USE_AVX
template <>
inline __m256 FMath::One<__m256>(){
    return _mm256_set1_ps(1.0);
}

template <>
inline __m256d FMath::One<__m256d>(){
    return _mm256_set1_pd(1.0);
}

template <>
inline __m256 FMath::Zero<__m256>(){
    return _mm256_setzero_ps();
}

template <>
inline __m256d FMath::Zero<__m256d>(){
    return _mm256_setzero_pd();
}

template <>
inline __m256 FMath::ConvertTo<__m256,float>(const float val){
    return _mm256_set1_ps(val);
}

template <>
inline __m256d FMath::ConvertTo<__m256d,double>(const double val){
    return _mm256_set1_pd(val);
}

template <>
inline __m256 FMath::ConvertTo<__m256,const float*>(const float* val){
    return _mm256_broadcast_ss(val);
}

template <>
inline __m256d FMath::ConvertTo<__m256d,const double*>(const double* val){
    return _mm256_broadcast_sd(val);
}

template <>
inline float FMath::ConvertTo<float,__m256>(const __m256 val){
    __attribute__((aligned(32))) float buffer[8];
    _mm256_store_ps(buffer, val);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
}

template <>
inline double FMath::ConvertTo<double,__m256d>(const __m256d val){
    __attribute__((aligned(32))) double buffer[4];
    _mm256_store_pd(buffer, val);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3];
}
#endif
#ifdef SCALFMM_USE_AVX2
#ifdef __MIC__
template <>
inline __m512 FMath::One<__m512>(){
    return _mm512_set1_ps(1.0);
}

template <>
inline __m512d FMath::One<__m512d>(){
    return _mm512_set1_pd(1.0);
}

template <>
inline __m512 FMath::Zero<__m512>(){
    return _mm512_setzero_ps();
}

template <>
inline __m512d FMath::Zero<__m512d>(){
    return _mm512_setzero_pd();
}

template <>
inline __m512 FMath::ConvertTo<__m512,__attribute__((aligned(64))) float>(const float val){
    return _mm512_set1_ps(val);
}

template <>
inline __m512d FMath::ConvertTo<__m512d,__attribute__((aligned(64))) double>(const double val){
    return _mm512_set1_pd(val);
}

template <>
inline __m512 FMath::ConvertTo<__m512,const __attribute__((aligned(64))) float*>(const float* val){
    return _mm512_set1_ps(val[0]);
}

template <>
inline __m512d FMath::ConvertTo<__m512d,const __attribute__((aligned(64))) double*>(const double* val){
    return _mm512_set1_pd(val[0]);
}

template <>
inline float FMath::ConvertTo<float,__m512>(const __m512 val){
    __attribute__((aligned(64))) float buffer[16];
    _mm512_store_ps(buffer, val);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7] + buffer[8] + buffer[9] + buffer[10] + buffer[11] + buffer[12] + buffer[13] + buffer[14] + buffer[15];
}

template <>
inline double FMath::ConvertTo<double,__m512d>(const __m512d val){
    __attribute__((aligned(64))) double buffer[8];
    _mm512_store_pd(buffer, val);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
}
#endif
#endif

#endif //FMATH_HPP
