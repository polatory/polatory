// See LICENCE file at project root

// @SCALFMM_PRIVATE

#ifndef FINTERPMATRIXKERNEL_COVARIANCE_HPP
#define FINTERPMATRIXKERNEL_COVARIANCE_HPP

#include <iostream>
#include <cstdlib>
#include <cmath> // for exp

// ScalFMM includes
#include "Utils/FNoCopyable.hpp"
#include "Utils/FMath.hpp"
#include "Utils/FPoint.hpp"
// for identifiers and types
#include "Kernels/Interpolation/FInterpMatrixKernel.hpp"



/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @date March 13th, 2014 
 */

// not extendable
enum CORRELATION_SUPPORT_EXTENSION{INFINITE,FINITE};

// TOADD
// bounded support correlation functions:
// * power model
// hole effect correlation (nonstrictly decreasing)
// Whittle model? (slow decay => important constraint on size of grid / length)
template <class FReal>
struct FAbstractCorrelationKernel : FNoCopyable
{ 
  virtual ~FAbstractCorrelationKernel(){}
  virtual FReal evaluate(const FReal*, const FReal*) const = 0;

};

/**
 * @class CORRELATIONKERNEL
 *
 * The following classes provide the evaluators for the correlation function listed below:
 * TODO Exponential decay (Matern for $\nu=1/2$): not smooth AT ALL, actually very rough. Application to Brownian motion.
 * Gaussian decay (Matern for $\nu=\infty$): infinitely smooth 
 * TODO Spherical correlation: not smooth AT ALL, but FINITE support! It is proportionnal to the intersecting area 
 * of 2 disks of radius $\ell$ whose centers are separated from $r$.
 * 
 * TODO Explicit version of Matern for $\nu=3/2$ and $\nu=5/2$ (application to machine learning)
 * Smaller values of $\nu$ lead to rough behaviours (already covered by exponential) 
 * while larger $\nu$ ($\leq 7/2$) are hard to differentiate from the Gaussian decay.
 * 
 * TODO Generic Matern for an arbitrary $\nu$ (evaluator involve Bessel functions and spectral density involves Gamma functions)
 * 
 * @tparam NAME description \f$latex symbol\f$
 */



/// Generic Gaussian correlation function
/// Special case of Matern function with $\nu \rightarrow \infty$ 
template<class FReal>
struct FInterpMatrixKernelGauss : FAbstractCorrelationKernel<FReal>
{
  static const KERNEL_FUNCTION_TYPE Type = NON_HOMOGENEOUS;
  static const CORRELATION_SUPPORT_EXTENSION Extension = INFINITE;
  static const unsigned int NCMP = 1; //< number of components
  static const unsigned int NPV  = 1; //< dim of physical values
  static const unsigned int NPOT = 1; //< dim of potentials
  static const unsigned int NRHS = 1; //< dim of mult exp
  static const unsigned int NLHS = 1; //< dim of loc exp
  FReal lengthScale_;

  FInterpMatrixKernelGauss(const FReal lengthScale = FReal(1.))
  : lengthScale_(lengthScale)
  {}

  // copy ctor
  FInterpMatrixKernelGauss(const FInterpMatrixKernelGauss& other)
  : lengthScale_(other.lengthScale_)
  {}

  // ID accessor
  static const char* getID() { return "GAUSS"; }

  static void printInfo() { std::cout << "K(x,y)=exp(-0.5*(r_i/l*r_i/l)) with r=|x-y|" << std::endl; }
  
  // returns position in reduced storage
  int getPosition(const unsigned int) const
  {return 0;}
  
  // returns coefficient of mutual interaction
  // 1 for symmetric kernels
  // -1 for antisymmetric kernels
  // somethings else if other property of symmetry
  FReal getMutualCoefficient() const{ return FReal(1.); }

  /*
   * r(x)=exp(-(|x|/l)^2)
   */
  FReal evaluate(const FReal* x, const FReal* y) const
  {
    FReal dist2 = FReal(0.0);
    for(int d=0; d<3; ++d){
      FReal distX = FMath::Abs(x[d]-y[d]) / lengthScale_;
      dist2 += distX*distX;
    }

    FReal res = FMath::Exp(FReal(-0.5)*dist2);

    return res;
  }

  // evaluate interaction
  template <class ValueClass>
  ValueClass evaluate(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                      const ValueClass& x2, const ValueClass& y2, const ValueClass& z2) const
  {
    const ValueClass diff[3] = {(x1-x2),(y1-y2),(z1-z2)};

    ValueClass dist2 = FMath::Zero<ValueClass>();
    for(int d=0; d<3; ++d){
      const ValueClass distX = diff[d] / FMath::ConvertTo<ValueClass,FReal>(lengthScale_);
      dist2 += distX*distX;
    }

    return FMath::Exp(FMath::ConvertTo<ValueClass,FReal>(-0.5)*dist2);

  }

  // evaluate interaction (blockwise)
  template <class ValueClass>
  void evaluateBlock(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                     const ValueClass& x2, const ValueClass& y2, const ValueClass& z2, ValueClass* block) const
  {
    block[0]=this->evaluate(x1,y1,z1,x2,y2,z2);
  }

  // evaluate interaction and derivative (blockwise)
  template <class ValueClass>
  void evaluateBlockAndDerivative(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                                  const ValueClass& x2, const ValueClass& y2, const ValueClass& z2,
                                  ValueClass block[1], ValueClass blockDerivative[3]) const
  {
    block[0]=this->evaluate(x1,y1,z1,x2,y2,z2);
    const ValueClass lengthScaleOpt = FMath::ConvertTo<ValueClass,FReal>(-1/(lengthScale_*lengthScale_));
    blockDerivative[0] = block[0]*(x1-x2) * lengthScaleOpt;
    blockDerivative[1] = block[0]*(y1-y2) * lengthScaleOpt;
    blockDerivative[2] = block[0]*(z1-z2) * lengthScaleOpt;
  }

  /*
   * scaling (for ScalFMM)
   */
  FReal getScaleFactor(const FReal, const int) const
  {
    // return 1 because non homogeneous kernel functions cannot be scaled!!!
    return FReal(1.);
  }

  FReal getScaleFactor(const FReal) const
  {
    // return 1 because non homogeneous kernel functions cannot be scaled!!!
    return FReal(1.);
  }

  FReal evaluate(const FPoint<FReal>& p1, const FPoint<FReal>& p2) const{
    return evaluate<FReal>(p1.getX(), p1.getY(), p1.getZ(), p2.getX(), p2.getY(), p2.getZ());
  }
  void evaluateBlock(const FPoint<FReal>& p1, const FPoint<FReal>& p2, FReal* block) const{
    evaluateBlock<FReal>(p1.getX(), p1.getY(), p1.getZ(), p2.getX(), p2.getY(), p2.getZ(), block);
  }
  void evaluateBlockAndDerivative(const FPoint<FReal>& p1, const FPoint<FReal>& p2,
                                  FReal block[1], FReal blockDerivative[3]) const {
    evaluateBlockAndDerivative<FReal>(p1.getX(), p1.getY(), p1.getZ(), p2.getX(), p2.getY(), p2.getZ(), block, blockDerivative);
  }

};






#endif /* FINTERPMATRIXKERNEL_COVARIANCE_HPP */
