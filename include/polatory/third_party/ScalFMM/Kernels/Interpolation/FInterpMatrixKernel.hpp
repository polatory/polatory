// See LICENCE file at project root
#ifndef FINTERPMATRIXKERNEL_HPP
#define FINTERPMATRIXKERNEL_HPP

#include <iostream>
#include <stdexcept>

#include "../../Utils/FPoint.hpp"
#include "../../Utils/FNoCopyable.hpp"
#include "../../Utils/FMath.hpp"
#include "../../Utils/FGlobal.hpp"


// probably not extendable :)
enum KERNEL_FUNCTION_TYPE {HOMOGENEOUS, NON_HOMOGENEOUS};


/**
 * @author Matthias Messner (matthias.messner@inria.fr)
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FInterpMatrixKernels
 * Please read the license
 *
 * This class provides the evaluators and scaling functions of the matrix
 * kernels. A matrix kernel should be understood in the sense of a kernel
 * of interaction (or the fundamental solution of a given equation).
 * It can either be scalar (NCMP=1) or tensorial (NCMP>1) depending on the
 * dimension of the equation considered. NCMP denotes the number of components
 * that are actually stored (e.g. 6 for a \f$3\times3\f$ symmetric tensor).
 *
 * Notes on the application scheme:
 * Let there be a kernel \f$K\f$ such that \f$Potential_i(X)X=K_{ij}(X,Y)PhysicalValue_j(Y)\f$
 * with \f$Potential\f$ the lhs of size NLHS and \f$PhysicalValues\f$ the rhs of size NRHS.
 * The table applyTab provides the indices in the reduced storage table
 * corresponding to the application scheme depicted earlier.
 *
 * PB: BEWARE! Homogeneous matrix kernels do not support cell width extension
 * yet. Is it possible to find a reference width and a scale factor such that
 * only 1 set of M2L opt can be used for all levels??
 *
 */
template <class FReal>
struct FInterpAbstractMatrixKernel : FNoCopyable
{ 
    virtual ~FInterpAbstractMatrixKernel(){} // to remove warning
    //virtual FReal evaluate(const FPoint<FReal>&, const FPoint<FReal>&) const = 0;
    // I need both functions because required arguments are not always given
    virtual FReal getScaleFactor(const FReal, const int) const = 0;
    virtual FReal getScaleFactor(const FReal) const = 0;
};


/// One over r
template <class FReal>
struct FInterpMatrixKernelR : FInterpAbstractMatrixKernel<FReal>
{
    static const KERNEL_FUNCTION_TYPE Type = HOMOGENEOUS;
    static const unsigned int NCMP = 1; //< number of components
    static const unsigned int NPV  = 1; //< dim of physical values
    static const unsigned int NPOT = 1; //< dim of potentials
    static const unsigned int NRHS = 1; //< dim of mult exp
    static const unsigned int NLHS = 1; //< dim of loc exp

    FInterpMatrixKernelR() {}

    // copy ctor
    FInterpMatrixKernelR(const FInterpMatrixKernelR& /*other*/) {}

    static const char* getID() { return "ONE_OVER_R"; }

    static void printInfo() { std::cout << "K(x,y)=1/r with r=|x-y|" << std::endl; }

    // returns position in reduced storage
    int getPosition(const unsigned int) const
    {return 0;}

    // returns coefficient of mutual interaction
    // 1 for symmetric kernels
    // -1 for antisymmetric kernels
    // Something else if other property of symmetry
    FReal getMutualCoefficient() const{ return FReal(1.); }

    // evaluate interaction
    template <class ValueClass>
    ValueClass evaluate(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                        const ValueClass& xs, const ValueClass& ys, const ValueClass& zs) const
    {
        // diff = t-s
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        return FMath::One<ValueClass>() / FMath::Sqrt(diffx*diffx + diffy*diffy + diffz*diffz);
    }

    // evaluate interaction (blockwise)
    template <class ValueClass>
    void evaluateBlock(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                       const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                       ValueClass* block) const
    {
        block[0] = this->evaluate(xt,yt,zt,xs,ys,zs);
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt,
                                    const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                                    ValueClass block[1], ValueClass blockDerivative[3]) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        const ValueClass one_over_r = FMath::One<ValueClass>() / FMath::Sqrt(diffx*diffx + diffy*diffy + diffz*diffz);

        const ValueClass one_over_r3 = one_over_r*one_over_r*one_over_r;

        block[0] = one_over_r;

        blockDerivative[0] = - one_over_r3 * diffx;
        blockDerivative[1] = - one_over_r3 * diffy;
        blockDerivative[2] = - one_over_r3 * diffz;
    }

    FReal getScaleFactor(const FReal RootCellWidth, const int TreeLevel) const
    {
        const FReal CellWidth(RootCellWidth / FReal(FMath::pow(2, TreeLevel)));
        return getScaleFactor(CellWidth);
    }

    FReal getScaleFactor(const FReal CellWidth) const
    {
        return FReal(2.) / CellWidth;
    }

    FReal evaluate(const FPoint<FReal>& pt, const FPoint<FReal>& ps) const {
        return evaluate<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ());
    }
    void evaluateBlock(const FPoint<FReal>& pt, const FPoint<FReal>& ps, FReal* block) const{
        evaluateBlock<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block);
    }
    void evaluateBlockAndDerivative(const FPoint<FReal>& pt, const FPoint<FReal>& ps,
                                    FReal block[1], FReal blockDerivative[3]) const {
        evaluateBlockAndDerivative<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block, blockDerivative);
    }
};

/// One over r when the box size is rescaled to 1
template <class FReal>
struct FInterpMatrixKernelRH :FInterpMatrixKernelR<FReal>{
    static const KERNEL_FUNCTION_TYPE Type = HOMOGENEOUS;
    static const unsigned int NCMP = 1; //< number of components
    static const unsigned int NPV  = 1; //< dim of physical values
    static const unsigned int NPOT = 1; //< dim of potentials
    static const unsigned int NRHS = 1; //< dim of mult exp
    static const unsigned int NLHS = 1; //< dim of loc exp
    FReal LX,LY,LZ ;

    FInterpMatrixKernelRH() 
    : LX(1.0),LY(1.0),LZ(1.0)
    { }

    // copy ctor
    FInterpMatrixKernelRH(const FInterpMatrixKernelRH& other)
    : FInterpMatrixKernelR<FReal>(other), LX(other.LX), LY(other.LY), LZ(other.LZ)
    {}

    static const char* getID() { return "ONE_OVER_RH"; }

    static void printInfo() { std::cout << "K(x,y)=1/rh with rh=sqrt(L_i*(x_i-y_i)^2)" << std::endl; }

    // evaluate interaction
    template <class ValueClass>
    ValueClass evaluate(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                        const ValueClass& xs, const ValueClass& ys, const ValueClass& zs) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        return FMath::One<ValueClass>() / FMath::Sqrt(FMath::ConvertTo<ValueClass,FReal>(LX)*diffx*diffx +
                                       FMath::ConvertTo<ValueClass,FReal>(LY)*diffy*diffy +
                                       FMath::ConvertTo<ValueClass,FReal>(LZ)*diffz*diffz);
    }
    void setCoeff(const FReal& a,  const FReal& b, const FReal& c)
    {LX= a*a ; LY = b*b ; LZ = c *c;}
    // returns position in reduced storage
    int getPosition(const unsigned int) const
    {return 0;}
    // returns coefficient of mutual interaction
    // 1 for symmetric kernels
    // -1 for antisymmetric kernels
    // Something else if other property of symmetry
    FReal getMutualCoefficient() const{ return FReal(1.); }

    template <class ValueClass>
    void evaluateBlock(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                       const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                       ValueClass* block) const
    {
        block[0]=this->evaluate(xt,yt,zt,xs,ys,zs);
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt,
                                    const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                                    ValueClass block[1], ValueClass blockDerivative[3]) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        const ValueClass one_over_rL = FMath::One<ValueClass>() / FMath::Sqrt(FMath::ConvertTo<ValueClass,FReal>(LX)*diffx*diffx +
                                                          FMath::ConvertTo<ValueClass,FReal>(LY)*diffy*diffy +
                                                          FMath::ConvertTo<ValueClass,FReal>(LZ)*diffz*diffz);
        const ValueClass one_over_rL3 = one_over_rL*one_over_rL*one_over_rL;

        block[0] = one_over_rL;

        blockDerivative[0] = FMath::ConvertTo<ValueClass,FReal>(LX) * one_over_rL3 * diffx;
        blockDerivative[1] = FMath::ConvertTo<ValueClass,FReal>(LY)* one_over_rL3 * diffy;
        blockDerivative[2] = FMath::ConvertTo<ValueClass,FReal>(LZ)* one_over_rL3 * diffz;

    }

    FReal getScaleFactor(const FReal RootCellWidth, const int TreeLevel) const
    {
        const FReal CellWidth(RootCellWidth / FReal(FMath::pow(2, TreeLevel)));
        return getScaleFactor(CellWidth);
    }

    FReal getScaleFactor(const FReal CellWidth) const
    {
        return FReal(2.) / CellWidth;
    }

    FReal evaluate(const FPoint<FReal>& pt, const FPoint<FReal>& ps) const{
        return evaluate<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ());
    }
    void evaluateBlock(const FPoint<FReal>& pt, const FPoint<FReal>& ps, FReal* block) const{
        evaluateBlock<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block);
    }
    void evaluateBlockAndDerivative(const FPoint<FReal>& pt, const FPoint<FReal>& ps,
                                    FReal block[1], FReal blockDerivative[3]) const {
        evaluateBlockAndDerivative<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block, blockDerivative);
    }
};


/// One over r^2
template <class FReal>
struct FInterpMatrixKernelRR : FInterpAbstractMatrixKernel<FReal>
{
    static const KERNEL_FUNCTION_TYPE Type = HOMOGENEOUS;
    static const unsigned int NCMP = 1; //< number of components
    static const unsigned int NPV  = 1; //< dim of physical values
    static const unsigned int NPOT = 1; //< dim of potentials
    static const unsigned int NRHS = 1; //< dim of mult exp
    static const unsigned int NLHS = 1; //< dim of loc exp

    FInterpMatrixKernelRR() {}

    // copy ctor
    FInterpMatrixKernelRR(const FInterpMatrixKernelRR& /*other*/) {}

    static const char* getID() { return "ONE_OVER_R_SQUARED"; }

    static void printInfo() { std::cout << "K(x,y)=1/r^2 with r=|x-y|" << std::endl; }

    // returns position in reduced storage
    int getPosition(const unsigned int) const
    {return 0;}

    // returns coefficient of mutual interaction
    // 1 for symmetric kernels
    // -1 for antisymmetric kernels
    // Something else if other property of symmetry
    FReal getMutualCoefficient() const{ return FReal(1.); }

    // evaluate interaction
    template <class ValueClass>
    ValueClass evaluate(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                        const ValueClass& xs, const ValueClass& ys, const ValueClass& zs) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        return FMath::One<ValueClass>() / FReal(diffx*diffx+diffy*diffy+diffz*diffz);
    }

    // evaluate interaction (blockwise)
    template <class ValueClass>
    void evaluateBlock(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                       const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                       ValueClass* block) const
    {
        block[0]=this->evaluate(xt,yt,zt,xs,ys,zs);
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt,
                                    const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                                    ValueClass block[1], ValueClass blockDerivative[3]) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        const ValueClass r2 = (diffx*diffx+diffy*diffy+diffz*diffz);
        const ValueClass one_over_r2 = FMath::One<ValueClass>() / (r2);
        const ValueClass one_over_r4 = one_over_r2*one_over_r2;

        block[0] = one_over_r2;

        const ValueClass coef = FMath::ConvertTo<ValueClass,FReal>(-2.) * one_over_r4;
        blockDerivative[0] = coef * diffx;
        blockDerivative[1] = coef * diffy;
        blockDerivative[2] = coef * diffz;

    }

    FReal getScaleFactor(const FReal RootCellWidth, const int TreeLevel) const
    {
        const FReal CellWidth(RootCellWidth / FReal(FMath::pow(2, TreeLevel)));
        return getScaleFactor(CellWidth);
    }

    FReal getScaleFactor(const FReal CellWidth) const
    {
        return FReal(4.) / (CellWidth*CellWidth);
    }

    FReal evaluate(const FPoint<FReal>& pt, const FPoint<FReal>& ps) const{
        return evaluate<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ());
    }
    void evaluateBlock(const FPoint<FReal>& pt, const FPoint<FReal>& ps, FReal* block) const{
        evaluateBlock<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block);
    }
    void evaluateBlockAndDerivative(const FPoint<FReal>& pt, const FPoint<FReal>& ps,
                                    FReal block[1], FReal blockDerivative[3]) const {
        evaluateBlockAndDerivative<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block, blockDerivative);
    }
};



/// One over r^12 - One over r^6
template <class FReal>
struct FInterpMatrixKernelLJ : FInterpAbstractMatrixKernel<FReal>
{
    static const KERNEL_FUNCTION_TYPE Type = NON_HOMOGENEOUS;
    static const unsigned int NCMP = 1; //< number of components
    static const unsigned int NPV  = 1; //< dim of physical values
    static const unsigned int NPOT = 1; //< dim of potentials
    static const unsigned int NRHS = 1; //< dim of mult exp
    static const unsigned int NLHS = 1; //< dim of loc exp

    FInterpMatrixKernelLJ() {}

    // copy ctor
    FInterpMatrixKernelLJ(const FInterpMatrixKernelLJ& /*other*/) {}

    static const char* getID() { return "LENNARD_JONES_POTENTIAL"; }

    static void printInfo() { std::cout << "K(x,y)=1/r with r=|x-y|" << std::endl; }

    // returns position in reduced storage
    int getPosition(const unsigned int) const
    {return 0;}

    // returns coefficient of mutual interaction
    // 1 for symmetric kernels
    // -1 for antisymmetric kernels
    // somethings else if other property of symmetry
    FReal getMutualCoefficient() const{ return FReal(1.); }

    // evaluate interaction
    template <class ValueClass>
    ValueClass evaluate(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                        const ValueClass& xs, const ValueClass& ys, const ValueClass& zs) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        const ValueClass r = FMath::Sqrt(diffx*diffx+diffy*diffy+diffz*diffz);
        const ValueClass r3 = r*r*r;
        const ValueClass one_over_r6 = FMath::One<ValueClass>() / (r3*r3);
        //return one_over_r6 * one_over_r6;
        //return one_over_r6;
        return one_over_r6 * one_over_r6 - one_over_r6;
    }

    // evaluate interaction (blockwise)
    template <class ValueClass>
    void evaluateBlock(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                       const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                       ValueClass* block) const
    {
        block[0]=this->evaluate(xt,yt,zt,xs,ys,zs);
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt,
                                    const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                                    ValueClass block[1], ValueClass blockDerivative[3]) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        const ValueClass r = FMath::Sqrt(diffx*diffx+diffy*diffy+diffz*diffz);
        const ValueClass r2 = r*r;
        const ValueClass r3 = r2*r;
        const ValueClass one_over_r6 = FMath::One<ValueClass>() / (r3*r3);
        const ValueClass one_over_r8 = one_over_r6 / (r2);

        block[0] = one_over_r6 * one_over_r6 - one_over_r6;

        const FReal coef = FMath::ConvertTo<ValueClass,FReal>(12.0)*one_over_r6*one_over_r8 - FMath::ConvertTo<ValueClass,FReal>(6.0)*one_over_r8;
        blockDerivative[0]= coef * diffx;
        blockDerivative[1]= coef * diffy;
        blockDerivative[2]= coef * diffz;

    }

    FReal getScaleFactor(const FReal, const int) const
    {
        // return 1 because non homogeneous kernel functions cannot be scaled!!!
        return FReal(1.0);
    }

    FReal getScaleFactor(const FReal) const
    {
        // return 1 because non homogeneous kernel functions cannot be scaled!!!
        return FReal(1.0);
    }



    FReal evaluate(const FPoint<FReal>& pt, const FPoint<FReal>& ps) const{
        return evaluate<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ());
    }
    void evaluateBlock(const FPoint<FReal>& pt, const FPoint<FReal>& ps, FReal* block) const{
        evaluateBlock<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block);
    }
    void evaluateBlockAndDerivative(const FPoint<FReal>& pt, const FPoint<FReal>& ps,
                                    FReal block[1], FReal blockDerivative[3]) const {
        evaluateBlockAndDerivative<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block, blockDerivative);
    }
};


/// One over (a+r^2)
template <class FReal>
struct FInterpMatrixKernelAPLUSRR : FInterpAbstractMatrixKernel<FReal>
{
    static const KERNEL_FUNCTION_TYPE Type = NON_HOMOGENEOUS;
    static const unsigned int NCMP = 1; //< number of components
    static const unsigned int NPV  = 1; //< dim of physical values
    static const unsigned int NPOT = 1; //< dim of potentials
    static const unsigned int NRHS = 1; //< dim of mult exp
    static const unsigned int NLHS = 1; //< dim of loc exp

    const FReal CoreWidth;

    FInterpMatrixKernelAPLUSRR(const FReal inCoreWidth = .25)
    : CoreWidth(inCoreWidth)
    {}

    // copy ctor
    FInterpMatrixKernelAPLUSRR(const FInterpMatrixKernelAPLUSRR& other)
    : CoreWidth(other.CoreWidth)
    {}

    static const char* getID() { return "ONE_OVER_A_PLUS_RR"; }

    static void printInfo() { std::cout << "K(x,y)=1/r with r=|x-y|" << std::endl; }

    // returns position in reduced storage
    int getPosition(const unsigned int) const
    {return 0;}

    // returns coefficient of mutual interaction
    // 1 for symmetric kernels
    // -1 for antisymmetric kernels
    // something else if other property of symmetry
    FReal getMutualCoefficient() const{ return FReal(1.); }

    // evaluate interaction
    template <class ValueClass>
    ValueClass evaluate(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt,
                        const ValueClass& xs, const ValueClass& ys, const ValueClass& zs) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        const ValueClass r2 = (diffx*diffx+diffy*diffy+diffz*diffz);
        return FMath::One<ValueClass>() / (r2 + FMath::ConvertTo<ValueClass,FReal>(CoreWidth));
    }

    // evaluate interaction (blockwise)
    template <class ValueClass>
    void evaluateBlock(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt, 
                       const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                       ValueClass* block) const
    {
        block[0]=this->evaluate(xt,yt,zt,xs,ys,zs);
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& xt, const ValueClass& yt, const ValueClass& zt,
                                    const ValueClass& xs, const ValueClass& ys, const ValueClass& zs,
                                    ValueClass block[1], ValueClass blockDerivative[3]) const
    {
        const ValueClass diffx = (xt-xs);
        const ValueClass diffy = (yt-ys);
        const ValueClass diffz = (zt-zs);
        const ValueClass r2 = (diffx*diffx+diffy*diffy+diffz*diffz);
        const ValueClass one_over_a_plus_r2 = FMath::One<ValueClass>() / (r2 + FMath::ConvertTo<ValueClass,FReal>(CoreWidth));
        const ValueClass one_over_a_plus_r2_squared = one_over_a_plus_r2*one_over_a_plus_r2;

        block[0] = one_over_a_plus_r2;

        // TODO Fix derivative
        const ValueClass coef = FMath::ConvertTo<ValueClass,FReal>(-2.) * one_over_a_plus_r2_squared;
        blockDerivative[0] = coef * diffx;
        blockDerivative[1] = coef * diffy;
        blockDerivative[2] = coef * diffz;

    }

    FReal getScaleFactor(const FReal, const int) const
    {
        // return 1 because non homogeneous kernel functions cannot be scaled!!!
        return FReal(1.0);
    }

    FReal getScaleFactor(const FReal) const
    {
        // return 1 because non homogeneous kernel functions cannot be scaled!!!
        return FReal(1.0);    
    }

    FReal evaluate(const FPoint<FReal>& pt, const FPoint<FReal>& ps) const{
        return evaluate<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ());
    }
    void evaluateBlock(const FPoint<FReal>& pt, const FPoint<FReal>& ps, FReal* block) const{
        evaluateBlock<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block);
    }
    void evaluateBlockAndDerivative(const FPoint<FReal>& pt, const FPoint<FReal>& ps,
                                    FReal block[1], FReal blockDerivative[3]) const {
        evaluateBlockAndDerivative<FReal>(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ(), block, blockDerivative);
    }
};



/*!  Functor which provides the interface to assemble a matrix based on the
  number of rows and cols and on the coordinates s and t and the type of the
  generating matrix-kernel function.
*/
template <class FReal, typename MatrixKernelClass>
class EntryComputer
{
    const MatrixKernelClass *const MatrixKernel;

    const unsigned int nt, ns;
    const FPoint<FReal> *const pt, *const ps;

    const FReal *const weights;

public:
    explicit EntryComputer(const MatrixKernelClass *const inMatrixKernel,
                           const unsigned int _nt, const FPoint<FReal> *const _ps,
                           const unsigned int _ns, const FPoint<FReal> *const _pt,
                           const FReal *const _weights = NULL)
        : MatrixKernel(inMatrixKernel),	nt(_nt), ns(_ns), pt(_pt), ps(_ps), weights(_weights) {}

    void operator()(const unsigned int tbeg, const unsigned int tend,
                    const unsigned int sbeg, const unsigned int send,
                    FReal *const data) const
    {
        unsigned int idx = 0;
        if (weights) {
            for (unsigned int j=tbeg; j<tend; ++j)
                for (unsigned int i=sbeg; i<send; ++i)
                    data[idx++] = weights[i] * weights[j] * MatrixKernel->evaluate(pt[i], ps[j]);
        } else {
            for (unsigned int j=tbeg; j<tend; ++j)
                for (unsigned int i=sbeg; i<send; ++i)
                    data[idx++] = MatrixKernel->evaluate(pt[i], ps[j]);
        }

        /*
    // apply weighting matrices
    if (weights) {
    if ((tend-tbeg) == (send-sbeg) && (tend-tbeg) == nt)
    for (unsigned int n=0; n<nt; ++n) {
    FBlas::scal(nt, weights[n], data + n,  nt); // scale rows
    FBlas::scal(nt, weights[n], data + n * nt); // scale cols
    }
    else if ((tend-tbeg) == 1 && (send-sbeg) == ns)
    for (unsigned int j=0; j<ns; ++j)	data[j] *= weights[j];
    else if ((send-sbeg) == 1 && (tend-tbeg) == nt)
    for (unsigned int i=0; i<nt; ++i)	data[i] *= weights[i];
    }
    */

    }
};





#endif // FINTERPMATRIXKERNEL_HPP

// [--END--]
