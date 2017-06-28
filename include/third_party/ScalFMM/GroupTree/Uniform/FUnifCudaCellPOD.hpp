#ifndef FUNIFCUDACELLPOD_HPP
#define FUNIFCUDACELLPOD_HPP

#include "../Core/FBasicCellPOD.hpp"

template <int ORDER> struct CudaTensorTraits
{
    enum {nnodes = ORDER*ORDER*ORDER};
};

template <class FReal>
struct FCudaUnifComplex {
    FReal complex[2];
};

template <class FReal, int ORDER, int NRHS = 1, int NLHS = 1, int NVALS = 1>
struct alignas(FStarPUDefaultAlign::StructAlign) FCudaUnifCellPODPole {
    static const int VectorSize = CudaTensorTraits<ORDER>::nnodes;
    static const int TransformedVectorSize = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1);
    FReal multipole_exp[NRHS * NVALS * VectorSize]; //< Multipole expansion
    FCudaUnifComplex<FReal> transformed_multipole_exp[NRHS * NVALS * TransformedVectorSize];
};

template <class FReal, int ORDER, int NRHS = 1, int NLHS = 1, int NVALS = 1>
struct alignas(FStarPUDefaultAlign::StructAlign) FCudaUnifCellPODLocal {
    static const int VectorSize = CudaTensorTraits<ORDER>::nnodes;
    static const int TransformedVectorSize = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1);
    FCudaUnifComplex<FReal>     transformed_local_exp[NLHS * NVALS * TransformedVectorSize];
    FReal     local_exp[NLHS * NVALS * VectorSize]; //< Local expansion
};

#endif // FUNIFCUDACELLPOD_HPP

