#ifndef FUNIFCUDASHAREDDATA_HPP
#define FUNIFCUDASHAREDDATA_HPP

#include "../Cuda/FCudaGlobal.hpp"
#include "../../Utils/FGlobal.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"
#include "FUnifCudaCellPOD.hpp"

template <class FReal, int ORDER>
struct alignas(FStarPUDefaultAlign::StructAlign) FUnifCudaSharedData {
    enum {
        rc = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1),
        opt_rc = rc/2+1,
        ninteractions = 343,
        sizeFc = opt_rc * ninteractions
    };

    FReal BoxWidth;
    FCudaUnifComplex<FReal> FC[sizeFc];
};

template <class FReal, int ORDER>
void FUnifCudaFillObject(void* cudaKernel, const FUnifCudaSharedData<FReal,ORDER>& hostData);

#endif // FUNIFCUDASHAREDDATA_HPP

