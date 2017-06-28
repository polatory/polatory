#ifndef FSTARPUKERNELCAPACITIES_HPP
#define FSTARPUKERNELCAPACITIES_HPP

#include "FStarPUUtils.hpp"

/** A class used with the starpu system should
  * implement this interface in order to inform the algorithm about what the kernel
  * is doing.
  */
class FStarPUKernelCapacities {
public:
    virtual bool supportP2M(const FStarPUTypes inPu) const = 0;
    virtual bool supportM2M(const FStarPUTypes inPu) const = 0;
    virtual bool supportM2L(const FStarPUTypes inPu) const = 0;
    virtual bool supportM2LExtern(const FStarPUTypes inPu) const = 0;
    virtual bool supportL2L(const FStarPUTypes inPu) const = 0;
    virtual bool supportL2P(const FStarPUTypes inPu) const = 0;
    virtual bool supportP2P(const FStarPUTypes inPu) const = 0;
    virtual bool supportP2PExtern(const FStarPUTypes inPu) const = 0;

    virtual bool supportM2LMpi(const FStarPUTypes inPu) const = 0;
    virtual bool supportP2PMpi(const FStarPUTypes inPu) const = 0;

};

class FStarPUAbstractCapacities : public FStarPUKernelCapacities {
protected:
    virtual bool check(const FStarPUTypes inPu) const = 0;
public:
    bool supportP2M(const FStarPUTypes inPu) const override {
        return check(inPu);
    }
    bool supportM2M(const FStarPUTypes inPu) const override {
        return check(inPu);
    }
    bool supportM2L(const FStarPUTypes inPu) const override {
        return check(inPu);
    }
    bool supportM2LExtern(const FStarPUTypes inPu) const override {
        return check(inPu);
    }
    bool supportL2L(const FStarPUTypes inPu) const override {
        return check(inPu);
    }
    bool supportL2P(const FStarPUTypes inPu) const override {
        return check(inPu);
    }
    bool supportP2P(const FStarPUTypes inPu) const override {
        return check(inPu);
    }
    bool supportP2PExtern(const FStarPUTypes inPu) const override {
        return check(inPu);
    }
    bool supportM2LMpi(const FStarPUTypes inPu) const override {
        return check(inPu);
    }
    bool supportP2PMpi(const FStarPUTypes inPu) const override {
        return check(inPu);
    }
};

/**
 * This is for the kernels that implement all the methods.
 */
template <class BaseClass>
class FStarPUAllYesCapacities : public BaseClass, public FStarPUAbstractCapacities {
    bool check(const FStarPUTypes inPu) const override{
        return true;
    }
public:
    using BaseClass::BaseClass;
};

template <class BaseClass>
class FStarPUAllCpuCapacities : public BaseClass, public FStarPUAbstractCapacities {
    bool check(const FStarPUTypes inPu) const override{
        return inPu == FSTARPU_CPU_IDX;
    }
public:
    using BaseClass::BaseClass;
};

#ifdef SCALFMM_ENABLE_CUDA_KERNEL
template <class BaseClass>
class FStarPUAllCudaCapacities : public BaseClass, public FStarPUAbstractCapacities {
    bool check(const FStarPUTypes inPu) const override{
        return inPu == FSTARPU_CUDA_IDX;
    }
public:
    using BaseClass::BaseClass;
};

template <class BaseClass>
class FStarPUAllCpuCudaCapacities : public BaseClass, public FStarPUAbstractCapacities {
    bool check(const FStarPUTypes inPu) const override{
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
public:
    using BaseClass::BaseClass;
};

template <class BaseClass>
class FStarPUCudaP2PCapacities : public BaseClass, public FStarPUAbstractCapacities {
    bool check(const FStarPUTypes inPu) const override{
        return inPu == FSTARPU_CPU_IDX;
    }
public:
    using BaseClass::BaseClass;

    bool supportP2P(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
    bool supportP2PExtern(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
    bool supportP2PMpi(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
};

template <class BaseClass>
class FStarPUCudaP2PM2LCapacities : public BaseClass, public FStarPUAbstractCapacities {
    bool check(const FStarPUTypes inPu) const override{
        return inPu == FSTARPU_CPU_IDX;
    }
public:
    using BaseClass::BaseClass;

    bool supportP2P(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
    bool supportP2PExtern(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
    bool supportP2PMpi(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }

    bool supportM2L(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
    bool supportM2LExtern(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
    bool supportM2LMpi(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
};

template <class BaseClass>
class FStarPUCudaM2LCapacities : public BaseClass, public FStarPUAbstractCapacities {
    bool check(const FStarPUTypes inPu) const override{
        return inPu == FSTARPU_CPU_IDX;
    }
public:
    using BaseClass::BaseClass;

    bool supportM2L(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
    bool supportM2LExtern(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
    bool supportM2LMpi(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_CUDA_IDX;
    }
};

#endif

#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
template <class BaseClass>
class FStarPUAllOpenCLCapacities : public BaseClass, public FStarPUAbstractCapacities {
    bool check(const FStarPUTypes inPu) const override{
        return inPu == FSTARPU_OPENCL_IDX;
    }
public:
    using BaseClass::BaseClass;
};

template <class BaseClass>
class FStarPUAllCpuOpenCLCapacities : public BaseClass, public FStarPUAbstractCapacities {
    bool check(const FStarPUTypes inPu) const override {
        return inPu == FSTARPU_CPU_IDX || inPu == FSTARPU_OPENCL_IDX;
    }
public:
    using BaseClass::BaseClass;
};
#endif


#endif // FSTARPUKERNELCAPACITIES_HPP

