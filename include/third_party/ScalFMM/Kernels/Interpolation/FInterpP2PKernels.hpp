#ifndef FINTERPP2PKERNELS_HPP
#define FINTERPP2PKERNELS_HPP


#include "../P2P/FP2P.hpp"
#include "../P2P/FP2PR.hpp"

///////////////////////////////////////////////////////
// P2P Wrappers
///////////////////////////////////////////////////////

template <class FReal, int NCMP, int NVALS>
struct DirectInteractionComputer
{
  template <typename ContainerClass, typename MatrixKernelClass>
  static void P2P( ContainerClass* const FRestrict TargetParticles,
                   ContainerClass* const NeighborSourceParticles[],
                   const int inSize,
                   const MatrixKernelClass *const MatrixKernel){
      FP2P::FullMutualKIJ<FReal, ContainerClass, MatrixKernelClass>(TargetParticles,NeighborSourceParticles,inSize,MatrixKernel);
  }

  template <typename ContainerClass, typename MatrixKernelClass>
  static void P2PInner( ContainerClass* const FRestrict TargetParticles,
                   const MatrixKernelClass *const MatrixKernel){
      FP2P::InnerKIJ<FReal, ContainerClass, MatrixKernelClass>(TargetParticles,MatrixKernel);
  }

  template <typename ContainerClass, typename MatrixKernelClass>
  static void P2PRemote( ContainerClass* const FRestrict inTargets,
                         const ContainerClass* const inNeighbors[],
                         const int inSize,
                         const MatrixKernelClass *const MatrixKernel){
      FP2P::FullRemoteKIJ<FReal, ContainerClass, MatrixKernelClass>(inTargets,inNeighbors,inSize,MatrixKernel);
  }
};


/*! Specialization for scalar kernels */
template <class FReal, int NVALS>
struct DirectInteractionComputer<FReal, 1,NVALS>
{
  template <typename ContainerClass, typename MatrixKernelClass>
  static void P2P( ContainerClass* const FRestrict TargetParticles,
                   ContainerClass* const NeighborSourceParticles[],
                   const int inSize,
                   const MatrixKernelClass *const MatrixKernel){
      FP2P::FullMutualMultiRhs<FReal, ContainerClass, MatrixKernelClass>(TargetParticles,NeighborSourceParticles,inSize,MatrixKernel);
  }

  template <typename ContainerClass, typename MatrixKernelClass>
  static void P2PInner( ContainerClass* const FRestrict TargetParticles,
                   const MatrixKernelClass *const MatrixKernel){
      FP2P::InnerMultiRhs<FReal, ContainerClass, MatrixKernelClass>(TargetParticles,MatrixKernel);
  }

  template <typename ContainerClass, typename MatrixKernelClass>
  static void P2PRemote( ContainerClass* const FRestrict inTargets,
                         const ContainerClass* const inNeighbors[],
                         const int inSize,
                         const MatrixKernelClass *const MatrixKernel){
      FP2P::FullRemoteMultiRhs<FReal, ContainerClass, MatrixKernelClass>(inTargets,inNeighbors,inSize,MatrixKernel);
  }
};

/*! Specialization for scalar kernels and single rhs*/
template <class FReal>
struct DirectInteractionComputer<FReal, 1,1>
{
  template <typename ContainerClass, typename MatrixKernelClass>
  static void P2P( ContainerClass* const FRestrict TargetParticles,
                   ContainerClass* const NeighborSourceParticles[],
                   const int inSize,
                   const MatrixKernelClass *const MatrixKernel){
      FP2PT<FReal>::template FullMutual<ContainerClass,MatrixKernelClass> (TargetParticles,NeighborSourceParticles,inSize,MatrixKernel);
  }

  template <typename ContainerClass, typename MatrixKernelClass>
  static void P2PInner( ContainerClass* const FRestrict TargetParticles,
                   const MatrixKernelClass *const MatrixKernel){
      FP2PT<FReal>::template Inner<ContainerClass, MatrixKernelClass>(TargetParticles,MatrixKernel);
  }

  template <typename ContainerClass, typename MatrixKernelClass>
  static void P2PRemote( ContainerClass* const FRestrict inTargets,
                         const ContainerClass* const inNeighbors[],
                         const int inSize,
                         const MatrixKernelClass *const MatrixKernel){
      FP2PT<FReal>::template FullRemote<ContainerClass,MatrixKernelClass>(inTargets,inNeighbors,inSize,MatrixKernel);
  }
};

#endif // FINTERPP2PKERNELS_HPP
