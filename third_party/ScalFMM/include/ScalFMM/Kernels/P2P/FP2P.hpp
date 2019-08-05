// See LICENCE file at project root
#ifndef FP2P_HPP
#define FP2P_HPP

namespace FP2P {

/**
   * @brief MutualParticles (generic version)
   * P2P mutual interaction,
   * this function computes the interaction for 2 particles.
   *
   * Formulas are:
   * \f[
   * F = - q_1 * q_2 * grad K{12}
   * P_1 = q_2 * K{12} ; P_2 = q_1 * K_{12}
   * \f]
   * In details for \f$K(x,y)=1/|x-y|=1/r\f$ :
   * \f$ P_1 = \frac{ q_2 }{ r } \f$
   * \f$ P_2 = \frac{ q_1 }{ r } \f$
   * \f$ F(x) = \frac{ \Delta_x * q_1 * q_2 }{ r^2 } \f$
   *
   * @param sourceX
   * @param sourceY
   * @param sourceZ
   * @param sourcePhysicalValue
   * @param targetX
   * @param targetY
   * @param targetZ
   * @param targetPhysicalValue
   * @param targetForceX
   * @param targetForceY
   * @param targetForceZ
   * @param targetPotential
   * @param MatrixKernel pointer to an interaction kernel evaluator
   */
template <class FReal, typename MatrixKernelClass>
inline void MutualParticles(const FReal targetX,const FReal targetY,const FReal targetZ, const FReal targetPhysicalValue,
                            FReal* targetForceX, FReal* targetForceY, FReal* targetForceZ, FReal* targetPotential,
                            const FReal sourceX,const FReal sourceY,const FReal sourceZ, const FReal sourcePhysicalValue,
                            FReal* sourceForceX, FReal* sourceForceY, FReal* sourceForceZ, FReal* sourcePotential,
                            const MatrixKernelClass *const MatrixKernel){

    // Compute kernel of interaction...
    const FPoint<FReal> sourcePoint(sourceX,sourceY,sourceZ);
    const FPoint<FReal> targetPoint(targetX,targetY,targetZ);
    FReal Kxy[1];
    FReal dKxy[3];
    MatrixKernel->evaluateBlockAndDerivative(targetPoint,sourcePoint,Kxy,dKxy);
    const FReal mutual_coeff = MatrixKernel->getMutualCoefficient(); // 1 if symmetric; -1 if antisymmetric

    FReal coef = (targetPhysicalValue * sourcePhysicalValue);

    (*targetForceX) += dKxy[0] * coef;
    (*targetForceY) += dKxy[1] * coef;
    (*targetForceZ) += dKxy[2] * coef;
    (*targetPotential) += ( Kxy[0] * sourcePhysicalValue );

    (*sourceForceX) -= dKxy[0] * coef;
    (*sourceForceY) -= dKxy[1] * coef;
    (*sourceForceZ) -= dKxy[2] * coef;
    (*sourcePotential) += ( mutual_coeff * Kxy[0] * targetPhysicalValue );
}

/**
   * @brief NonMutualParticles (generic version)
   * P2P mutual interaction,
   * this function computes the interaction for 2 particles.
   *
   * Formulas are:
   * \f[
   * F = - q_1 * q_2 * grad K{12}
   * P_1 = q_2 * K{12} ; P_2 = q_1 * K_{12}
   * \f]
   * In details for \f$K(x,y)=1/|x-y|=1/r\f$ :
   * \f$ P_1 = \frac{ q_2 }{ r } \f$
   * \f$ P_2 = \frac{ q_1 }{ r } \f$
   * \f$ F(x) = \frac{ \Delta_x * q_1 * q_2 }{ r^2 } \f$
   */
template <class FReal, typename MatrixKernelClass>
inline void NonMutualParticles(const FReal targetX,const FReal targetY,const FReal targetZ, const FReal targetPhysicalValue,
                               FReal* targetForceX, FReal* targetForceY, FReal* targetForceZ, FReal* targetPotential,
                               const FReal sourceX,const FReal sourceY,const FReal sourceZ, const FReal sourcePhysicalValue,
                               const MatrixKernelClass *const MatrixKernel){

    // Compute kernel of interaction...
    const FPoint<FReal> sourcePoint(sourceX,sourceY,sourceZ);
    const FPoint<FReal> targetPoint(targetX,targetY,targetZ);
    FReal Kxy[1];
    FReal dKxy[3];
    MatrixKernel->evaluateBlockAndDerivative(targetPoint,sourcePoint,Kxy,dKxy);

    FReal coef = (targetPhysicalValue * sourcePhysicalValue);

    (*targetForceX) += dKxy[0] * coef;
    (*targetForceY) += dKxy[1] * coef;
    (*targetForceZ) += dKxy[2] * coef;
    (*targetPotential) += ( Kxy[0] * sourcePhysicalValue );
}




template <class FReal, class ContainerClass, class MatrixKernelClass, class ComputeClass, int NbFRealInComputeClass>
static void GenericFullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                              const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){

    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValues();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesX();
    FReal*const targetsForcesY = inTargets->getForcesY();
    FReal*const targetsForcesZ = inTargets->getForcesZ();
    FReal*const targetsPotentials = inTargets->getPotentials();

    for(FSize idxNeighbors = 0 ; idxNeighbors < limiteNeighbors ; ++idxNeighbors){
        if( inNeighbors[idxNeighbors] ){
            const FSize nbParticlesSources = (inNeighbors[idxNeighbors]->getNbParticles()+NbFRealInComputeClass-1)/NbFRealInComputeClass;
            const ComputeClass*const sourcesPhysicalValues = (const ComputeClass*)inNeighbors[idxNeighbors]->getPhysicalValues();
            const ComputeClass*const sourcesX = (const ComputeClass*)inNeighbors[idxNeighbors]->getPositions()[0];
            const ComputeClass*const sourcesY = (const ComputeClass*)inNeighbors[idxNeighbors]->getPositions()[1];
            const ComputeClass*const sourcesZ = (const ComputeClass*)inNeighbors[idxNeighbors]->getPositions()[2];
            ComputeClass*const sourcesForcesX = (ComputeClass*)inNeighbors[idxNeighbors]->getForcesX();
            ComputeClass*const sourcesForcesY = (ComputeClass*)inNeighbors[idxNeighbors]->getForcesY();
            ComputeClass*const sourcesForcesZ = (ComputeClass*)inNeighbors[idxNeighbors]->getForcesZ();
            ComputeClass*const sourcesPotentials = (ComputeClass*)inNeighbors[idxNeighbors]->getPotentials();

            for(FSize idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
                const ComputeClass tx = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsX[idxTarget]);
                const ComputeClass ty = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsY[idxTarget]);
                const ComputeClass tz = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsZ[idxTarget]);
                const ComputeClass tv = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsPhysicalValues[idxTarget]);
                ComputeClass  tfx = FMath::Zero<ComputeClass>();
                ComputeClass  tfy = FMath::Zero<ComputeClass>();
                ComputeClass  tfz = FMath::Zero<ComputeClass>();
                ComputeClass  tpo = FMath::Zero<ComputeClass>();

                for(FSize idxSource = 0 ; idxSource < nbParticlesSources ; ++idxSource){
                    ComputeClass Kxy[1];
                    ComputeClass dKxy[3];
                    MatrixKernel->evaluateBlockAndDerivative(tx,ty,tz,
                                                             sourcesX[idxSource],sourcesY[idxSource],sourcesZ[idxSource],
                                                             Kxy,dKxy);
                    const ComputeClass mutual_coeff = FMath::ConvertTo<ComputeClass, FReal>(MatrixKernel->getMutualCoefficient());; // 1 if symmetric; -1 if antisymmetric

                    const ComputeClass coef = (tv * sourcesPhysicalValues[idxSource]);

                    dKxy[0] *= coef;
                    dKxy[1] *= coef;
                    dKxy[2] *= coef;

                    tfx += dKxy[0];
                    tfy += dKxy[1];
                    tfz += dKxy[2];
                    tpo += Kxy[0] * sourcesPhysicalValues[idxSource];

                    sourcesForcesX[idxSource] -= dKxy[0];
                    sourcesForcesY[idxSource] -= dKxy[1];
                    sourcesForcesZ[idxSource] -= dKxy[2];
                    sourcesPotentials[idxSource] += mutual_coeff * Kxy[0] * tv;
                }

                targetsForcesX[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfx);
                targetsForcesY[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfy);
                targetsForcesZ[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfz);
                targetsPotentials[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tpo);
            }
        }
    }
}

template <class FReal, class ContainerClass, class MatrixKernelClass, class ComputeClass, int NbFRealInComputeClass>
static void GenericInner(ContainerClass* const FRestrict inTargets, const MatrixKernelClass *const MatrixKernel){

    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValues();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesX();
    FReal*const targetsForcesY = inTargets->getForcesY();
    FReal*const targetsForcesZ = inTargets->getForcesZ();
    FReal*const targetsPotentials = inTargets->getPotentials();

    {//In this part, we compute (vectorially) the interaction
        //within the target leaf.

        const FSize nbParticlesSources = (nbParticlesTargets+NbFRealInComputeClass-1)/NbFRealInComputeClass;
        const ComputeClass*const sourcesPhysicalValues = (const ComputeClass*)targetsPhysicalValues;
        const ComputeClass*const sourcesX = (const ComputeClass*)targetsX;
        const ComputeClass*const sourcesY = (const ComputeClass*)targetsY;
        const ComputeClass*const sourcesZ = (const ComputeClass*)targetsZ;
        ComputeClass*const sourcesForcesX = (ComputeClass*)targetsForcesX;
        ComputeClass*const sourcesForcesY = (ComputeClass*)targetsForcesY;
        ComputeClass*const sourcesForcesZ = (ComputeClass*)targetsForcesZ;
        ComputeClass*const sourcesPotentials = (ComputeClass*)targetsPotentials;

        for(FSize idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
            const ComputeClass tx = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsX[idxTarget]);
            const ComputeClass ty = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsY[idxTarget]);
            const ComputeClass tz = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsZ[idxTarget]);
            const ComputeClass tv = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsPhysicalValues[idxTarget]);
            ComputeClass  tfx = FMath::Zero<ComputeClass>();
            ComputeClass  tfy = FMath::Zero<ComputeClass>();
            ComputeClass  tfz = FMath::Zero<ComputeClass>();
            ComputeClass  tpo = FMath::Zero<ComputeClass>();

            for(FSize idxSource = (idxTarget+NbFRealInComputeClass)/NbFRealInComputeClass ; idxSource < nbParticlesSources ; ++idxSource){
                ComputeClass Kxy[1];
                ComputeClass dKxy[3];
                MatrixKernel->evaluateBlockAndDerivative(tx,ty,tz,
                                                         sourcesX[idxSource],sourcesY[idxSource],sourcesZ[idxSource],
                                                         Kxy,dKxy);
                const ComputeClass mutual_coeff = FMath::ConvertTo<ComputeClass, FReal>(MatrixKernel->getMutualCoefficient()); // 1 if symmetric; -1 if antisymmetric

                const ComputeClass coef = (tv * sourcesPhysicalValues[idxSource]);

                dKxy[0] *= coef;
                dKxy[1] *= coef;
                dKxy[2] *= coef;

                tfx += dKxy[0];
                tfy += dKxy[1];
                tfz += dKxy[2];
        		tpo = FMath::FMAdd(Kxy[0],sourcesPhysicalValues[idxSource],tpo);

                sourcesForcesX[idxSource] -= dKxy[0];
                sourcesForcesY[idxSource] -= dKxy[1];
                sourcesForcesZ[idxSource] -= dKxy[2];
        		sourcesPotentials[idxSource] = FMath::FMAdd(mutual_coeff * Kxy[0],tv,sourcesPotentials[idxSource]);
            }

            targetsForcesX[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfx);
            targetsForcesY[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfy);
            targetsForcesZ[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfz);
            targetsPotentials[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tpo);
        }
    }

    for(FSize idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
        const FSize limitForTarget = NbFRealInComputeClass-(idxTarget%NbFRealInComputeClass);
        for(FSize idxS = 1 ; idxS < limitForTarget ; ++idxS){
            const FSize idxSource = idxTarget + idxS;
            FReal Kxy[1];
            FReal dKxy[3];
            MatrixKernel->evaluateBlockAndDerivative(targetsX[idxTarget],targetsY[idxTarget],targetsZ[idxTarget],
                                                     targetsX[idxSource],targetsY[idxSource],targetsZ[idxSource],
                                                     Kxy,dKxy);
            const FReal mutual_coeff = MatrixKernel->getMutualCoefficient(); // 1 if symmetric; -1 if antisymmetric

            const FReal coef = (targetsPhysicalValues[idxTarget] * targetsPhysicalValues[idxSource]);

            dKxy[0] *= coef;
            dKxy[1] *= coef;
            dKxy[2] *= coef;

            targetsForcesX[idxTarget] += dKxy[0];
            targetsForcesY[idxTarget] += dKxy[1];
            targetsForcesZ[idxTarget] += dKxy[2];
            targetsPotentials[idxTarget] += Kxy[0] * targetsPhysicalValues[idxSource];

            targetsForcesX[idxSource] -= dKxy[0];
            targetsForcesY[idxSource] -= dKxy[1];
            targetsForcesZ[idxSource] -= dKxy[2];
            targetsPotentials[idxSource] += mutual_coeff * Kxy[0] * targetsPhysicalValues[idxTarget];
        }
    }
}

template <class FReal, class ContainerClass, class MatrixKernelClass, class ComputeClass, int NbFRealInComputeClass>
static void GenericFullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                              const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){

    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValues();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesX();
    FReal*const targetsForcesY = inTargets->getForcesY();
    FReal*const targetsForcesZ = inTargets->getForcesZ();
    FReal*const targetsPotentials = inTargets->getPotentials();

    for(FSize idxNeighbors = 0 ; idxNeighbors < limiteNeighbors ; ++idxNeighbors){
        if( inNeighbors[idxNeighbors] ){
            const FSize nbParticlesSources = (inNeighbors[idxNeighbors]->getNbParticles()+NbFRealInComputeClass-1)/NbFRealInComputeClass;
            const ComputeClass*const sourcesPhysicalValues = (const ComputeClass*)inNeighbors[idxNeighbors]->getPhysicalValues();
            const ComputeClass*const sourcesX = (const ComputeClass*)inNeighbors[idxNeighbors]->getPositions()[0];
            const ComputeClass*const sourcesY = (const ComputeClass*)inNeighbors[idxNeighbors]->getPositions()[1];
            const ComputeClass*const sourcesZ = (const ComputeClass*)inNeighbors[idxNeighbors]->getPositions()[2];

            for(FSize idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
                const ComputeClass tx = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsX[idxTarget]);
                const ComputeClass ty = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsY[idxTarget]);
                const ComputeClass tz = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsZ[idxTarget]);
                const ComputeClass tv = FMath::ConvertTo<ComputeClass, const FReal*>(&targetsPhysicalValues[idxTarget]);
                ComputeClass  tfx = FMath::Zero<ComputeClass>();
                ComputeClass  tfy = FMath::Zero<ComputeClass>();
                ComputeClass  tfz = FMath::Zero<ComputeClass>();
                ComputeClass  tpo = FMath::Zero<ComputeClass>();

                for(FSize idxSource = 0 ; idxSource < nbParticlesSources ; ++idxSource){
                    ComputeClass Kxy[1];
                    ComputeClass dKxy[3];
                    MatrixKernel->evaluateBlockAndDerivative(tx,ty,tz,
                                                             sourcesX[idxSource],sourcesY[idxSource],sourcesZ[idxSource],
                                                             Kxy,dKxy);
                    const ComputeClass coef = (tv * sourcesPhysicalValues[idxSource]);

                    dKxy[0] *= coef;
                    dKxy[1] *= coef;
                    dKxy[2] *= coef;

                    tfx += dKxy[0];
                    tfy += dKxy[1];
                    tfz += dKxy[2];
                    tpo += Kxy[0] * sourcesPhysicalValues[idxSource];
                }

                targetsForcesX[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfx);
                targetsForcesY[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfy);
                targetsForcesZ[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfz);
                targetsPotentials[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tpo);
            }
        }
    }
}

} // End namespace

template <class FReal>
struct FP2PT{
};

#if defined(SCALFMM_USE_AVX)
template <>
struct FP2PT<double>{
    template <class ContainerClass, class MatrixKernelClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullMutual<double, ContainerClass, MatrixKernelClass, __m256d, 4>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }


    template <class ContainerClass, class MatrixKernelClass>
    static void Inner(ContainerClass* const FRestrict inTargets, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericInner<double, ContainerClass, MatrixKernelClass, __m256d, 4>(inTargets, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullRemote<double, ContainerClass, MatrixKernelClass, __m256d, 4>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }
};

template <>
struct FP2PT<float>{
    template <class ContainerClass, class MatrixKernelClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullMutual<float, ContainerClass, MatrixKernelClass, __m256, 8>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void Inner(ContainerClass* const FRestrict inTargets, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericInner<float, ContainerClass, MatrixKernelClass, __m256, 8>(inTargets, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullRemote<float, ContainerClass, MatrixKernelClass, __m256, 8>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }
};
#elif defined(SCALFMM_USE_AVX2)
template <>
struct FP2PT<double>{
    template <class ContainerClass, class MatrixKernelClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullMutual<double, ContainerClass, MatrixKernelClass, __m512d, 8>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void Inner(ContainerClass* const FRestrict inTargets, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericInner<double, ContainerClass, MatrixKernelClass, __m512d, 8>(inTargets, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullRemote<double, ContainerClass, MatrixKernelClass, __m512d, 8>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }
};

template <>
struct FP2PT<float>{
    template <class ContainerClass, class MatrixKernelClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullMutual<float, ContainerClass, MatrixKernelClass, __m512, 16>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }


    template <class ContainerClass, class MatrixKernelClass>
    static void Inner(ContainerClass* const FRestrict inTargets, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericInner<float, ContainerClass, MatrixKernelClass, __m512, 16>(inTargets, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullRemote<float, ContainerClass, MatrixKernelClass, __m512, 16>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }
};
#elif defined(SCALFMM_USE_SSE)
template <>
struct FP2PT<double>{
    template <class ContainerClass, class MatrixKernelClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullMutual<double, ContainerClass, MatrixKernelClass, __m128d, 2>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }


    template <class ContainerClass, class MatrixKernelClass>
    static void Inner(ContainerClass* const FRestrict inTargets, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericInner<double, ContainerClass, MatrixKernelClass, __m128d, 2>(inTargets, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullRemote<double, ContainerClass, MatrixKernelClass, __m128d, 2>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }
};

template <>
struct FP2PT<float>{
    template <class ContainerClass, class MatrixKernelClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullMutual<float, ContainerClass, MatrixKernelClass, __m128, 4>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void Inner(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericInner<float, ContainerClass, MatrixKernelClass, __m128, 4>(inTargets, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullRemote<float, ContainerClass, MatrixKernelClass, __m128, 4>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }
};
#else
template <>
struct FP2PT<double>{
    template <class ContainerClass, class MatrixKernelClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullMutual<double, ContainerClass, MatrixKernelClass, double, 1>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void Inner(ContainerClass* const FRestrict inTargets, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericInner<double, ContainerClass, MatrixKernelClass, double, 1>(inTargets, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullRemote<double, ContainerClass, MatrixKernelClass, double, 1>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }
};

template <>
struct FP2PT<float>{
    template <class ContainerClass, class MatrixKernelClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullMutual<float, ContainerClass, MatrixKernelClass, float, 1>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void Inner(ContainerClass* const FRestrict inTargets, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericInner<float, ContainerClass, MatrixKernelClass, float, 1>(inTargets, MatrixKernel);
    }

    template <class ContainerClass, class MatrixKernelClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                           const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
        FP2P::GenericFullRemote<float, ContainerClass, MatrixKernelClass, float, 1>(inTargets, inNeighbors, limiteNeighbors, MatrixKernel);
    }
};
#endif

#include "FP2PTensorialKij.hpp"

#include "FP2PMultiRhs.hpp"

#endif // FP2P_HPP
