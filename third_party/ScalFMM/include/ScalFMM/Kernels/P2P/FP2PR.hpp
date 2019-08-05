// See LICENCE file at project root

#ifndef FP2PR_HPP
#define FP2PR_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Utils/FMath.hpp"


/**
 * @brief The FP2PR namespace
 */
namespace FP2PR{
template <class FReal>
inline void MutualParticles(const FReal targetX,const FReal targetY,const FReal targetZ, const FReal targetPhysicalValue,
                            FReal* targetForceX, FReal* targetForceY, FReal* targetForceZ, FReal* targetPotential,
                            const FReal sourceX,const FReal sourceY,const FReal sourceZ, const FReal sourcePhysicalValue,
                            FReal* sourceForceX, FReal* sourceForceY, FReal* sourceForceZ, FReal* sourcePotential){
    FReal dx = targetX - sourceX;
    FReal dy = targetY - sourceY;
    FReal dz = targetZ - sourceZ;

    FReal inv_square_distance = FReal(1.0) / (dx*dx + dy*dy + dz*dz);
    FReal inv_distance = FMath::Sqrt(inv_square_distance);

    inv_square_distance *= inv_distance;
    inv_square_distance *= targetPhysicalValue * sourcePhysicalValue;

    dx *= - inv_square_distance;
    dy *= - inv_square_distance;
    dz *= - inv_square_distance;

    *targetForceX += dx;
    *targetForceY += dy;
    *targetForceZ += dz;
    *targetPotential += ( inv_distance * sourcePhysicalValue );

    *sourceForceX -= dx;
    *sourceForceY -= dy;
    *sourceForceZ -= dz;
    *sourcePotential += ( inv_distance * targetPhysicalValue );
}

template <class FReal>
inline void NonMutualParticles(const FReal targetX,const FReal targetY,const FReal targetZ, const FReal targetPhysicalValue,
                               FReal* targetForceX, FReal* targetForceY, FReal* targetForceZ, FReal* targetPotential,
                               const FReal sourceX,const FReal sourceY,const FReal sourceZ, const FReal sourcePhysicalValue){
    FReal dx = targetX - sourceX;
    FReal dy = targetY - sourceY;
    FReal dz = targetZ - sourceZ;

    FReal inv_square_distance = FReal(1.0) / (dx*dx + dy*dy + dz*dz);
    FReal inv_distance = FMath::Sqrt(inv_square_distance);

    inv_square_distance *= inv_distance;
    inv_square_distance *= targetPhysicalValue * sourcePhysicalValue;

    // d/dx(1/|x-y|)=-(x-y)/r^3
    dx *= - inv_square_distance;
    dy *= - inv_square_distance;
    dz *= - inv_square_distance;

    *targetForceX += dx;
    *targetForceY += dy;
    *targetForceZ += dz;
    *targetPotential += ( inv_distance * sourcePhysicalValue );
}


template <class FReal, class ContainerClass, class ComputeClass, int NbFRealInComputeClass>
static void GenericFullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                              const int limiteNeighbors){

    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValues();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesX();
    FReal*const targetsForcesY = inTargets->getForcesY();
    FReal*const targetsForcesZ = inTargets->getForcesZ();
    FReal*const targetsPotentials = inTargets->getPotentials();

    const ComputeClass mOne = FMath::One<ComputeClass>();

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
                    ComputeClass dx = tx - sourcesX[idxSource];
                    ComputeClass dy = ty - sourcesY[idxSource];
                    ComputeClass dz = tz - sourcesZ[idxSource];

                    ComputeClass inv_square_distance = mOne / (dx*dx + dy*dy + dz*dz);
                    const ComputeClass inv_distance = FMath::Sqrt(inv_square_distance);

                    inv_square_distance *= inv_distance;
                    inv_square_distance *= tv * sourcesPhysicalValues[idxSource];

                    dx *= - inv_square_distance;
                    dy *= - inv_square_distance;
                    dz *= - inv_square_distance;

                    tfx += dx;
                    tfy += dy;
                    tfz += dz;
                    tpo += inv_distance * sourcesPhysicalValues[idxSource];

                    sourcesForcesX[idxSource] -= dx;
                    sourcesForcesY[idxSource] -= dy;
                    sourcesForcesZ[idxSource] -= dz;
                    sourcesPotentials[idxSource] += inv_distance * tv;
                }

                targetsForcesX[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfx);
                targetsForcesY[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfy);
                targetsForcesZ[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tfz);
                targetsPotentials[idxTarget] += FMath::ConvertTo<FReal, ComputeClass>(tpo);
            }
        }
    }
}

template <class FReal, class ContainerClass, class ComputeClass, int NbFRealInComputeClass>
static void GenericInner(ContainerClass* const FRestrict inTargets){

    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValues();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesX();
    FReal*const targetsForcesY = inTargets->getForcesY();
    FReal*const targetsForcesZ = inTargets->getForcesZ();
    FReal*const targetsPotentials = inTargets->getPotentials();

    const ComputeClass mOne = FMath::One<ComputeClass>();

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

                ComputeClass dx = tx - sourcesX[idxSource];
                ComputeClass dy = ty - sourcesY[idxSource];
                ComputeClass dz = tz - sourcesZ[idxSource];
                ComputeClass inv_square_distance = mOne / (dx*dx + dy*dy + dz*dz);
                const ComputeClass inv_distance = FMath::Sqrt(inv_square_distance);

                inv_square_distance *= inv_distance;
                inv_square_distance *= tv * sourcesPhysicalValues[idxSource];

                dx *= - inv_square_distance;
                dy *= - inv_square_distance;
                dz *= - inv_square_distance;

                tfx += dx;
                tfy += dy;
                tfz += dz;
                tpo += inv_distance * sourcesPhysicalValues[idxSource];

                sourcesForcesX[idxSource] -= dx;
                sourcesForcesY[idxSource] -= dy;
                sourcesForcesZ[idxSource] -= dz;
                sourcesPotentials[idxSource] += inv_distance * tv;
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
            FReal dx = targetsX[idxTarget] - targetsX[idxSource];
            FReal dy = targetsY[idxTarget] - targetsY[idxSource];
            FReal dz = targetsZ[idxTarget] - targetsZ[idxSource];

            FReal inv_square_distance = FReal(1.0) / (dx*dx + dy*dy + dz*dz);
            const FReal inv_distance = FMath::Sqrt(inv_square_distance);

            inv_square_distance *= inv_distance;
            inv_square_distance *= targetsPhysicalValues[idxTarget] * targetsPhysicalValues[idxSource];

            dx *= - inv_square_distance;
            dy *= - inv_square_distance;
            dz *= - inv_square_distance;

            targetsForcesX[idxTarget] += dx;
            targetsForcesY[idxTarget] += dy;
            targetsForcesZ[idxTarget] += dz;
            targetsPotentials[idxTarget] += inv_distance * targetsPhysicalValues[idxSource];

            targetsForcesX[idxSource] -= dx;
            targetsForcesY[idxSource] -= dy;
            targetsForcesZ[idxSource] -= dz;
            targetsPotentials[idxSource] += inv_distance * targetsPhysicalValues[idxTarget];
        }
    }
}

template <class FReal, class ContainerClass, class ComputeClass, int NbFRealInComputeClass>
static void GenericFullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                       const int limiteNeighbors){
    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValues();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesX();
    FReal*const targetsForcesY = inTargets->getForcesY();
    FReal*const targetsForcesZ = inTargets->getForcesZ();
    FReal*const targetsPotentials = inTargets->getPotentials();

    const ComputeClass mOne = FMath::One<ComputeClass>();

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
                    ComputeClass dx = tx - sourcesX[idxSource];
                    ComputeClass dy = ty - sourcesY[idxSource];
                    ComputeClass dz = tz - sourcesZ[idxSource];

                    ComputeClass inv_square_distance = mOne / (dx*dx + dy*dy + dz*dz);
                    const ComputeClass inv_distance = FMath::Sqrt(inv_square_distance);

                    inv_square_distance *= inv_distance;
                    inv_square_distance *= tv * sourcesPhysicalValues[idxSource];

                    dx *= - inv_square_distance;
                    dy *= - inv_square_distance;
                    dz *= - inv_square_distance;

                    tfx += dx;
                    tfy += dy;
                    tfz += dz;
                    tpo += inv_distance * sourcesPhysicalValues[idxSource];
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
struct FP2PRT{
};

#if defined(SCALFMM_USE_AVX)

template <>
struct FP2PRT<double>{
    template <class ContainerClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors){
        FP2PR::GenericFullMutual<double, ContainerClass, __m256d, 4>(inTargets, inNeighbors, limiteNeighbors);
    }

    template <class ContainerClass>
    static void Inner(ContainerClass* const FRestrict inTargets){
        FP2PR::GenericInner<double, ContainerClass, __m256d, 4>(inTargets);
    }

    template <class ContainerClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
               const int limiteNeighbors){
        FP2PR::GenericFullRemote<double, ContainerClass, __m256d, 4>(inTargets, inNeighbors, limiteNeighbors);
    }
};

template <>
struct FP2PRT<float>{
    template <class ContainerClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors){
        FP2PR::GenericFullMutual<float, ContainerClass, __m256, 8>(inTargets, inNeighbors, limiteNeighbors);
    }

    template <class ContainerClass>
    static void Inner(ContainerClass* const FRestrict inTargets){
        FP2PR::GenericFullMutual<float, ContainerClass, __m256, 8>(inTargets);
    }

    template <class ContainerClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
               const int limiteNeighbors){
        FP2PR::GenericFullRemote<float, ContainerClass, __m256, 8>(inTargets, inNeighbors, limiteNeighbors);
    }
};
#elif defined(SCALFMM_USE_AVX2)
template <>
struct FP2PRT<double>{
    template <class ContainerClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors){
        FP2PR::GenericFullMutual<double, ContainerClass, __m512d, 8>(inTargets, inNeighbors, limiteNeighbors);
    }

    template <class ContainerClass>
    static void Inner(ContainerClass* const FRestrict inTargets){
        FP2PR::GenericInner<double, ContainerClass, __m512d, 8>(inTargets);
    }

    template <class ContainerClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
               const int limiteNeighbors){
        FP2PR::GenericFullRemote<double, ContainerClass, __m512d, 8>(inTargets, inNeighbors, limiteNeighbors);
    }
};

template <>
struct FP2PRT<float>{
    template <class ContainerClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors){
        FP2PR::GenericFullMutual<float, ContainerClass, __m512, 16>(inTargets, inNeighbors, limiteNeighbors);
    }

    template <class ContainerClass>
    static void Inner(ContainerClass* const FRestrict inTargets){
        FP2PR::GenericFullMutual<float, ContainerClass, __m512, 16>(inTargets);
    }

    template <class ContainerClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
               const int limiteNeighbors){
        FP2PR::GenericFullRemote<float, ContainerClass, __m512, 16>(inTargets, inNeighbors, limiteNeighbors);
    }
};

#elif defined(SCALFMM_USE_SSE)
template <>
struct FP2PRT<double>{
    template <class ContainerClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors){
        FP2PR::GenericFullMutual<double, ContainerClass, __m128d, 2>(inTargets, inNeighbors, limiteNeighbors);
    }

    template <class ContainerClass>
    static void Inner(ContainerClass* const FRestrict inTargets){
        FP2PR::GenericInner<double, ContainerClass, __m128d, 2>(inTargets);
    }

    template <class ContainerClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
               const int limiteNeighbors){
        FP2PR::GenericFullRemote<double, ContainerClass, __m128d, 2>(inTargets, inNeighbors, limiteNeighbors);
    }
};

template <>
struct FP2PRT<float>{
    template <class ContainerClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors){
        FP2PR::GenericFullMutual<float, ContainerClass, __m128, 4>(inTargets, inNeighbors, limiteNeighbors);
    }

    template <class ContainerClass>
    static void Inner(ContainerClass* const FRestrict inTargets){
        FP2PR::GenericInner<float, ContainerClass, __m128, 4>(inTargets);
    }

    template <class ContainerClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
               const int limiteNeighbors){
        FP2PR::GenericFullRemote<float, ContainerClass, __m128, 4>(inTargets, inNeighbors, limiteNeighbors);
    }
};

#else
template <>
struct FP2PRT<double>{
    template <class ContainerClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors){
        FP2PR::GenericFullMutual<double, ContainerClass, double, 1>(inTargets, inNeighbors, limiteNeighbors);
    }

    template <class ContainerClass>
    static void Inner(ContainerClass* const FRestrict inTargets){
        FP2PR::GenericInner<double, ContainerClass, double, 1>(inTargets);
    }

    template <class ContainerClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
               const int limiteNeighbors){
        FP2PR::GenericFullRemote<double, ContainerClass, double, 1>(inTargets, inNeighbors, limiteNeighbors);
    }
};

template <>
struct FP2PRT<float>{
    template <class ContainerClass>
    static void FullMutual(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                           const int limiteNeighbors){
        FP2PR::GenericFullMutual<float, ContainerClass, float, 1>(inTargets, inNeighbors, limiteNeighbors);
    }

    template <class ContainerClass>
    static void Inner(ContainerClass* const FRestrict inTargets){
        FP2PR::GenericInner<float, ContainerClass, float, 1>(inTargets);
    }

    template <class ContainerClass>
    static void FullRemote(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
               const int limiteNeighbors){
        FP2PR::GenericFullRemote<float, ContainerClass, float, 1>(inTargets, inNeighbors, limiteNeighbors);
    }
};
#endif




#endif // FP2PR_HPP
