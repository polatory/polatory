// See LICENCE file at project root
#ifndef FP2P_TENSORIALKIJ_HPP
#define FP2P_TENSORIALKIJ_HPP

namespace FP2P {

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tensorial Matrix Kernels: K_IJ / p_i=\sum_j K_{ij} w_j
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
   * @brief MutualParticlesKIJ
   * @param sourceX
   * @param sourceY
   * @param sourceZ
   * @param sourcePhysicalValue
   * @param sourceForceX
   * @param sourceForceY
   * @param sourceForceZ
   * @param sourcePotential
   * @param targetX
   * @param targetY
   * @param targetZ
   * @param targetPhysicalValue
   * @param targetForceX
   * @param targetForceY
   * @param targetForceZ
   * @param targetPotential
   */
template<class FReal, typename MatrixKernelClass>
inline void MutualParticlesKIJ(const FReal targetX,const FReal targetY,const FReal targetZ, const FReal* targetPhysicalValue,
                               FReal* targetForceX, FReal* targetForceY, FReal* targetForceZ, FReal* targetPotential,
                               const FReal sourceX,const FReal sourceY,const FReal sourceZ, const FReal* sourcePhysicalValue,
                               FReal* sourceForceX, FReal* sourceForceY, FReal* sourceForceZ, FReal* sourcePotential,
                               const MatrixKernelClass *const MatrixKernel){

    // get information on tensorial aspect of matrix kernel
    const int ncmp = MatrixKernelClass::NCMP;    
    const int npv = MatrixKernelClass::NPV;    
    const int npot = MatrixKernelClass::NPOT;    

    // evaluate kernel and its partial derivatives
    const FPoint<FReal> sourcePoint(sourceX,sourceY,sourceZ);
    const FPoint<FReal> targetPoint(targetX,targetY,targetZ);
    FReal Kxy[ncmp];
    FReal dKxy[ncmp][3];
    MatrixKernel->evaluateBlockAndDerivative(targetPoint,sourcePoint,Kxy,dKxy);
    const FReal mutual_coeff = MatrixKernel->getMutualCoefficient(); // 1 if symmetric; -1 if antisymmetric

    for(unsigned int i = 0 ; i < npot ; ++i){
        for(unsigned int j = 0 ; j < npv ; ++j){

            // update component to be applied
            const int d = MatrixKernel->getPosition(i*npv+j);

            // forces prefactor
            const FReal coef = (targetPhysicalValue[j] * sourcePhysicalValue[j]);

            targetForceX[i] += dKxy[d][0] * coef;
            targetForceY[i] += dKxy[d][1] * coef;
            targetForceZ[i] += dKxy[d][2] * coef;
            targetPotential[i] += ( Kxy[d] * sourcePhysicalValue[j] );

            sourceForceX[i] -= dKxy[d][0] * coef;
            sourceForceY[i] -= dKxy[d][1] * coef;
            sourceForceZ[i] -= dKxy[d][2] * coef;
            sourcePotential[i] += ( mutual_coeff * Kxy[d] * targetPhysicalValue[j] );

        }// j
    }// i

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tensorial Matrix Kernels: K_IJ
// TODO: Implement SSE and AVX variants then move following FullMutualKIJ and FullRemoteKIJ to FP2P.hpp
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief FullMutualKIJ
 */
template <class FReal, class ContainerClass, typename MatrixKernelClass>
inline void FullMutualKIJ(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                          const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){

    // get information on tensorial aspect of matrix kernel
    const int ncmp = MatrixKernelClass::NCMP;
    const int npv = MatrixKernelClass::NPV;    
    const int npot = MatrixKernelClass::NPOT;

    // get info on targets
    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];

    for(FSize idxNeighbors = 0 ; idxNeighbors < limiteNeighbors ; ++idxNeighbors){
        for(FSize idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
            if( inNeighbors[idxNeighbors] ){
                const FSize nbParticlesSources = inNeighbors[idxNeighbors]->getNbParticles();
                const FReal*const sourcesX = inNeighbors[idxNeighbors]->getPositions()[0];
                const FReal*const sourcesY = inNeighbors[idxNeighbors]->getPositions()[1];
                const FReal*const sourcesZ = inNeighbors[idxNeighbors]->getPositions()[2];

                for(FSize idxSource = 0 ; idxSource < nbParticlesSources ; ++idxSource){

                    // evaluate kernel and its partial derivatives
                    const FPoint<FReal> sourcePoint(sourcesX[idxSource],sourcesY[idxSource],sourcesZ[idxSource]);
                    const FPoint<FReal> targetPoint(targetsX[idxTarget],targetsY[idxTarget],targetsZ[idxTarget]);
                    FReal Kxy[ncmp];
                    FReal dKxy[ncmp][3];
                    MatrixKernel->evaluateBlockAndDerivative(targetPoint,sourcePoint,Kxy,dKxy);
                    const FReal mutual_coeff = MatrixKernel->getMutualCoefficient(); // 1 if symmetric; -1 if antisymmetric

                    for(unsigned int i = 0 ; i < npot ; ++i){
                        FReal*const targetsPotentials = inTargets->getPotentials(i);
                        FReal*const targetsForcesX = inTargets->getForcesX(i);
                        FReal*const targetsForcesY = inTargets->getForcesY(i);
                        FReal*const targetsForcesZ = inTargets->getForcesZ(i);
                        FReal*const sourcesPotentials = inNeighbors[idxNeighbors]->getPotentials(i);
                        FReal*const sourcesForcesX = inNeighbors[idxNeighbors]->getForcesX(i);
                        FReal*const sourcesForcesY = inNeighbors[idxNeighbors]->getForcesY(i);
                        FReal*const sourcesForcesZ = inNeighbors[idxNeighbors]->getForcesZ(i);

                        for(unsigned int j = 0 ; j < npv ; ++j){
                            const FReal*const targetsPhysicalValues = inTargets->getPhysicalValues(j);
                            const FReal*const sourcesPhysicalValues = inNeighbors[idxNeighbors]->getPhysicalValues(j);

                            // update component to be applied
                            const int d = MatrixKernel->getPosition(i*npv+j);

                            // forces prefactor
                            FReal coef = (targetsPhysicalValues[idxTarget] * sourcesPhysicalValues[idxSource]);

                            targetsForcesX[idxTarget] += dKxy[d][0] * coef;
                            targetsForcesY[idxTarget] += dKxy[d][1] * coef;
                            targetsForcesZ[idxTarget] += dKxy[d][2] * coef;
                            targetsPotentials[idxTarget] += ( Kxy[d] * sourcesPhysicalValues[idxSource] );

                            sourcesForcesX[idxSource] -= dKxy[d][0] * coef;
                            sourcesForcesY[idxSource] -= dKxy[d][1] * coef;
                            sourcesForcesZ[idxSource] -= dKxy[d][2] * coef;
                            sourcesPotentials[idxSource] += mutual_coeff * Kxy[d] * targetsPhysicalValues[idxTarget];

                        }// j
                    }// i
                }
            }
        }
    }
}

template <class FReal, class ContainerClass, typename MatrixKernelClass>
inline void InnerKIJ(ContainerClass* const FRestrict inTargets, const MatrixKernelClass *const MatrixKernel){

    // get information on tensorial aspect of matrix kernel
    const int ncmp = MatrixKernelClass::NCMP;
    const int npv = MatrixKernelClass::NPV;    
    const int npot = MatrixKernelClass::NPOT;

    // get info on targets
    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];

    for(FSize idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
        for(FSize idxSource = idxTarget + 1 ; idxSource < nbParticlesTargets ; ++idxSource){

            // evaluate kernel and its partial derivatives
            const FPoint<FReal> sourcePoint(targetsX[idxSource],targetsY[idxSource],targetsZ[idxSource]);
            const FPoint<FReal> targetPoint(targetsX[idxTarget],targetsY[idxTarget],targetsZ[idxTarget]);
            FReal Kxy[ncmp];
            FReal dKxy[ncmp][3];
            MatrixKernel->evaluateBlockAndDerivative(targetPoint,sourcePoint,Kxy,dKxy);
            const FReal mutual_coeff = MatrixKernel->getMutualCoefficient(); // 1 if symmetric; -1 if antisymmetric

            for(unsigned int i = 0 ; i < npot ; ++i){
                FReal*const targetsPotentials = inTargets->getPotentials(i);
                FReal*const targetsForcesX = inTargets->getForcesX(i);
                FReal*const targetsForcesY = inTargets->getForcesY(i);
                FReal*const targetsForcesZ = inTargets->getForcesZ(i);

                for(unsigned int j = 0 ; j < npv ; ++j){
                    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValues(j);

                    // update component to be applied
                    const int d = MatrixKernel->getPosition(i*npv+j);

                    // forces prefactor
                    const FReal coef = (targetsPhysicalValues[idxTarget] * targetsPhysicalValues[idxSource]);

                    targetsForcesX[idxTarget] += dKxy[d][0] * coef;
                    targetsForcesY[idxTarget] += dKxy[d][1] * coef;
                    targetsForcesZ[idxTarget] += dKxy[d][2] * coef;
                    targetsPotentials[idxTarget] += ( Kxy[d] * targetsPhysicalValues[idxSource] );

                    targetsForcesX[idxSource] -= dKxy[d][0] * coef;
                    targetsForcesY[idxSource] -= dKxy[d][1] * coef;
                    targetsForcesZ[idxSource] -= dKxy[d][2] * coef;
                    targetsPotentials[idxSource] += mutual_coeff * Kxy[d] * targetsPhysicalValues[idxTarget];
                }// j
            }// i

        }
    }
}

/**
   * @brief FullRemoteKIJ
   */
template <class FReal, class ContainerClass, typename MatrixKernelClass>
inline void FullRemoteKIJ(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                          const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){

    // get information on tensorial aspect of matrix kernel
    const int ncmp = MatrixKernelClass::NCMP;
    const int npv = MatrixKernelClass::NPV;    
    const int npot = MatrixKernelClass::NPOT;

    // get info on targets
    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];

    for(FSize idxNeighbors = 0 ; idxNeighbors < limiteNeighbors ; ++idxNeighbors){
        for(FSize idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
            if( inNeighbors[idxNeighbors] ){
                const FSize nbParticlesSources = inNeighbors[idxNeighbors]->getNbParticles();
                const FReal*const sourcesX = inNeighbors[idxNeighbors]->getPositions()[0];
                const FReal*const sourcesY = inNeighbors[idxNeighbors]->getPositions()[1];
                const FReal*const sourcesZ = inNeighbors[idxNeighbors]->getPositions()[2];

                for(FSize idxSource = 0 ; idxSource < nbParticlesSources ; ++idxSource){

                    // evaluate kernel and its partial derivatives
                    const FPoint<FReal> sourcePoint(sourcesX[idxSource],sourcesY[idxSource],sourcesZ[idxSource]);
                    const FPoint<FReal> targetPoint(targetsX[idxTarget],targetsY[idxTarget],targetsZ[idxTarget]);
                    FReal Kxy[ncmp];
                    FReal dKxy[ncmp][3];
                    MatrixKernel->evaluateBlockAndDerivative(targetPoint,sourcePoint,Kxy,dKxy);

                    for(unsigned int i = 0 ; i < npot ; ++i){
                        FReal*const targetsPotentials = inTargets->getPotentials(i);
                        FReal*const targetsForcesX = inTargets->getForcesX(i);
                        FReal*const targetsForcesY = inTargets->getForcesY(i);
                        FReal*const targetsForcesZ = inTargets->getForcesZ(i);

                        for(unsigned int j = 0 ; j < npv ; ++j){
                            const FReal*const targetsPhysicalValues = inTargets->getPhysicalValues(j);
                            const FReal*const sourcesPhysicalValues = inNeighbors[idxNeighbors]->getPhysicalValues(j);

                            // update component to be applied
                            const int d = MatrixKernel->getPosition(i*npv+j);

                            // forces prefactor
                            const FReal coef = (targetsPhysicalValues[idxTarget] * sourcesPhysicalValues[idxSource]);

                            targetsForcesX[idxTarget] += dKxy[d][0] * coef;
                            targetsForcesY[idxTarget] += dKxy[d][1] * coef;
                            targetsForcesZ[idxTarget] += dKxy[d][2] * coef;
                            targetsPotentials[idxTarget] += ( Kxy[d] * sourcesPhysicalValues[idxSource] );

                        }// j
                    }// i

                }
            }
        }
    }
}



} // End namespace


#endif // FP2P_TENSORIALKIJ_HPP
