// See LICENCE file at project root
#ifndef FP2PMULTIRHS_HPP
#define FP2PMULTIRHS_HPP

namespace FP2P {

  /*
   * FullMutualMultiRhs (generic version)
   */
  template <class FReal, class ContainerClass, typename MatrixKernelClass>
  inline void FullMutualMultiRhs(ContainerClass* const FRestrict inTargets, ContainerClass* const inNeighbors[],
                                 const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){

    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValuesArray();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesXArray();
    FReal*const targetsForcesY = inTargets->getForcesYArray();
    FReal*const targetsForcesZ = inTargets->getForcesZArray();
    FReal*const targetsPotentials = inTargets->getPotentialsArray();
    const int NVALS = inTargets->getNVALS();
    const FSize targetsLD  = inTargets->getLeadingDimension();

    for(FSize idxNeighbors = 0 ; idxNeighbors < limiteNeighbors ; ++idxNeighbors){
        for(FSize idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
            if( inNeighbors[idxNeighbors] ){
                const FSize nbParticlesSources = inNeighbors[idxNeighbors]->getNbParticles();
                const FReal*const sourcesPhysicalValues = inNeighbors[idxNeighbors]->getPhysicalValuesArray();
                const FReal*const sourcesX = inNeighbors[idxNeighbors]->getPositions()[0];
                const FReal*const sourcesY = inNeighbors[idxNeighbors]->getPositions()[1];
                const FReal*const sourcesZ = inNeighbors[idxNeighbors]->getPositions()[2];
                FReal*const sourcesForcesX = inNeighbors[idxNeighbors]->getForcesXArray();
                FReal*const sourcesForcesY = inNeighbors[idxNeighbors]->getForcesYArray();
                FReal*const sourcesForcesZ = inNeighbors[idxNeighbors]->getForcesZArray();
                FReal*const sourcesPotentials = inNeighbors[idxNeighbors]->getPotentialsArray();
                const FSize sourcesLD  = inNeighbors[idxNeighbors]->getLeadingDimension();

                for(FSize idxSource = 0 ; idxSource < nbParticlesSources ; ++idxSource){

                    // Compute kernel of interaction and its derivative
                    const FPoint<FReal> sourcePoint(sourcesX[idxSource],sourcesY[idxSource],sourcesZ[idxSource]);
                    const FPoint<FReal> targetPoint(targetsX[idxTarget],targetsY[idxTarget],targetsZ[idxTarget]);
                    FReal Kxy[1];
                    FReal dKxy[3];
                    MatrixKernel->evaluateBlockAndDerivative(targetPoint.getX(),targetPoint.getY(),targetPoint.getZ(),
                                                             sourcePoint.getX(),sourcePoint.getY(),sourcePoint.getZ(),
                                                             Kxy,dKxy);
                    const FReal mutual_coeff = MatrixKernel->getMutualCoefficient(); // 1 if symmetric; -1 if antisymmetric

                    for(int idxVals = 0 ; idxVals < NVALS ; ++idxVals){
                      
                        const FSize idxTargetValue = idxVals*targetsLD+idxTarget;
                        const FSize idxSourceValue = idxVals*sourcesLD+idxSource;
                        
                        FReal coef = (targetsPhysicalValues[idxTargetValue] * sourcesPhysicalValues[idxSourceValue]);
                        
                        targetsForcesX[idxTargetValue] += dKxy[0] * coef;
                        targetsForcesY[idxTargetValue] += dKxy[1] * coef;
                        targetsForcesZ[idxTargetValue] += dKxy[2] * coef;
                        targetsPotentials[idxTargetValue] += ( Kxy[0] * sourcesPhysicalValues[idxSourceValue] );
                        
                        sourcesForcesX[idxSourceValue] -= dKxy[0] * coef;
                        sourcesForcesY[idxSourceValue] -= dKxy[1] * coef;
                        sourcesForcesZ[idxSourceValue] -= dKxy[2] * coef;
                        sourcesPotentials[idxSourceValue] += ( mutual_coeff * Kxy[0] * targetsPhysicalValues[idxTargetValue] );

                    } // NVALS

                }
            }
        }
    }
  }

  template <class FReal, class ContainerClass, typename MatrixKernelClass>
  inline void InnerMultiRhs(ContainerClass* const FRestrict inTargets, const MatrixKernelClass *const MatrixKernel){

    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValuesArray();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesXArray();
    FReal*const targetsForcesY = inTargets->getForcesYArray();
    FReal*const targetsForcesZ = inTargets->getForcesZArray();
    FReal*const targetsPotentials = inTargets->getPotentialsArray();
    const int NVALS = inTargets->getNVALS();
    const FSize targetsLD  = inTargets->getLeadingDimension();

    for(FSize idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
        for(FSize idxSource = idxTarget + 1 ; idxSource < nbParticlesTargets ; ++idxSource){

            // Compute kernel of interaction...
            const FPoint<FReal> sourcePoint(targetsX[idxSource],targetsY[idxSource],targetsZ[idxSource]);
            const FPoint<FReal> targetPoint(targetsX[idxTarget],targetsY[idxTarget],targetsZ[idxTarget]);
            FReal Kxy[1];
            FReal dKxy[3];
            MatrixKernel->evaluateBlockAndDerivative(targetPoint.getX(),targetPoint.getY(),targetPoint.getZ(),
                                                     sourcePoint.getX(),sourcePoint.getY(),sourcePoint.getZ(),
                                                     Kxy,dKxy);
            const FReal mutual_coeff = MatrixKernel->getMutualCoefficient(); // 1 if symmetric; -1 if antisymmetric

            for(int idxVals = 0 ; idxVals < NVALS ; ++idxVals){
                      
                const FSize idxTargetValue = idxVals*targetsLD+idxTarget;
                const FSize idxSourceValue = idxVals*targetsLD+idxSource;
                
                FReal coef = (targetsPhysicalValues[idxTargetValue] * targetsPhysicalValues[idxSourceValue]);
                
                targetsForcesX[idxTargetValue] += dKxy[0] * coef;
                targetsForcesY[idxTargetValue] += dKxy[1] * coef;
                targetsForcesZ[idxTargetValue] += dKxy[2] * coef;
                targetsPotentials[idxTargetValue] += ( Kxy[0] * targetsPhysicalValues[idxSourceValue] );
                
                targetsForcesX[idxSourceValue] -= dKxy[0] * coef;
                targetsForcesY[idxSourceValue] -= dKxy[1] * coef;
                targetsForcesZ[idxSourceValue] -= dKxy[2] * coef;
                targetsPotentials[idxSourceValue] += ( mutual_coeff * Kxy[0] * targetsPhysicalValues[idxTargetValue] );

            }// NVALS

        }
    }
}


/**
   * FullRemoteMultiRhs (generic version)
   */
template <class FReal, class ContainerClass, typename MatrixKernelClass>
inline void FullRemoteMultiRhs(ContainerClass* const FRestrict inTargets, const ContainerClass* const inNeighbors[],
                       const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){

    const FSize nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValuesArray();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesXArray();
    FReal*const targetsForcesY = inTargets->getForcesYArray();
    FReal*const targetsForcesZ = inTargets->getForcesZArray();
    FReal*const targetsPotentials = inTargets->getPotentialsArray();
    const int NVALS = inTargets->getNVALS();
    const FSize targetsLD  = inTargets->getLeadingDimension();

    for(FSize idxNeighbors = 0 ; idxNeighbors < limiteNeighbors ; ++idxNeighbors){
        for(FSize idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
            if( inNeighbors[idxNeighbors] ){
                const FSize nbParticlesSources = inNeighbors[idxNeighbors]->getNbParticles();
                const FReal*const sourcesPhysicalValues = inNeighbors[idxNeighbors]->getPhysicalValuesArray();
                const FReal*const sourcesX = inNeighbors[idxNeighbors]->getPositions()[0];
                const FReal*const sourcesY = inNeighbors[idxNeighbors]->getPositions()[1];
                const FReal*const sourcesZ = inNeighbors[idxNeighbors]->getPositions()[2];
                const FSize sourcesLD  = inNeighbors[idxNeighbors]->getLeadingDimension();

                for(FSize idxSource = 0 ; idxSource < nbParticlesSources ; ++idxSource){

                    // Compute kernel of interaction...
                    const FPoint<FReal> sourcePoint(sourcesX[idxSource],sourcesY[idxSource],sourcesZ[idxSource]);
                    const FPoint<FReal> targetPoint(targetsX[idxTarget],targetsY[idxTarget],targetsZ[idxTarget]);
                    FReal Kxy[1];
                    FReal dKxy[3];
                    MatrixKernel->evaluateBlockAndDerivative(targetPoint.getX(),targetPoint.getY(),targetPoint.getZ(),
                                                             sourcePoint.getX(),sourcePoint.getY(),sourcePoint.getZ(),
                                                             Kxy,dKxy);

                    for(int idxVals = 0 ; idxVals < NVALS ; ++idxVals){

                        const FSize idxTargetValue = idxVals*targetsLD+idxTarget;
                        const FSize idxSourceValue = idxVals*sourcesLD+idxSource;
                        
                        FReal coef = (targetsPhysicalValues[idxTargetValue] * sourcesPhysicalValues[idxSourceValue]);
                        
                        targetsForcesX[idxTargetValue] += dKxy[0] * coef;
                        targetsForcesY[idxTargetValue] += dKxy[1] * coef;
                        targetsForcesZ[idxTargetValue] += dKxy[2] * coef;
                        targetsPotentials[idxTargetValue] += ( Kxy[0] * sourcesPhysicalValues[idxSourceValue] );

                    } // NVALS

                }
            }
        }
    }
}

}

#endif // FP2PMULTIRHS_HPP
