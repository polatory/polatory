#ifndef FSTARPUFMMPRIORITIESV2_HPP
#define FSTARPUFMMPRIORITIESV2_HPP


#include "../../Utils/FGlobal.hpp"
#include "FStarPUUtils.hpp"

#include "FStarPUKernelCapacities.hpp"

/**
 * @brief The FStarPUFmmPrioritiesV2 class
 * This class should have an static method to be called by hetero getPrio.
 */
#ifdef STARPU_SUPPORT_SCHEDULER

#include "FStarPUHeteoprio.hpp"

class FStarPUFmmPrioritiesV2{
    int insertionPositionP2M;
    int insertionPositionM2M;

    int insertionPositionP2MSend;
    int insertionPositionM2MSend;

    int insertionPositionM2L;
    int insertionPositionM2LExtern;
    int insertionPositionL2L;
    int insertionPositionL2P;
    int insertionPositionP2P;
    int insertionPositionP2PExtern;

    int treeHeight;

    FStarPUKernelCapacities* capacities;

    static FStarPUFmmPrioritiesV2 controller;


    FStarPUFmmPrioritiesV2(){
    }

public:
    static FStarPUFmmPrioritiesV2& Controller(){
        return controller;
    }

    static void InitSchedulerCallback(unsigned sched_ctx_id, void* heteroprio){
        Controller().initSchedulerCallback(sched_ctx_id, (struct _starpu_heteroprio_center_policy_heteroprio*)heteroprio);
    }

    void init(struct starpu_conf* conf, const int inTreeHeight,
              FStarPUKernelCapacities* inCapacities){
        capacities = inCapacities;

        conf->sched_policy = &_starpu_sched_heteroprio_policy;
        starpu_heteroprio_set_callback(&InitSchedulerCallback);

        treeHeight  = inTreeHeight;

        if(inTreeHeight > 2){
            int incPrio = 0;

            FLOG( FLog::Controller << "Buckets:\n" );

            insertionPositionP2MSend = incPrio++;
            FLOG( FLog::Controller << "\t P2M Send "  << insertionPositionP2MSend << "\n" );
            insertionPositionP2M     = incPrio++;
            FLOG( FLog::Controller << "\t P2M "  << insertionPositionP2M << "\n" );

            insertionPositionM2MSend = incPrio++;
            FLOG( FLog::Controller << "\t M2M Send "  << insertionPositionM2MSend << "\n" );
            insertionPositionM2M     = incPrio++;
            FLOG( FLog::Controller << "\t M2M "  << insertionPositionM2M << "\n" );

            insertionPositionP2P       = incPrio++;
            FLOG( FLog::Controller << "\t P2P "  << insertionPositionP2P << "\n" );

            insertionPositionP2PExtern      = incPrio++;
            FLOG( FLog::Controller << "\t P2P Ex "  << insertionPositionP2PExtern << "\n" );

            insertionPositionL2L     = incPrio++;
            FLOG( FLog::Controller << "\t L2L "  << insertionPositionL2L << "\n" );

            insertionPositionM2LExtern     = incPrio++;
            insertionPositionM2L     = incPrio++;

            insertionPositionL2P     = incPrio++;
            FLOG( FLog::Controller << "\t L2P "  << insertionPositionL2P << "\n" );

        }
        else{
            int incPrio = 0;

            insertionPositionP2MSend = -1;
            insertionPositionP2M     = -1;

            insertionPositionM2MSend = -1;
            insertionPositionM2M     = -1;

            insertionPositionM2LExtern     = -1;
            insertionPositionM2L     = -1;

            insertionPositionL2L     = -1;

            insertionPositionP2P     = incPrio++;
            insertionPositionP2PExtern     = incPrio++;

            insertionPositionL2P     = -1;
            assert(incPrio == 2);
        }
    }

    void initSchedulerCallback(unsigned /*sched_ctx_id*/,
                               struct _starpu_heteroprio_center_policy_heteroprio *heteroprio){
        const bool workOnlyOnLeaves = (treeHeight <= 2);
#ifdef STARPU_USE_CPU
        // CPU follows the real prio
        {
            bool insertInnerP2PLater = false;
#ifdef STARPU_USE_CUDA
           insertInnerP2PLater = capacities->supportP2P(FSTARPU_CUDA_IDX);
#endif
#ifdef STARPU_USE_OPENCL
           insertInnerP2PLater = capacities->supportP2P(FSTARPU_OPENCL_IDX);
#endif

            int cpuCountPrio = 0;

            if( !workOnlyOnLeaves && capacities->supportP2M(FSTARPU_CPU_IDX)){
                FLOG( FLog::Controller << "\t CPU prio P2M Send "  << cpuCountPrio << " bucket " << insertionPositionP2MSend << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionP2MSend;
                heteroprio->buckets[insertionPositionP2MSend].valide_archs |= STARPU_CPU;

                FLOG( FLog::Controller << "\t CPU prio P2M "  << cpuCountPrio << " bucket " << insertionPositionP2M << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionP2M;
                heteroprio->buckets[insertionPositionP2M].valide_archs |= STARPU_CPU;
            }
            if(!workOnlyOnLeaves && capacities->supportM2M(FSTARPU_CPU_IDX)){
                FLOG( FLog::Controller << "\t CPU prio M2M Send "  << cpuCountPrio << " bucket " << insertionPositionM2MSend << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionM2MSend;
                heteroprio->buckets[insertionPositionM2MSend].valide_archs |= STARPU_CPU;

                FLOG( FLog::Controller << "\t CPU prio M2M "  << cpuCountPrio << " bucket " << insertionPositionM2M << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionM2M;
                heteroprio->buckets[insertionPositionM2M].valide_archs |= STARPU_CPU;
            }
            if( capacities->supportP2P(FSTARPU_CPU_IDX) && !insertInnerP2PLater){
                FLOG( FLog::Controller << "\t CPU prio P2P "  << cpuCountPrio << " bucket " << insertionPositionP2P << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionP2P;
                heteroprio->buckets[insertionPositionP2P].valide_archs |= STARPU_CPU;
            }

            if( !workOnlyOnLeaves && capacities->supportL2L(FSTARPU_CPU_IDX)){
                FLOG( FLog::Controller << "\t CPU prio L2L "  << cpuCountPrio << " bucket " << insertionPositionL2L << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionL2L;
                heteroprio->buckets[insertionPositionL2L].valide_archs |= STARPU_CPU;
            }

            if(!workOnlyOnLeaves &&  capacities->supportM2L(FSTARPU_CPU_IDX)){
                FLOG( FLog::Controller << "\t CPU prio M2L "  << cpuCountPrio << " bucket " << insertionPositionM2LExtern << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionM2LExtern;
                heteroprio->buckets[insertionPositionM2LExtern].valide_archs |= STARPU_CPU;
            }
            if(!workOnlyOnLeaves &&  capacities->supportM2L(FSTARPU_CPU_IDX)){
                FLOG( FLog::Controller << "\t CPU prio M2L "  << cpuCountPrio << " bucket " << insertionPositionM2L << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionM2L;
                heteroprio->buckets[insertionPositionM2L].valide_archs |= STARPU_CPU;
            }

            if( !workOnlyOnLeaves && capacities->supportL2P(FSTARPU_CPU_IDX)){
                FLOG( FLog::Controller << "\t CPU prio L2P "  << cpuCountPrio << " bucket " << insertionPositionL2P << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionL2P;
                heteroprio->buckets[insertionPositionL2P].valide_archs |= STARPU_CPU;
            }

            if( capacities->supportP2PExtern(FSTARPU_CPU_IDX) ){
                FLOG( FLog::Controller << "\t CPU prio P2P Ex "  << cpuCountPrio << " bucket " << insertionPositionP2PExtern << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionP2PExtern;
                heteroprio->buckets[insertionPositionP2PExtern].valide_archs |= STARPU_CPU;
            }
            if( capacities->supportP2P(FSTARPU_CPU_IDX) && insertInnerP2PLater){
                FLOG( FLog::Controller << "\t CPU prio P2P "  << cpuCountPrio << " bucket " << insertionPositionP2P << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CPU_IDX][cpuCountPrio++] = insertionPositionP2P;
                heteroprio->buckets[insertionPositionP2P].valide_archs |= STARPU_CPU;
            }

            heteroprio->nb_prio_per_arch_index[FSTARPU_CPU_IDX] = unsigned(cpuCountPrio);
            FLOG( FLog::Controller << "\t CPU Priorities: "  << cpuCountPrio << "\n" );
        }
#endif
#ifdef STARPU_USE_CUDA
        {
            int cudaCountPrio = 0;

            if(capacities->supportP2P(FSTARPU_CUDA_IDX)){
                FLOG( FLog::Controller << "\t CUDA prio P2P "  << cudaCountPrio << " bucket " << insertionPositionP2P << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CUDA_IDX][cudaCountPrio++] = insertionPositionP2P;
                heteroprio->buckets[insertionPositionP2P].valide_archs |= STARPU_CUDA;
                heteroprio->buckets[insertionPositionP2P].factor_base_arch_index = FSTARPU_CUDA_IDX;
#ifdef STARPU_USE_CPU
                if(capacities->supportP2P(FSTARPU_CPU_IDX)){
                    heteroprio->buckets[insertionPositionP2P].slow_factors_per_index[FSTARPU_CPU_IDX] = 4.0f;
                }
#endif
            }

            if(!workOnlyOnLeaves && capacities->supportM2L(FSTARPU_CUDA_IDX)){
                FLOG( FLog::Controller << "\t CUDA prio M2L "  << cudaCountPrio << " bucket " << insertionPositionM2L << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CUDA_IDX][cudaCountPrio++] = insertionPositionM2L;
                heteroprio->buckets[insertionPositionM2L].valide_archs |= STARPU_CUDA;
                FAssertLF(capacities->supportM2LExtern(FSTARPU_CUDA_IDX));
            }
            if(capacities->supportP2PExtern(FSTARPU_CUDA_IDX)){
                FLOG( FLog::Controller << "\t CUDA prio P2P "  << cudaCountPrio << " bucket " << insertionPositionP2PExtern << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CUDA_IDX][cudaCountPrio++] = insertionPositionP2PExtern;
                heteroprio->buckets[insertionPositionP2PExtern].valide_archs |= STARPU_CUDA;
            }

            if( !workOnlyOnLeaves && capacities->supportP2M(FSTARPU_CUDA_IDX)){
                FLOG( FLog::Controller << "\t CUDA prio P2M send "  << cudaCountPrio << " bucket " << insertionPositionP2MSend << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CUDA_IDX][cudaCountPrio++] = insertionPositionP2MSend;
                heteroprio->buckets[insertionPositionP2MSend].valide_archs |= STARPU_CUDA;

                FLOG( FLog::Controller << "\t CUDA prio P2M "  << cudaCountPrio << " bucket " << insertionPositionP2M << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CUDA_IDX][cudaCountPrio++] = insertionPositionP2M;
                heteroprio->buckets[insertionPositionP2M].valide_archs |= STARPU_CUDA;
            }

            if( !workOnlyOnLeaves && capacities->supportM2M(FSTARPU_CUDA_IDX)){
                FLOG( FLog::Controller << "\t CUDA prio M2M send "  << cudaCountPrio << " bucket " << insertionPositionM2MSend << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CUDA_IDX][cudaCountPrio++] = insertionPositionM2MSend;
                heteroprio->buckets[insertionPositionM2MSend].valide_archs |= STARPU_CUDA;

                FLOG( FLog::Controller << "\t CUDA prio M2M "  << cudaCountPrio << " bucket " << insertionPositionM2M << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CUDA_IDX][cudaCountPrio++] = insertionPositionM2M;
                heteroprio->buckets[insertionPositionM2M].valide_archs |= STARPU_CUDA;
            }

            if( !workOnlyOnLeaves && capacities->supportL2L(FSTARPU_CUDA_IDX)){
                FLOG( FLog::Controller << "\t CUDA prio L2L "  << cudaCountPrio << " bucket " << insertionPositionL2L << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CUDA_IDX][cudaCountPrio++] = insertionPositionL2L;
                heteroprio->buckets[insertionPositionL2L].valide_archs |= STARPU_CUDA;
            }

            if( !workOnlyOnLeaves && capacities->supportL2P(FSTARPU_CUDA_IDX)){
                FLOG( FLog::Controller << "\t CUDA prio L2P "  << cudaCountPrio << " bucket " << insertionPositionL2P << "\n" );
                heteroprio->prio_mapping_per_arch_index[FSTARPU_CUDA_IDX][cudaCountPrio++] = insertionPositionL2P;
                heteroprio->buckets[insertionPositionL2P].valide_archs |= STARPU_CUDA;
            }

            heteroprio->nb_prio_per_arch_index[FSTARPU_CUDA_IDX] = int(cudaCountPrio);
            FLOG( FLog::Controller << "\t CUDA Priorities: "  << cudaCountPrio << "\n" );
        }
#endif
    }

    int getInsertionPosP2M() const {
        return insertionPositionP2M;
    }
    int getInsertionPosM2M(const int /*inLevel*/) const {
        return insertionPositionM2M;
    }
    int getInsertionPosP2M(bool willBeSend) const {
        return willBeSend?insertionPositionP2MSend:insertionPositionP2M;
    }
    int getInsertionPosM2M(const int /*inLevel*/, bool willBeSend) const {
        return willBeSend?insertionPositionM2MSend:insertionPositionM2M;
    }
    int getInsertionPosM2L(const int /*inLevel*/) const {
        return insertionPositionM2L;
    }
    int getInsertionPosM2LExtern(const int /*inLevel*/) const {
        return insertionPositionM2LExtern;
    }
    int getInsertionPosL2L(const int inLevel) const {
        return insertionPositionL2L;
    }
    int getInsertionPosL2P() const {
        return insertionPositionL2P;
    }
    int getInsertionPosP2P() const {
        return insertionPositionP2P;
    }
    int getInsertionPosP2PExtern() const {
        return insertionPositionP2PExtern;
    }
};

FStarPUFmmPrioritiesV2 FStarPUFmmPrioritiesV2::controller;

#elif defined(SCALFMM_STARPU_USE_PRIO)// STARPU_SUPPORT_SCHEDULER

#include "FOmpPriorities.hpp"

class FStarPUFmmPrioritiesV2 {
    static FStarPUFmmPrioritiesV2 controller;
    FOmpPriorities ompPrio;

    FStarPUFmmPrioritiesV2() : ompPrio(0){
    }

public:
    static FStarPUFmmPrioritiesV2& Controller(){
        return controller;
    }


    void init(struct starpu_conf* /*conf*/, const int inTreeHeight,
              FStarPUKernelCapacities* /*inCapacities*/){
        ompPrio = FOmpPriorities(inTreeHeight);
    }

    int getInsertionPosP2M() const {
        return ompPrio.getInsertionPosP2M();
    }
    int getInsertionPosM2M(const int inLevel) const {
        return ompPrio.getInsertionPosM2M(inLevel);
    }
    int getInsertionPosP2M(bool /*willBeSend*/) const {
        return ompPrio.getInsertionPosP2M();
    }
    int getInsertionPosM2M(const int inLevel, bool /*willBeSend*/) const {
        return ompPrio.getInsertionPosM2M(inLevel);
    }
    int getInsertionPosM2L(const int inLevel) const {
        return ompPrio.getInsertionPosM2L(inLevel);
    }
    int getInsertionPosM2LExtern(const int inLevel) const {
        return ompPrio.getInsertionPosM2LExtern(inLevel);
    }
    int getInsertionPosL2L(const int inLevel) const {
        return ompPrio.getInsertionPosL2L(inLevel);
    }
    int getInsertionPosL2P() const {
        return ompPrio.getInsertionPosL2P();
    }
    int getInsertionPosP2P() const {
        return ompPrio.getInsertionPosP2P();
    }
    int getInsertionPosP2PExtern() const {
        return ompPrio.getInsertionPosP2PExtern();
    }
};

FStarPUFmmPrioritiesV2 FStarPUFmmPrioritiesV2::controller;

#endif // SCALFMM_STARPU_USE_PRIO - STARPU_SUPPORT_SCHEDULER

#endif // FSTARPUFMMPRIORITIESV2_HPP

