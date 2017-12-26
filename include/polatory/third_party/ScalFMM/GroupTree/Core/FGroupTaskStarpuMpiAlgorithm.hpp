// Keep in private GIT
#ifndef FGROUPTASKSTARPUMPIALGORITHM_HPP
#define FGROUPTASKSTARPUMPIALGORITHM_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Core/FCoreCommon.hpp"
#include "../../Utils/FQuickSort.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../../Utils/FLog.hpp"
#include "../../Utils/FTic.hpp"
#include "../../Utils/FAssert.hpp"
#include "../../Utils/FAlignedMemory.hpp"
#include "../../Utils/FAssert.hpp"
#include "../../Utils/FEnv.hpp"

#include "../../Utils/FMpi.hpp"

#include "FOutOfBlockInteraction.hpp"

#include <vector>
#include <memory>

#include <omp.h>
#include <unordered_map>
#include <set>
#include <starpu.h>
#include <starpu_mpi.h>
#include "../StarPUUtils/FStarPUUtils.hpp"
#include "../StarPUUtils/FStarPUFmmPriorities.hpp"
#include "../StarPUUtils/FStarPUFmmPrioritiesV2.hpp"
#include "../StarPUUtils/FStarPUReduxCpu.hpp"

#ifdef STARPU_USE_CPU
#include "../StarPUUtils/FStarPUCpuWrapper.hpp"
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
#include "../StarPUUtils/FStarPUCudaWrapper.hpp"
#include "../Cuda/FCudaEmptyKernel.hpp"
#include "../Cuda/FCudaGroupAttachedLeaf.hpp"
#include "../Cuda/FCudaGroupOfParticles.hpp"
#include "../Cuda/FCudaGroupOfCells.hpp"
#include "../Cuda/FCudaEmptyCellSymb.hpp"
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
#include "../StarPUUtils/FStarPUOpenClWrapper.hpp"
#include "../OpenCl/FOpenCLDeviceWrapper.hpp"
#include "../OpenCl/FEmptyOpenCLCode.hpp"
#endif

#include "../StarPUUtils/FStarPUReduxCpu.hpp"

#ifdef SCALFMM_SIMGRID_TASKNAMEPARAMS
#include "../StarPUUtils/FStarPUTaskNameParams.hpp"
#endif

template <class OctreeClass, class CellContainerClass, class KernelClass, class ParticleGroupClass, class StarPUCpuWrapperClass
          #ifdef SCALFMM_ENABLE_CUDA_KERNEL
          , class StarPUCudaWrapperClass = FStarPUCudaWrapper<KernelClass, FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                                              FCudaGroupOfParticles<int, 0, 0, int>, FCudaGroupAttachedLeaf<int, 0, 0, int>, FCudaEmptyKernel<int> >
          #endif
          #ifdef SCALFMM_ENABLE_OPENCL_KERNEL
          , class StarPUOpenClWrapperClass = FStarPUOpenClWrapper<KernelClass, FOpenCLDeviceWrapper<KernelClass>>
          #endif
          >
class FGroupTaskStarPUMpiAlgorithm : public FAbstractAlgorithm {
protected:
    typedef FGroupTaskStarPUMpiAlgorithm<OctreeClass, CellContainerClass, KernelClass, ParticleGroupClass, StarPUCpuWrapperClass
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
    , StarPUCudaWrapperClass
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
    , StarPUOpenClWrapperClass
#endif
    > ThisClass;

    int getTag(const int inLevel, const MortonIndex mindex, const int idxBloc, const int mode, const int otherProc) const{
        int shift = 0, s_mindex = 0;
        int height = tree->getHeight();
        int h_mindex = idxBloc;
        while(height) { shift += 1; height >>= 1; }
        while(h_mindex) { s_mindex += 1; h_mindex >>= 1; }

        FAssertLF((s_mindex + shift + 8) <= 32, "Tag overflow !!");
        const int tag = int(((((idxBloc<<shift) + inLevel) << 3) + mode) << 5);
        int *tag_ub = 0;
        int ok = 0;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &ok);
        FAssertLF(tag < *tag_ub, "Tag overflow: Tag greater than MPI_TAG_UB");
        {
            struct TagInfo{
                int level;
                MortonIndex mindex;
                int idxBloc;
                int mode;
                bool operator==(TagInfo const& a) const
                {
                    return (a.idxBloc == idxBloc && a.level == level && a.mindex == mindex && a.mode == mode);
                }
             };
            static std::unordered_map<int, TagInfo> previousTag;
            const TagInfo currentInfo = {inLevel, mindex, idxBloc, mode};
            auto found = previousTag.find(tag);
            if(found != previousTag.end()){
                const TagInfo prev = found->second;
                assert(currentInfo == prev);
            }
            else{
                previousTag[tag] = currentInfo;
            }
        }
        return tag;

    }

    const FMpi::FComm& comm;

    template <class OtherBlockClass>
    struct BlockInteractions{
        OtherBlockClass* otherBlock;
        int otherBlockId;
        std::vector<OutOfBlockInteraction> interactions;
    };

    struct CellHandles{
        starpu_data_handle_t symb;
        starpu_data_handle_t up;
        starpu_data_handle_t down;
        int intervalSize;
    };

    struct ParticleHandles{
        starpu_data_handle_t symb;
        starpu_data_handle_t down;
        int intervalSize;
    };

    std::vector< std::vector< std::vector<BlockInteractions<CellContainerClass>>>> externalInteractionsAllLevel;
    std::vector< std::vector<BlockInteractions<ParticleGroupClass>>> externalInteractionsLeafLevel;

    OctreeClass*const tree;       //< The Tree
    KernelClass*const originalCpuKernel;

    std::vector<CellHandles>* cellHandles;
    std::vector<ParticleHandles> particleHandles;

    starpu_codelet p2m_cl;
    starpu_codelet m2m_cl;
    starpu_codelet l2l_cl;
    starpu_codelet l2l_cl_nocommute;
    starpu_codelet l2p_cl;

    starpu_codelet m2l_cl_in;
    starpu_codelet m2l_cl_inout;
    starpu_codelet m2l_cl_inout_mpi;

    starpu_codelet p2p_cl_in;
    starpu_codelet p2p_cl_inout;
    starpu_codelet p2p_cl_inout_mpi;


#ifdef STARPU_USE_REDUX
    starpu_codelet p2p_redux_init;
    starpu_codelet p2p_redux_perform;
    starpu_codelet p2p_redux_read;
#endif

    const bool noCommuteAtLastLevel;
    const bool noCommuteBetweenLevel;

#ifdef STARPU_USE_CPU
    StarPUCpuWrapperClass cpuWrapper;
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
    StarPUCudaWrapperClass cudaWrapper;
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
    StarPUOpenClWrapperClass openclWrapper;
#endif

    FStarPUPtrInterface wrappers;
    FStarPUPtrInterface* wrapperptr;

#ifdef STARPU_SUPPORT_ARBITER
    starpu_arbiter_t arbiterGlobal;
#endif

#ifdef STARPU_USE_TASK_NAME
    std::vector<std::unique_ptr<char[]>> m2mTaskNames;
    std::vector<std::unique_ptr<char[]>> m2lTaskNames;
    std::vector<std::unique_ptr<char[]>> m2lOuterTaskNames;
    std::vector<std::unique_ptr<char[]>> l2lTaskNames;
    std::unique_ptr<char[]> p2mTaskNames;
    std::unique_ptr<char[]> l2pTaskNames;
    std::unique_ptr<char[]> p2pTaskNames;
    std::unique_ptr<char[]> p2pOuterTaskNames;
#endif
#ifdef SCALFMM_STARPU_USE_PRIO
    typedef FStarPUFmmPrioritiesV2 PrioClass;// FStarPUFmmPriorities
#endif

public:
    FGroupTaskStarPUMpiAlgorithm(const FMpi::FComm& inComm, OctreeClass*const inTree, KernelClass* inKernels)
        :   comm(inComm), tree(inTree), originalCpuKernel(inKernels),
          cellHandles(nullptr),
          noCommuteAtLastLevel(FEnv::GetBool("SCALFMM_NO_COMMUTE_LAST_L2L", true)),
          noCommuteBetweenLevel(FEnv::GetBool("SCALFMM_NO_COMMUTE_M2L_L2L", false)),
      #ifdef STARPU_USE_CPU
          cpuWrapper(tree->getHeight()),
      #endif
      #ifdef SCALFMM_ENABLE_CUDA_KERNEL
          cudaWrapper(tree->getHeight()),
      #endif
      #ifdef SCALFMM_ENABLE_OPENCL_KERNEL
          openclWrapper(tree->getHeight()),
      #endif
          wrapperptr(&wrappers){
        FAssertLF(tree, "tree cannot be null");
        FAssertLF(inKernels, "kernels cannot be null");

        FAbstractAlgorithm::setNbLevelsInTree(tree->getHeight());

        struct starpu_conf conf;
        FAssertLF(starpu_conf_init(&conf) == 0);
#ifdef SCALFMM_STARPU_USE_PRIO
        PrioClass::Controller().init(&conf, tree->getHeight(), inKernels);
#endif
        FAssertLF(starpu_init(&conf) == 0);
        FAssertLF(starpu_mpi_init ( 0, 0, 0 ) == 0);
		int *tag_ub = 0;
        int ok = 0;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &ok);


        starpu_malloc_set_align(32);

        starpu_pthread_mutex_t initMutex;
        starpu_pthread_mutex_init(&initMutex, NULL);
#ifdef STARPU_USE_CPU
        FStarPUUtils::ExecOnWorkers(STARPU_CPU, [&](){
            starpu_pthread_mutex_lock(&initMutex);
            cpuWrapper.initKernel(starpu_worker_get_id(), inKernels);
            starpu_pthread_mutex_unlock(&initMutex);
        });
        wrappers.set(FSTARPU_CPU_IDX, &cpuWrapper);
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        FStarPUUtils::ExecOnWorkers(STARPU_CUDA, [&](){
            starpu_pthread_mutex_lock(&initMutex);
            cudaWrapper.initKernel(starpu_worker_get_id(), inKernels);
            starpu_pthread_mutex_unlock(&initMutex);
        });
        wrappers.set(FSTARPU_CUDA_IDX, &cudaWrapper);
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        FStarPUUtils::ExecOnWorkers(STARPU_OPENCL, [&](){
            starpu_pthread_mutex_lock(&initMutex);
            openclWrapper.initKernel(starpu_worker_get_id(), inKernels);
            starpu_pthread_mutex_unlock(&initMutex);
        });
        wrappers.set(FSTARPU_OPENCL_IDX, &openclWrapper);
#endif
        starpu_pthread_mutex_destroy(&initMutex);

        starpu_pause();

        cellHandles   = new std::vector<CellHandles>[tree->getHeight()];

#ifdef STARPU_SUPPORT_ARBITER
        arbiterGlobal = starpu_arbiter_create();
#endif
        initCodelet();
        initCodeletMpi();
        rebuildInteractions();

        FLOG(FLog::Controller << "FGroupTaskStarPUAlgorithm (Max Worker " << starpu_worker_get_count() << ")\n");
#ifdef STARPU_USE_CPU
        FLOG(FLog::Controller << "FGroupTaskStarPUAlgorithm (Max CPU " << starpu_cpu_worker_get_count() << ")\n");
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        FLOG(FLog::Controller << "FGroupTaskStarPUAlgorithm (Max OpenCL " << starpu_opencl_worker_get_count() << ")\n");
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        FLOG(FLog::Controller << "FGroupTaskStarPUAlgorithm (Max CUDA " << starpu_cuda_worker_get_count() << ")\n");
#endif
        FLOG(FLog::Controller << "SCALFMM_NO_COMMUTE_LAST_L2L " << noCommuteAtLastLevel << "\n");
        FLOG(FLog::Controller << "SCALFMM_NO_COMMUTE_M2L_L2L " << noCommuteBetweenLevel << "\n");

        buildTaskNames();
    }

    void buildTaskNames(){
#ifdef STARPU_USE_TASK_NAME
        const int namesLength = 128;
        m2mTaskNames.resize(tree->getHeight());
        m2lTaskNames.resize(tree->getHeight());
        m2lOuterTaskNames.resize(tree->getHeight());
        l2lTaskNames.resize(tree->getHeight());
        for(int idxLevel = 0 ; idxLevel < tree->getHeight() ; ++idxLevel){
            m2mTaskNames[idxLevel].reset(new char[namesLength]);
            snprintf(m2mTaskNames[idxLevel].get(), namesLength, "M2M-level-%d", idxLevel);
            m2lTaskNames[idxLevel].reset(new char[namesLength]);
            snprintf(m2lTaskNames[idxLevel].get(), namesLength, "M2L-level-%d", idxLevel);
            m2lOuterTaskNames[idxLevel].reset(new char[namesLength]);
            snprintf(m2lOuterTaskNames[idxLevel].get(), namesLength, "M2L-out-level-%d", idxLevel);
            l2lTaskNames[idxLevel].reset(new char[namesLength]);
            snprintf(l2lTaskNames[idxLevel].get(), namesLength, "L2L-level-%d", idxLevel);
        }

        p2mTaskNames.reset(new char[namesLength]);
        snprintf(p2mTaskNames.get(), namesLength, "P2M");
        l2pTaskNames.reset(new char[namesLength]);
        snprintf(l2pTaskNames.get(), namesLength, "L2P");
        p2pTaskNames.reset(new char[namesLength]);
        snprintf(p2pTaskNames.get(), namesLength, "P2P");
        p2pOuterTaskNames.reset(new char[namesLength]);
        snprintf(p2pOuterTaskNames.get(), namesLength, "P2P-out");
#endif
    }

    void syncData(){
        for(int idxLevel = 0 ; idxLevel < tree->getHeight() ; ++idxLevel){
            for(int idxHandle = 0 ; idxHandle < int(cellHandles[idxLevel].size()) ; ++idxHandle){
                starpu_data_acquire(cellHandles[idxLevel][idxHandle].symb, STARPU_R);
                starpu_data_release(cellHandles[idxLevel][idxHandle].symb);
                starpu_data_acquire(cellHandles[idxLevel][idxHandle].up, STARPU_R);
                starpu_data_release(cellHandles[idxLevel][idxHandle].up);
                starpu_data_acquire(cellHandles[idxLevel][idxHandle].down, STARPU_R);
                starpu_data_release(cellHandles[idxLevel][idxHandle].down);
            }
        }
        {
            for(int idxHandle = 0 ; idxHandle < int(particleHandles.size()) ; ++idxHandle){
                starpu_data_acquire(particleHandles[idxHandle].symb, STARPU_R);
                starpu_data_release(particleHandles[idxHandle].symb);
                starpu_data_acquire(particleHandles[idxHandle].down, STARPU_R);
                starpu_data_release(particleHandles[idxHandle].down);
            }
        }
    }

    ~FGroupTaskStarPUMpiAlgorithm(){
        starpu_resume();

        cleanHandle();
        cleanHandleMpi();
        delete[] cellHandles;

        starpu_pthread_mutex_t releaseMutex;
        starpu_pthread_mutex_init(&releaseMutex, NULL);
#ifdef STARPU_USE_CPU
        FStarPUUtils::ExecOnWorkers(STARPU_CPU, [&](){
            starpu_pthread_mutex_lock(&releaseMutex);
            cpuWrapper.releaseKernel(starpu_worker_get_id());
            starpu_pthread_mutex_unlock(&releaseMutex);
        });
        wrappers.set(FSTARPU_CPU_IDX, &cpuWrapper);
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        FStarPUUtils::ExecOnWorkers(STARPU_CUDA, [&](){
            starpu_pthread_mutex_lock(&releaseMutex);
            cudaWrapper.releaseKernel(starpu_worker_get_id());
            starpu_pthread_mutex_unlock(&releaseMutex);
        });
        wrappers.set(FSTARPU_CUDA_IDX, &cudaWrapper);
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        FStarPUUtils::ExecOnWorkers(STARPU_OPENCL, [&](){
            starpu_pthread_mutex_lock(&releaseMutex);
            openclWrapper.releaseKernel(starpu_worker_get_id());
            starpu_pthread_mutex_unlock(&releaseMutex);
        });
        wrappers.set(FSTARPU_OPENCL_IDX, &openclWrapper);
#endif
        starpu_pthread_mutex_destroy(&releaseMutex);


#ifdef STARPU_SUPPORT_ARBITER
        starpu_arbiter_destroy(arbiterGlobal);
#endif
        starpu_mpi_shutdown();
        starpu_shutdown();
    }

    void rebuildInteractions(){
        FAssertLF(getenv("OMP_WAIT_POLICY") == nullptr
                || strcmp(getenv("OMP_WAIT_POLICY"), "PASSIVE") == 0
                  || strcmp(getenv("OMP_WAIT_POLICY"), "passive") == 0);

#pragma omp parallel
#pragma omp single
        buildExternalInteractionVecs();
        buildHandles();

#pragma omp parallel
#pragma omp single
        buildRemoteInteractionsAndHandles();
    }

#ifdef STARPU_USE_CPU
    void forEachCpuWorker(std::function<void(void)> func){
        starpu_resume();
        FStarPUUtils::ExecOnWorkers(STARPU_CPU, func);
        starpu_pause();
    }

    void forEachCpuWorker(std::function<void(KernelClass*)> func){
        starpu_resume();
        FStarPUUtils::ExecOnWorkers(STARPU_CPU, [&](){
            func(cpuWrapper.getKernel(starpu_worker_get_id()));
        });
        starpu_pause();
    }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
    void forEachCudaWorker(std::function<void(void)> func){
        starpu_resume();
        FStarPUUtils::ExecOnWorkers(STARPU_CUDA, func);
        starpu_pause();
    }
    void forEachCudaWorker(std::function<void(void*)> func){
        starpu_resume();
        FStarPUUtils::ExecOnWorkers(STARPU_CUDA, [&](){
            func(cudaWrapper.getKernel(starpu_worker_get_id()));
        });
        starpu_pause();
    }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
    void forEachOpenCLWorker(std::function<void(void)> func){
        starpu_resume();
        FStarPUUtils::ExecOnWorkers(STARPU_OPENCL, func);
        starpu_pause();
    }
    void forEachOpenCLWorker(std::function<void(void*)> func){
        starpu_resume();
        FStarPUUtils::ExecOnWorkers(STARPU_OPENCL, [&](){
            func(openclWrapper.getKernel(starpu_worker_get_id()));
        });
        starpu_pause();
    }
#endif

protected:

    /**
      * Runs the complete algorithm.
      */
    void executeCore(const unsigned operationsToProceed) override {
        FLOG( FLog::Controller << "\tStart FGroupTaskStarPUMpiAlgorithm\n" );
        const bool directOnly = (tree->getHeight() <= 2);
#ifdef STARPU_USE_CPU
        FTIME_TASKS(cpuWrapper.taskTimeRecorder.start());
#endif

        FLOG(FTic timerSoumission);

        starpu_resume();
        postRecvAllocatedBlocks();

        if( operationsToProceed & FFmmP2P ) insertParticlesSend();
        if( operationsToProceed & FFmmP2P ) directPass();
        if( operationsToProceed & FFmmP2P ) directPassMpi();

        if(operationsToProceed & FFmmP2M && !directOnly) bottomPass();
        if(operationsToProceed & FFmmM2M && !directOnly) upwardPass();
        if(operationsToProceed & FFmmM2L && !directOnly) insertCellsSend();
         if(operationsToProceed & FFmmM2L && !directOnly) transferPass(FAbstractAlgorithm::upperWorkingLevel, FAbstractAlgorithm::lowerWorkingLevel-1 , true, true);
        if(operationsToProceed & FFmmM2L && !directOnly) transferPass(FAbstractAlgorithm::lowerWorkingLevel-1, FAbstractAlgorithm::lowerWorkingLevel, false, false);
        if(operationsToProceed & FFmmM2L && !directOnly) transferPassMpi();
         if(operationsToProceed & FFmmL2L && !directOnly) downardPass();

        if(operationsToProceed & FFmmM2L && !directOnly) transferPass(FAbstractAlgorithm::lowerWorkingLevel-1, FAbstractAlgorithm::lowerWorkingLevel, true, true);

        if( operationsToProceed & FFmmL2P && !directOnly) mergePass();
#ifdef STARPU_USE_REDUX
        if( operationsToProceed & FFmmL2P && !directOnly) readParticle();
#endif
        FLOG( FLog::Controller << "\t\t Submitting the tasks took " << timerSoumission.tacAndElapsed() << "s\n" );
        starpu_task_wait_for_all();
        FLOG( FTic timerSync; );
        syncData();
        FLOG( FLog::Controller << "\t\t Moving data to the host took " << timerSync.tacAndElapsed() << "s\n" );
        starpu_pause();

#ifdef STARPU_USE_CPU
        FTIME_TASKS(cpuWrapper.taskTimeRecorder.end());
        FTIME_TASKS(cpuWrapper.taskTimeRecorder.saveToDisk("/tmp/taskstime-FGroupTaskStarPUAlgorithm.txt"));
#endif
    }

    void initCodelet(){
        memset(&p2m_cl, 0, sizeof(p2m_cl));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportP2M(FSTARPU_CPU_IDX)){
            p2m_cl.cpu_funcs[0] = StarPUCpuWrapperClass::bottomPassCallback;
            p2m_cl.where |= STARPU_CPU;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportP2M(FSTARPU_CUDA_IDX)){
            p2m_cl.cuda_funcs[0] = StarPUCudaWrapperClass::bottomPassCallback;
            p2m_cl.where |= STARPU_CUDA;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportP2M(FSTARPU_OPENCL_IDX)){
            p2m_cl.opencl_funcs[0] = StarPUOpenClWrapperClass::bottomPassCallback;
            p2m_cl.where |= STARPU_OPENCL;
        }
#endif
        p2m_cl.nbuffers = 3;
        p2m_cl.modes[0] = STARPU_R;
        p2m_cl.modes[1] = STARPU_RW;
        p2m_cl.modes[2] = STARPU_R;
        p2m_cl.name = "p2m_cl";

        memset(&m2m_cl, 0, sizeof(m2m_cl));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportM2M(FSTARPU_CPU_IDX)){
            m2m_cl.cpu_funcs[0] = StarPUCpuWrapperClass::upwardPassCallback;
            m2m_cl.where |= STARPU_CPU;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportM2M(FSTARPU_CUDA_IDX)){
            m2m_cl.cuda_funcs[0] = StarPUCudaWrapperClass::upwardPassCallback;
            m2m_cl.where |= STARPU_CUDA;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportM2M(FSTARPU_OPENCL_IDX)){
            m2m_cl.opencl_funcs[0] = StarPUOpenClWrapperClass::upwardPassCallback;
            m2m_cl.where |= STARPU_OPENCL;
        }
#endif
        m2m_cl.nbuffers = 4;
        m2m_cl.dyn_modes = (starpu_data_access_mode*)malloc(m2m_cl.nbuffers*sizeof(starpu_data_access_mode));
        m2m_cl.dyn_modes[0] = STARPU_R;
        m2m_cl.dyn_modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED);
        m2m_cl.name = "m2m_cl";
        m2m_cl.dyn_modes[2] = STARPU_R;
        m2m_cl.dyn_modes[3] = STARPU_R;

        memset(&l2l_cl, 0, sizeof(l2l_cl));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportL2L(FSTARPU_CPU_IDX)){
            l2l_cl.cpu_funcs[0] = StarPUCpuWrapperClass::downardPassCallback;
            l2l_cl.where |= STARPU_CPU;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportL2L(FSTARPU_CUDA_IDX)){
            l2l_cl.cuda_funcs[0] = StarPUCudaWrapperClass::downardPassCallback;
            l2l_cl.where |= STARPU_CUDA;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportL2L(FSTARPU_OPENCL_IDX)){
            l2l_cl.opencl_funcs[0] = StarPUOpenClWrapperClass::downardPassCallback;
            l2l_cl.where |= STARPU_OPENCL;
        }
#endif
        l2l_cl.nbuffers = 4;
        l2l_cl.dyn_modes = (starpu_data_access_mode*)malloc(l2l_cl.nbuffers*sizeof(starpu_data_access_mode));
        l2l_cl.dyn_modes[0] = STARPU_R;
        l2l_cl.dyn_modes[1] = STARPU_R;
        l2l_cl.name = "l2l_cl";
        l2l_cl.dyn_modes[2] = STARPU_R;
        l2l_cl.dyn_modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED);

        memset(&l2l_cl_nocommute, 0, sizeof(l2l_cl_nocommute));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportL2L(FSTARPU_CPU_IDX)){
            l2l_cl_nocommute.cpu_funcs[0] = StarPUCpuWrapperClass::downardPassCallback;
            l2l_cl_nocommute.where |= STARPU_CPU;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportL2L(FSTARPU_CUDA_IDX)){
            l2l_cl_nocommute.cuda_funcs[0] = StarPUCudaWrapperClass::downardPassCallback;
            l2l_cl_nocommute.where |= STARPU_CUDA;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportL2L(FSTARPU_OPENCL_IDX)){
            l2l_cl_nocommute.opencl_funcs[0] = StarPUOpenClWrapperClass::downardPassCallback;
            l2l_cl_nocommute.where |= STARPU_OPENCL;
        }
#endif
        l2l_cl_nocommute.nbuffers = 4;
        l2l_cl_nocommute.dyn_modes = (starpu_data_access_mode*)malloc(l2l_cl_nocommute.nbuffers*sizeof(starpu_data_access_mode));
        l2l_cl_nocommute.dyn_modes[0] = STARPU_R;
        l2l_cl_nocommute.dyn_modes[1] = STARPU_R;
        l2l_cl_nocommute.name = "l2l_cl";
        l2l_cl_nocommute.dyn_modes[2] = STARPU_R;
        l2l_cl_nocommute.dyn_modes[3] = STARPU_RW;

        memset(&l2p_cl, 0, sizeof(l2p_cl));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportL2P(FSTARPU_CPU_IDX)){
            l2p_cl.cpu_funcs[0] = StarPUCpuWrapperClass::mergePassCallback;
            l2p_cl.where |= STARPU_CPU;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportL2P(FSTARPU_CUDA_IDX)){
            l2p_cl.cuda_funcs[0] = StarPUCudaWrapperClass::mergePassCallback;
            l2p_cl.where |= STARPU_CUDA;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportL2P(FSTARPU_OPENCL_IDX)){
            l2p_cl.opencl_funcs[0] = StarPUOpenClWrapperClass::mergePassCallback;
            l2p_cl.where |= STARPU_OPENCL;
        }
#endif
        l2p_cl.nbuffers = 4;
        l2p_cl.modes[0] = STARPU_R;
        l2p_cl.modes[1] = STARPU_R;
        l2p_cl.modes[2] = STARPU_R;
#ifdef STARPU_USE_REDUX
        l2p_cl.modes[3] = STARPU_REDUX;
#else
        l2p_cl.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED);
#endif
        l2p_cl.name = "l2p_cl";

        memset(&p2p_cl_in, 0, sizeof(p2p_cl_in));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportP2P(FSTARPU_CPU_IDX)){
            p2p_cl_in.cpu_funcs[0] = StarPUCpuWrapperClass::directInPassCallback;
            p2p_cl_in.where |= STARPU_CPU;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportP2P(FSTARPU_CUDA_IDX)){
            p2p_cl_in.cuda_funcs[0] = StarPUCudaWrapperClass::directInPassCallback;
            p2p_cl_in.where |= STARPU_CUDA;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportP2P(FSTARPU_OPENCL_IDX)){
            p2p_cl_in.opencl_funcs[0] = StarPUOpenClWrapperClass::directInPassCallback;
            p2p_cl_in.where |= STARPU_OPENCL;
        }
#endif
        p2p_cl_in.nbuffers = 2;
        p2p_cl_in.modes[0] = STARPU_R;
#ifdef STARPU_USE_REDUX
        p2p_cl_in.modes[1] = STARPU_REDUX;
#else
        p2p_cl_in.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED);
#endif
        p2p_cl_in.name = "p2p_cl_in";
        memset(&p2p_cl_inout, 0, sizeof(p2p_cl_inout));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportP2PExtern(FSTARPU_CPU_IDX)){
            p2p_cl_inout.cpu_funcs[0] = StarPUCpuWrapperClass::directInoutPassCallback;
            p2p_cl_inout.where |= STARPU_CPU;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportP2PExtern(FSTARPU_CUDA_IDX)){
            p2p_cl_inout.cuda_funcs[0] = StarPUCudaWrapperClass::directInoutPassCallback;
            p2p_cl_inout.where |= STARPU_CUDA;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportP2PExtern(FSTARPU_OPENCL_IDX)){
            p2p_cl_inout.opencl_funcs[0] = StarPUOpenClWrapperClass::directInoutPassCallback;
            p2p_cl_inout.where |= STARPU_OPENCL;
        }
#endif
        p2p_cl_inout.nbuffers = 4;
        p2p_cl_inout.modes[0] = STARPU_R;
#ifdef STARPU_USE_REDUX
        p2p_cl_inout.modes[1] = STARPU_REDUX;
#else
        p2p_cl_inout.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED);
#endif
        p2p_cl_inout.modes[2] = STARPU_R;
#ifdef STARPU_USE_REDUX
        p2p_cl_inout.modes[3] = STARPU_REDUX;
#else
        p2p_cl_inout.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED);
#endif
        p2p_cl_inout.name = "p2p_cl_inout";

        memset(&m2l_cl_in, 0, sizeof(m2l_cl_in));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportM2L(FSTARPU_CPU_IDX)){
            m2l_cl_in.cpu_funcs[0] = StarPUCpuWrapperClass::transferInPassCallback;
            m2l_cl_in.where |= STARPU_CPU;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportM2L(FSTARPU_CUDA_IDX)){
            m2l_cl_in.cuda_funcs[0] = StarPUCudaWrapperClass::transferInPassCallback;
            m2l_cl_in.where |= STARPU_CUDA;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportM2L(FSTARPU_OPENCL_IDX)){
            m2l_cl_in.opencl_funcs[0] = StarPUOpenClWrapperClass::transferInPassCallback;
            m2l_cl_in.where |= STARPU_OPENCL;
        }
#endif
        m2l_cl_in.nbuffers = 3;
        m2l_cl_in.modes[0] = STARPU_R;
        m2l_cl_in.modes[1] = STARPU_R;
        m2l_cl_in.modes[2] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED);
        m2l_cl_in.name = "m2l_cl_in";

        memset(&m2l_cl_inout, 0, sizeof(m2l_cl_inout));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportM2LExtern(FSTARPU_CPU_IDX)){
            m2l_cl_inout.cpu_funcs[0] = StarPUCpuWrapperClass::transferInoutPassCallback;
            m2l_cl_inout.where |= STARPU_CPU;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportM2LExtern(FSTARPU_CUDA_IDX)){
            m2l_cl_inout.cuda_funcs[0] = StarPUCudaWrapperClass::transferInoutPassCallback;
            m2l_cl_inout.where |= STARPU_CUDA;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportM2LExtern(FSTARPU_OPENCL_IDX)){
            m2l_cl_inout.opencl_funcs[0] = StarPUOpenClWrapperClass::transferInoutPassCallback;
            m2l_cl_inout.where |= STARPU_OPENCL;
        }
#endif
        m2l_cl_inout.nbuffers = 4;
        m2l_cl_inout.modes[0] = STARPU_R;
        m2l_cl_inout.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED);
        m2l_cl_inout.modes[2] = STARPU_R;
        m2l_cl_inout.modes[3] = STARPU_R;
        m2l_cl_inout.name = "m2l_cl_inout";

#ifdef STARPU_USE_REDUX
        memset(&p2p_redux_init, 0, sizeof(p2p_redux_init));
#ifdef STARPU_USE_CPU
        p2p_redux_init.cpu_funcs[0] = FStarPUReduxCpu::InitData<typename ParticleGroupClass::ParticleDataType>;
        p2p_redux_init.where |= STARPU_CPU;
#endif
        p2p_redux_init.nbuffers = 1;
        p2p_redux_init.modes[0] = STARPU_RW;
        p2p_redux_init.name = "p2p_redux_init";

        memset(&p2p_redux_perform, 0, sizeof(p2p_redux_perform));
#ifdef STARPU_USE_CPU
        p2p_redux_perform.cpu_funcs[0] = FStarPUReduxCpu::ReduceData<typename ParticleGroupClass::ParticleDataType>;
        p2p_redux_perform.where |= STARPU_CPU;
#endif
        p2p_redux_perform.nbuffers = 2;
        p2p_redux_perform.modes[0] = STARPU_RW;
        p2p_redux_perform.modes[1] = STARPU_R;
        p2p_redux_perform.name = "p2p_redux_perform";

        memset(&p2p_redux_read, 0, sizeof(p2p_redux_read));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportL2P(FSTARPU_CPU_IDX)){
            p2p_redux_read.cpu_funcs[0] = FStarPUReduxCpu::EmptyCodelet<typename ParticleGroupClass::ParticleDataType>;
            p2p_redux_read.where |= STARPU_CPU;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportL2P(FSTARPU_CUDA_IDX)){
            p2p_redux_read.cuda_funcs[0] = FStarPUReduxCpu::EmptyCodelet<typename ParticleGroupClass::ParticleDataType>;
            p2p_redux_read.where |= STARPU_CUDA;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportL2P(FSTARPU_OPENCL_IDX)){
            p2p_redux_read.opencl_funcs[0] = FStarPUReduxCpu::EmptyCodelet<typename ParticleGroupClass::ParticleDataType>;
            p2p_redux_read.where |= STARPU_OPENCL;
        }
#endif
        p2p_redux_read.nbuffers = 1;
        p2p_redux_read.modes[0] = STARPU_R;
        p2p_redux_read.name = "p2p_redux_read";
#endif
    }

    /** dealloc in a starpu way all the defined handles */
    void cleanHandle(){
        for(int idxLevel = 0 ; idxLevel < tree->getHeight() ; ++idxLevel){
            for(int idxHandle = 0 ; idxHandle < int(cellHandles[idxLevel].size()) ; ++idxHandle){
                starpu_data_unregister(cellHandles[idxLevel][idxHandle].symb);
                starpu_data_unregister(cellHandles[idxLevel][idxHandle].up);
                starpu_data_unregister(cellHandles[idxLevel][idxHandle].down);
            }
            cellHandles[idxLevel].clear();
        }
        {
            for(int idxHandle = 0 ; idxHandle < int(particleHandles.size()) ; ++idxHandle){
                starpu_data_unregister(particleHandles[idxHandle].symb);
                starpu_data_unregister(particleHandles[idxHandle].down);
            }
            particleHandles.clear();
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    void initCodeletMpi(){
        memset(&p2p_cl_inout_mpi, 0, sizeof(p2p_cl_inout_mpi));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportP2PMpi(FSTARPU_CPU_IDX)){
            p2p_cl_inout_mpi.where |= STARPU_CPU;
            p2p_cl_inout_mpi.cpu_funcs[0] = StarPUCpuWrapperClass::directInoutPassCallbackMpi;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportP2PMpi(FSTARPU_CUDA_IDX)){
            p2p_cl_inout_mpi.where |= STARPU_CUDA;
            p2p_cl_inout_mpi.cuda_funcs[0] = StarPUCudaWrapperClass::directInoutPassCallbackMpi;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportP2PMpi(FSTARPU_OPENCL_IDX)){
            p2p_cl_inout_mpi.where |= STARPU_OPENCL;
            p2p_cl_inout_mpi.opencl_funcs[0] = StarPUOpenClWrapperClass::directInoutPassCallbackMpi;
        }
#endif
        p2p_cl_inout_mpi.nbuffers = 3;
        p2p_cl_inout_mpi.modes[0] = STARPU_R;
        p2p_cl_inout_mpi.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED);
        p2p_cl_inout_mpi.modes[2] = STARPU_R;
        p2p_cl_inout_mpi.name = "p2p_cl_inout_mpi";

        memset(&m2l_cl_inout_mpi, 0, sizeof(m2l_cl_inout_mpi));
#ifdef STARPU_USE_CPU
        if(originalCpuKernel->supportM2LMpi(FSTARPU_CPU_IDX)){
            m2l_cl_inout_mpi.where |= STARPU_CPU;
            m2l_cl_inout_mpi.cpu_funcs[0] = StarPUCpuWrapperClass::transferInoutPassCallbackMpi;
        }
#endif
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
        if(originalCpuKernel->supportM2LMpi(FSTARPU_CUDA_IDX)){
            m2l_cl_inout_mpi.where |= STARPU_CUDA;
            m2l_cl_inout_mpi.cuda_funcs[0] = StarPUCudaWrapperClass::transferInoutPassCallbackMpi;
        }
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
        if(originalCpuKernel->supportM2LMpi(FSTARPU_OPENCL_IDX)){
            m2l_cl_inout_mpi.where |= STARPU_OPENCL;
            m2l_cl_inout_mpi.opencl_funcs[0] = StarPUOpenClWrapperClass::transferInoutPassCallbackMpi;
        }
#endif
        m2l_cl_inout_mpi.nbuffers = 4;
        m2l_cl_inout_mpi.modes[0] = STARPU_R;
        m2l_cl_inout_mpi.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED);
        m2l_cl_inout_mpi.modes[2] = STARPU_R;
        m2l_cl_inout_mpi.modes[3] = STARPU_R;
        m2l_cl_inout_mpi.name = "m2l_cl_inout_mpi";
    }

    std::vector<std::pair<MortonIndex,MortonIndex>> processesIntervalPerLevels;
    struct BlockDescriptor{
        MortonIndex firstIndex;
        MortonIndex lastIndex;
		int globalIdx;
        int owner;
        int bufferSize;
        size_t bufferSizeSymb;
        size_t bufferSizeUp;
        size_t bufferSizeDown;
        size_t leavesBufferSize;
    };
    std::vector<std::vector<BlockDescriptor>> processesBlockInfos;
    std::vector<int> nbBlocksPerLevelAll;
    std::vector<int> nbBlocksBeforeMinPerLevel;

    std::vector< std::vector< std::vector<BlockInteractions<CellContainerClass>>>> externalInteractionsAllLevelMpi;
    std::vector< std::vector<BlockInteractions<ParticleGroupClass>>> externalInteractionsLeafLevelMpi;

    struct RemoteHandle{
        RemoteHandle() : ptrSymb(nullptr), ptrUp(nullptr), ptrDown(nullptr){
            memset(&handleSymb, 0, sizeof(handleSymb));
            memset(&handleUp, 0, sizeof(handleUp));
            memset(&handleDown, 0, sizeof(handleDown));
        }

        unsigned char * ptrSymb;
        starpu_data_handle_t handleSymb;
        unsigned char * ptrUp;
        starpu_data_handle_t handleUp;
        unsigned char * ptrDown;
        starpu_data_handle_t handleDown;

        int intervalSize;
    };

    std::vector<std::vector<RemoteHandle>> remoteCellGroups;
    std::vector<RemoteHandle> remoteParticleGroupss;

    void buildRemoteInteractionsAndHandles(){
        cleanHandleMpi();

        // We need to have information about all other blocks
        std::unique_ptr<int[]> nbBlocksPerLevel(new int[tree->getHeight()]);
        nbBlocksPerLevel[0] = 0;
        for(int idxLevel = 1 ; idxLevel < tree->getHeight() ; ++idxLevel){
            nbBlocksPerLevel[idxLevel] = tree->getNbCellGroupAtLevel(idxLevel);
        }
        // Exchange the number of blocks per proc
        nbBlocksPerLevelAll.resize(tree->getHeight() * comm.processCount());
        FMpi::Assert(MPI_Allgather(nbBlocksPerLevel.get(), tree->getHeight(), MPI_INT,
                                   nbBlocksPerLevelAll.data(), tree->getHeight(), MPI_INT,
                                   comm.getComm()), __LINE__);
        // Compute the number of blocks before mine
        nbBlocksBeforeMinPerLevel.resize(tree->getHeight());
        for(int idxLevel = 1 ; idxLevel < tree->getHeight() ; ++idxLevel){
            nbBlocksBeforeMinPerLevel[idxLevel] = 0;
            for(int idxProc = 0 ; idxProc < comm.processId() ; ++idxProc){
                nbBlocksBeforeMinPerLevel[idxLevel] += nbBlocksPerLevelAll[idxProc*tree->getHeight() + idxLevel];
            }
        }
        // Prepare the block infos
        processesBlockInfos.resize(tree->getHeight());
        std::unique_ptr<int[]> recvBlocksCount(new int[comm.processCount()]);
        std::unique_ptr<int[]> recvBlockDispl(new int[comm.processCount()]);
        // Exchange the block info per level
        for(int idxLevel = 1 ; idxLevel < tree->getHeight() ; ++idxLevel){
            // Count the total number of blocks
            int nbBlocksInLevel = 0;
            recvBlockDispl[0] = 0;
            for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
                nbBlocksInLevel += nbBlocksPerLevelAll[idxProc*tree->getHeight() + idxLevel];
                // Count and displacement for the MPI all gatherv
                recvBlocksCount[idxProc] = nbBlocksPerLevelAll[idxProc*tree->getHeight() + idxLevel] * int(sizeof(BlockDescriptor));
                if(idxProc) recvBlockDispl[idxProc] = recvBlockDispl[idxProc-1] + recvBlocksCount[idxProc-1];
            }
            processesBlockInfos[idxLevel].resize(nbBlocksInLevel);
            // Fill my blocks
            std::vector<BlockDescriptor> myBlocksAtLevel;
            myBlocksAtLevel.resize(nbBlocksPerLevel[idxLevel]);
            FAssertLF(tree->getNbCellGroupAtLevel(idxLevel) == int(myBlocksAtLevel.size()));
            FAssertLF(nbBlocksPerLevel[idxLevel] == nbBlocksPerLevelAll[comm.processId()*tree->getHeight() + idxLevel]);

            for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
                CellContainerClass*const currentCells = tree->getCellGroup(idxLevel, idxGroup);
                myBlocksAtLevel[idxGroup].firstIndex = currentCells->getStartingIndex();
                myBlocksAtLevel[idxGroup].lastIndex  = currentCells->getEndingIndex();
                myBlocksAtLevel[idxGroup].owner = comm.processId();
				myBlocksAtLevel[idxGroup].globalIdx = nbBlocksBeforeMinPerLevel[idxLevel] + idxGroup;
                myBlocksAtLevel[idxGroup].bufferSizeSymb = currentCells->getBufferSizeInByte();
                myBlocksAtLevel[idxGroup].bufferSizeUp   = currentCells->getMultipoleBufferSizeInByte();
                myBlocksAtLevel[idxGroup].bufferSizeDown = currentCells->getLocalBufferSizeInByte();

                if(idxLevel == tree->getHeight() - 1){
                    myBlocksAtLevel[idxGroup].leavesBufferSize = tree->getParticleGroup(idxGroup)->getBufferSizeInByte();
                }
                else{
                    myBlocksAtLevel[idxGroup].leavesBufferSize = 0;
                }
            }
            // Exchange with all other
            FMpi::Assert(MPI_Allgatherv(myBlocksAtLevel.data(), int(myBlocksAtLevel.size()*sizeof(BlockDescriptor)), MPI_BYTE,
                                        processesBlockInfos[idxLevel].data(), recvBlocksCount.get(), recvBlockDispl.get(), MPI_BYTE,
                                        comm.getComm()), __LINE__);
        }
        // Prepare remate ptr and handles
        remoteCellGroups.resize( tree->getHeight() );
        for(int idxLevel = 1 ; idxLevel < tree->getHeight() ; ++idxLevel){
            remoteCellGroups[idxLevel].resize( processesBlockInfos[idxLevel].size());
        }
        remoteParticleGroupss.resize(processesBlockInfos[tree->getHeight()-1].size());

        // From now we have the number of blocks for all process
        // we also have the size of the blocks therefor we can
        // create the handles we need
        // We will now detect the relation between our blocks and others
        // During the M2M (which is the same for the L2L)
        // During the M2L and during the P2P
        // I need to insert the task that read my data or that write the data I need.
        // M2L
        externalInteractionsAllLevelMpi.clear();
        externalInteractionsAllLevelMpi.resize(tree->getHeight());
        for(int idxLevel = tree->getHeight()-1 ; idxLevel >= 2 ; --idxLevel){
            // From this level there are no more blocks
            if(tree->getNbCellGroupAtLevel(idxLevel) == 0){
                // We stop here
                break;
            }
            // What are my morton interval at this level
            const MortonIndex myFirstIndex = tree->getCellGroup(idxLevel, 0)->getStartingIndex();
            const MortonIndex myLastIndex = tree->getCellGroup(idxLevel, tree->getNbCellGroupAtLevel(idxLevel)-1)->getEndingIndex();

            externalInteractionsAllLevelMpi[idxLevel].resize(tree->getNbCellGroupAtLevel(idxLevel));

            for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
                CellContainerClass* currentCells = tree->getCellGroup(idxLevel, idxGroup);

                std::vector<BlockInteractions<CellContainerClass>>* externalInteractions = &externalInteractionsAllLevelMpi[idxLevel][idxGroup];

#pragma omp task default(none) firstprivate(idxGroup, currentCells, idxLevel, externalInteractions)
                {
                    std::vector<OutOfBlockInteraction> outsideInteractions;

                    for(int idxCell = 0 ; idxCell < currentCells->getNumberOfCellsInBlock() ; ++idxCell){
                        const MortonIndex mindex = currentCells->getCellMortonIndex(idxCell);

                        MortonIndex interactionsIndexes[189];
                        int interactionsPosition[189];
                        const FTreeCoordinate coord(mindex);
                        int counter = coord.getInteractionNeighbors(idxLevel,interactionsIndexes,interactionsPosition);

                        for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
                            // This interactions need a block owned by someoneelse
                            if(interactionsIndexes[idxInter] < myFirstIndex || myLastIndex <= interactionsIndexes[idxInter]){
                                OutOfBlockInteraction property;
                                property.insideIndex = mindex;
                                property.outIndex    = interactionsIndexes[idxInter];
                                property.relativeOutPosition = interactionsPosition[idxInter];
                                property.insideIdxInBlock = idxCell;
                                property.outsideIdxInBlock = -1;
                                outsideInteractions.push_back(property);
                            }
                        }
                    }

                    // Manage outofblock interaction
                    FQuickSort<OutOfBlockInteraction, int>::QsSequential(outsideInteractions.data(),int(outsideInteractions.size()));

                    int currentOutInteraction = 0;
                    for(int idxOtherGroup = 0 ; idxOtherGroup < int(processesBlockInfos[idxLevel].size())
                        && currentOutInteraction < int(outsideInteractions.size()) ; ++idxOtherGroup){
                        // Skip my blocks
                        if(idxOtherGroup == nbBlocksBeforeMinPerLevel[idxLevel]){
                            idxOtherGroup += tree->getNbCellGroupAtLevel(idxLevel);
                            if(idxOtherGroup == int(processesBlockInfos[idxLevel].size())){
                                break;
                            }
                            FAssertLF(idxOtherGroup < int(processesBlockInfos[idxLevel].size()));
                        }

                        const MortonIndex blockStartIdxOther = processesBlockInfos[idxLevel][idxOtherGroup].firstIndex;
                        const MortonIndex blockEndIdxOther   = processesBlockInfos[idxLevel][idxOtherGroup].lastIndex;

                        while(currentOutInteraction < int(outsideInteractions.size()) && outsideInteractions[currentOutInteraction].outIndex < blockStartIdxOther){
                            currentOutInteraction += 1;
                        }

                        int lastOutInteraction = currentOutInteraction;
                        while(lastOutInteraction < int(outsideInteractions.size()) && outsideInteractions[lastOutInteraction].outIndex < blockEndIdxOther){
                            lastOutInteraction += 1;
                        }

                        // Create interactions
                        const int nbInteractionsBetweenBlocks = (lastOutInteraction-currentOutInteraction);
                        if(nbInteractionsBetweenBlocks){
                            if(remoteCellGroups[idxLevel][idxOtherGroup].ptrSymb == nullptr){
#pragma omp critical(CreateM2LRemotes)
                                {
                                    if(remoteCellGroups[idxLevel][idxOtherGroup].ptrSymb == nullptr){
                                        const size_t nbBytesInBlockSymb = processesBlockInfos[idxLevel][idxOtherGroup].bufferSizeSymb;
                                        unsigned char* memoryBlockSymb = (unsigned char*)FAlignedMemory::AllocateBytes<32>(nbBytesInBlockSymb);
                                        remoteCellGroups[idxLevel][idxOtherGroup].ptrSymb = memoryBlockSymb;
                                        starpu_variable_data_register(&remoteCellGroups[idxLevel][idxOtherGroup].handleSymb, 0,
                                                                      (uintptr_t)remoteCellGroups[idxLevel][idxOtherGroup].ptrSymb, nbBytesInBlockSymb);
                                        const size_t nbBytesInBlockUp = processesBlockInfos[idxLevel][idxOtherGroup].bufferSizeUp;
                                        unsigned char* memoryBlockUp = (unsigned char*)FAlignedMemory::AllocateBytes<32>(nbBytesInBlockUp);
                                        remoteCellGroups[idxLevel][idxOtherGroup].ptrUp = memoryBlockUp;
                                        starpu_variable_data_register(&remoteCellGroups[idxLevel][idxOtherGroup].handleUp, 0,
                                                                      (uintptr_t)remoteCellGroups[idxLevel][idxOtherGroup].ptrUp, nbBytesInBlockUp);
                                    }
                                }
                            }

                            externalInteractions->emplace_back();
                            BlockInteractions<CellContainerClass>* interactions = &externalInteractions->back();
                            //interactions->otherBlock = remoteCellGroups[idxLevel][idxOtherGroup].ptr;
                            interactions->otherBlockId = idxOtherGroup;
                            interactions->interactions.resize(nbInteractionsBetweenBlocks);
                            std::copy(outsideInteractions.begin() + currentOutInteraction,
                                      outsideInteractions.begin() + lastOutInteraction,
                                      interactions->interactions.begin());
                        }

                        currentOutInteraction = lastOutInteraction;
                    }
                }
            }
        }
        // P2P
        // We create one big vector per block
        {
            const MortonIndex myFirstIndex = tree->getParticleGroup(0)->getStartingIndex();
            const MortonIndex myLastIndex = tree->getParticleGroup(tree->getNbParticleGroup()-1)->getEndingIndex();

            externalInteractionsLeafLevelMpi.clear();
            externalInteractionsLeafLevelMpi.resize(tree->getNbParticleGroup());
            for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
                // Create the vector
                ParticleGroupClass* containers = tree->getParticleGroup(idxGroup);

                std::vector<BlockInteractions<ParticleGroupClass>>* externalInteractions = &externalInteractionsLeafLevelMpi[idxGroup];

#pragma omp task default(none) firstprivate(idxGroup, containers, externalInteractions)
                { // Can be a task(inout:iterCells)
                    std::vector<OutOfBlockInteraction> outsideInteractions;

                    for(int idxLeaf = 0 ; idxLeaf < containers->getNumberOfLeavesInBlock() ; ++idxLeaf){
                        // ParticleContainerClass particles = containers->template getLeaf<ParticleContainerClass>(mindex);
                        const MortonIndex mindex = containers->getLeafMortonIndex(idxLeaf);
                        if(containers->exists(mindex)){
                            MortonIndex interactionsIndexes[26];
                            int interactionsPosition[26];
                            FTreeCoordinate coord(mindex);
                            int counter = coord.getNeighborsIndexes(tree->getHeight(),interactionsIndexes,interactionsPosition);

                            for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
                                if(interactionsIndexes[idxInter] < myFirstIndex ||
                                        myLastIndex <= interactionsIndexes[idxInter]){
                                    OutOfBlockInteraction property;
                                    property.insideIndex = mindex;
                                    property.outIndex    = interactionsIndexes[idxInter];
                                    property.relativeOutPosition = interactionsPosition[idxInter];
                                    property.outsideIdxInBlock = -1;
                                    property.insideIdxInBlock = idxLeaf;
                                    outsideInteractions.push_back(property);
                                }
                            }
                        }
                    }

                    // Sort to match external order
                    FQuickSort<OutOfBlockInteraction, int>::QsSequential(outsideInteractions.data(),int(outsideInteractions.size()));

                    int currentOutInteraction = 0;
                    for(int idxOtherGroup = 0 ; idxOtherGroup < int(processesBlockInfos[tree->getHeight()-1].size())
                        && currentOutInteraction < int(outsideInteractions.size()) ; ++idxOtherGroup){
                        // Skip my blocks
                        if(idxOtherGroup == nbBlocksBeforeMinPerLevel[tree->getHeight()-1]){
                            idxOtherGroup += tree->getNbCellGroupAtLevel(tree->getHeight()-1);
                            if(idxOtherGroup == int(processesBlockInfos[tree->getHeight()-1].size())){
                                break;
                            }
                            FAssertLF(idxOtherGroup < int(processesBlockInfos[tree->getHeight()-1].size()));
                        }

                        const MortonIndex blockStartIdxOther = processesBlockInfos[tree->getHeight()-1][idxOtherGroup].firstIndex;
                        const MortonIndex blockEndIdxOther   = processesBlockInfos[tree->getHeight()-1][idxOtherGroup].lastIndex;

                        while(currentOutInteraction < int(outsideInteractions.size()) && outsideInteractions[currentOutInteraction].outIndex < blockStartIdxOther){
                            currentOutInteraction += 1;
                        }

                        int lastOutInteraction = currentOutInteraction;
                        while(lastOutInteraction < int(outsideInteractions.size()) && outsideInteractions[lastOutInteraction].outIndex < blockEndIdxOther){
                            lastOutInteraction += 1;
                        }

                        const int nbInteractionsBetweenBlocks = (lastOutInteraction-currentOutInteraction);
                        if(nbInteractionsBetweenBlocks){
                            if(remoteParticleGroupss[idxOtherGroup].ptrSymb == nullptr){
#pragma omp critical(CreateM2LRemotes)
                                {
                                    if(remoteParticleGroupss[idxOtherGroup].ptrSymb == nullptr){
                                        const size_t nbBytesInBlock = processesBlockInfos[tree->getHeight()-1][idxOtherGroup].leavesBufferSize;
                                        unsigned char* memoryBlock = (unsigned char*)FAlignedMemory::AllocateBytes<32>(nbBytesInBlock);
                                        remoteParticleGroupss[idxOtherGroup].ptrSymb = memoryBlock;
                                        starpu_variable_data_register(&remoteParticleGroupss[idxOtherGroup].handleSymb, 0,
                                                                      (uintptr_t)remoteParticleGroupss[idxOtherGroup].ptrSymb, nbBytesInBlock);
                                    }
                                }
                            }

                            externalInteractions->emplace_back();
                            BlockInteractions<ParticleGroupClass>* interactions = &externalInteractions->back();
                            //interactions->otherBlock = remoteParticleGroupss[idxOtherGroup].ptr;
                            interactions->otherBlockId = idxOtherGroup;
                            interactions->interactions.resize(nbInteractionsBetweenBlocks);
                            std::copy(outsideInteractions.begin() + currentOutInteraction,
                                      outsideInteractions.begin() + lastOutInteraction,
                                      interactions->interactions.begin());
                        }

                        currentOutInteraction = lastOutInteraction;
                    }
                }
            }
        }
    }

    struct MpiDependency{
        int src;
        int dest;
        int level;
        int globalBlockId;
    };

    std::vector<MpiDependency> toSend;

    void postRecvAllocatedBlocks(){
        std::vector<MpiDependency> toRecv;
        FAssertLF(tree->getHeight() == int(remoteCellGroups.size()));
        const bool directOnly = (tree->getHeight() <= 2);
        // We do not perform the real send here because it will be mixed with starpu call to MPI
        if(!directOnly){
            for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel){
                for(int idxHandle = 0 ; idxHandle < int(remoteCellGroups[idxLevel].size()) ; ++idxHandle){
                    if(remoteCellGroups[idxLevel][idxHandle].ptrSymb){
                        FAssertLF(remoteCellGroups[idxLevel][idxHandle].ptrUp);
                        // starpu_mpi_irecv_detached cannot be sent here

                        toRecv.push_back({processesBlockInfos[idxLevel][idxHandle].owner,
                                          comm.processId(), idxLevel, idxHandle});
                    }
                }
            }
        }
        {
            for(int idxHandle = 0 ; idxHandle < int(remoteParticleGroupss.size()) ; ++idxHandle){
                if(remoteParticleGroupss[idxHandle].ptrSymb){
                    // starpu_mpi_irecv_detached cannot be sent here

                    toRecv.push_back({processesBlockInfos[tree->getHeight()-1][idxHandle].owner,
                                      comm.processId(), tree->getHeight(), idxHandle});
                }
            }
        }

        FQuickSort<MpiDependency, int>::QsSequential(toRecv.data(),int(toRecv.size()),[](const MpiDependency& d1, const MpiDependency& d2){
            return d1.src <= d2.src;
        });

        std::unique_ptr<int[]> nbBlocksToRecvFromEach(new int[comm.processCount()]);
        memset(nbBlocksToRecvFromEach.get(), 0, sizeof(int)*comm.processCount());
        for(int idxDep = 0 ; idxDep < int(toRecv.size()) ; ++idxDep){
            nbBlocksToRecvFromEach[toRecv[idxDep].src] += 1;
        }

        FAssertLF(nbBlocksToRecvFromEach[comm.processId()] == 0);
        int offset = 0;

        for(int idxProc = 0 ; idxProc < comm.processCount() ; ++idxProc){
            if(idxProc == comm.processId()){
                // How much to send to each
                std::unique_ptr<int[]> nbBlocksToSendToEach(new int[comm.processCount()]);
                FMpi::Assert(MPI_Gather(&nbBlocksToRecvFromEach[idxProc], 1,
                                        MPI_INT, nbBlocksToSendToEach.get(), 1,
                                        MPI_INT, idxProc, comm.getComm() ), __LINE__);

                std::unique_ptr<int[]> displs(new int[comm.processCount()]);
                displs[0] = 0;
                for(int idxProcOther = 1 ; idxProcOther < comm.processCount() ; ++idxProcOther){
                    displs[idxProcOther] = displs[idxProcOther-1] + nbBlocksToSendToEach[idxProcOther-1];
                }
                toSend.resize(displs[comm.processCount()-1] + nbBlocksToSendToEach[comm.processCount()-1]);

                // We work in bytes
                for(int idxProcOther = 0 ; idxProcOther < comm.processCount() ; ++idxProcOther){
                    nbBlocksToSendToEach[idxProcOther] *= int(sizeof(MpiDependency));
                    displs[idxProcOther] *= int(sizeof(MpiDependency));
                }

                FMpi::Assert(MPI_Gatherv( nullptr, 0, MPI_BYTE,
                                          toSend.data(),
                                          nbBlocksToSendToEach.get(), displs.get(),
                                          MPI_BYTE, idxProc, comm.getComm()), __LINE__);
            }
            else{
                FMpi::Assert(MPI_Gather(&nbBlocksToRecvFromEach[idxProc], 1,
                                        MPI_INT, 0, 0, MPI_INT, idxProc, comm.getComm() ), __LINE__);
                FMpi::Assert(MPI_Gatherv(
                                 &toRecv[offset], int(nbBlocksToRecvFromEach[idxProc]*sizeof(MpiDependency)), MPI_BYTE,
                                 0, 0, 0, MPI_BYTE, idxProc, comm.getComm() ), __LINE__);

                offset += nbBlocksToRecvFromEach[idxProc];
            }
        }
        FLOG(FLog::Controller.flush() );
        // We can do it here
        if(!directOnly){
            for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel){
                for(int idxHandle = 0 ; idxHandle < int(remoteCellGroups[idxLevel].size()) ; ++idxHandle){
                    if(remoteCellGroups[idxLevel][idxHandle].ptrSymb){
                        FAssertLF(remoteCellGroups[idxLevel][idxHandle].ptrUp);
                        FLOG(FLog::Controller << "[SMpi] " << idxLevel << " Post a recv during M2L for Idx " << processesBlockInfos[idxLevel][idxHandle].firstIndex <<
                             " and dest is " << processesBlockInfos[idxLevel][idxHandle].owner << " tag " << getTag(idxLevel, processesBlockInfos[idxLevel][idxHandle].firstIndex, processesBlockInfos[idxLevel][idxHandle].globalIdx, 0, processesBlockInfos[idxLevel][idxHandle].owner) << "\n");
                        FLOG(FLog::Controller << "[SMpi] " << idxLevel << " Post a recv during M2L for Idx " << processesBlockInfos[idxLevel][idxHandle].firstIndex <<
                             " and dest is " << processesBlockInfos[idxLevel][idxHandle].owner << " tag " << getTag(idxLevel, processesBlockInfos[idxLevel][idxHandle].firstIndex, processesBlockInfos[idxLevel][idxHandle].globalIdx, 1, processesBlockInfos[idxLevel][idxHandle].owner) << "\n");

                        starpu_mpi_irecv_detached( remoteCellGroups[idxLevel][idxHandle].handleSymb,
                                                   processesBlockInfos[idxLevel][idxHandle].owner,
                                                   getTag(idxLevel,processesBlockInfos[idxLevel][idxHandle].firstIndex, processesBlockInfos[idxLevel][idxHandle].globalIdx, 0, processesBlockInfos[idxLevel][idxHandle].owner),
                                                   comm.getComm(), 0, 0 );
                        starpu_mpi_irecv_detached( remoteCellGroups[idxLevel][idxHandle].handleUp,
                                                   processesBlockInfos[idxLevel][idxHandle].owner,
                                                   getTag(idxLevel,processesBlockInfos[idxLevel][idxHandle].firstIndex, processesBlockInfos[idxLevel][idxHandle].globalIdx, 1, processesBlockInfos[idxLevel][idxHandle].owner),
                                                   comm.getComm(), 0, 0 );
                    }
                }
            }
        }
        {
            for(int idxHandle = 0 ; idxHandle < int(remoteParticleGroupss.size()) ; ++idxHandle){
                if(remoteParticleGroupss[idxHandle].ptrSymb){
                    FLOG(FLog::Controller << "[SMpi] Post a recv during P2P for Idx " << processesBlockInfos[tree->getHeight()-1][idxHandle].firstIndex <<
                                  " and dest is " << processesBlockInfos[tree->getHeight()-1][idxHandle].owner << " tag " << getTag(tree->getHeight(), processesBlockInfos[tree->getHeight()-1][idxHandle].firstIndex, processesBlockInfos[tree->getHeight()-1][idxHandle].globalIdx, 0, processesBlockInfos[tree->getHeight()-1][idxHandle].owner) << "\n");

                    starpu_mpi_irecv_detached( remoteParticleGroupss[idxHandle].handleSymb,
                                               processesBlockInfos[tree->getHeight()-1][idxHandle].owner,
                            getTag(tree->getHeight(),processesBlockInfos[tree->getHeight()-1][idxHandle].firstIndex, processesBlockInfos[tree->getHeight()-1][idxHandle].globalIdx, 0, processesBlockInfos[tree->getHeight()-1][idxHandle].owner),
                            comm.getComm(), 0, 0 );
                }
            }
        }
    }

    void insertParticlesSend(){
        for(int idxSd = 0 ; idxSd < int(toSend.size()) ; ++idxSd){
            const MpiDependency sd = toSend[idxSd];
            if(sd.level == tree->getHeight()){
                const int localId = sd.globalBlockId - nbBlocksBeforeMinPerLevel[tree->getHeight()-1];
                FAssertLF(sd.src == comm.processId());
                FAssertLF(0 <= localId);
                FAssertLF(localId < tree->getNbParticleGroup());

                FLOG(FLog::Controller << "[SMpi] Post a send during P2P for Idx " << tree->getParticleGroup(localId)->getStartingIndex() <<
                     " and dest is " << sd.dest << " tag " << getTag(tree->getHeight(), tree->getParticleGroup(localId)->getStartingIndex(), nbBlocksBeforeMinPerLevel[tree->getHeight()-1] + localId, 0, sd.dest) <<  "\n");

                starpu_mpi_isend_detached( particleHandles[localId].symb, sd.dest,
                                           getTag(tree->getHeight(), tree->getParticleGroup(localId)->getStartingIndex(), nbBlocksBeforeMinPerLevel[tree->getHeight()-1] + localId, 0, sd.dest),
                                           comm.getComm(), 0/*callback*/, 0/*arg*/ );
            }
        }
        FLOG(FLog::Controller.flush() );
    }

    void insertCellsSend(){
        for(int idxSd = 0 ; idxSd < int(toSend.size()) ; ++idxSd){
            const MpiDependency sd = toSend[idxSd];
            if(sd.level != tree->getHeight()){
                const int localId = sd.globalBlockId - nbBlocksBeforeMinPerLevel[sd.level];
                FAssertLF(sd.src == comm.processId());
                FAssertLF(0 <= localId);
                FAssertLF(localId < tree->getNbCellGroupAtLevel(sd.level));

                FLOG(FLog::Controller << "[SMpi] " << sd.level << " Post a send during M2L for Idx " << tree->getCellGroup(sd.level, localId)->getStartingIndex() <<
                     " and dest is " << sd.dest << " tag " << getTag(sd.level, tree->getCellGroup(sd.level, localId)->getStartingIndex(), nbBlocksBeforeMinPerLevel[sd.level] + localId, 0, sd.dest) << "\n");
                FLOG(FLog::Controller << "[SMpi] " << sd.level << " Post a send during M2L for Idx " << tree->getCellGroup(sd.level, localId)->getStartingIndex() <<
                     " and dest is " << sd.dest << " tag " << getTag(sd.level, tree->getCellGroup(sd.level, localId)->getStartingIndex(), nbBlocksBeforeMinPerLevel[sd.level] + localId, 1, sd.dest) << "\n");

                starpu_mpi_isend_detached( cellHandles[sd.level][localId].symb, sd.dest,
                        getTag(sd.level, tree->getCellGroup(sd.level, localId)->getStartingIndex(), nbBlocksBeforeMinPerLevel[sd.level] + localId, 0, sd.dest),
                        comm.getComm(), 0/*callback*/, 0/*arg*/ );
                starpu_mpi_isend_detached( cellHandles[sd.level][localId].up, sd.dest,
                        getTag(sd.level, tree->getCellGroup(sd.level, localId)->getStartingIndex(), nbBlocksBeforeMinPerLevel[sd.level] + localId, 1, sd.dest),
                        comm.getComm(), 0/*callback*/, 0/*arg*/ );
            }
        }
        FLOG(FLog::Controller.flush() );
    }

    void cleanHandleMpi(){
        for(int idxLevel = 0 ; idxLevel < int(remoteCellGroups.size()) ; ++idxLevel){
            for(int idxHandle = 0 ; idxHandle < int(remoteCellGroups[idxLevel].size()) ; ++idxHandle){
                if(remoteCellGroups[idxLevel][idxHandle].ptrSymb){
                    starpu_data_unregister(remoteCellGroups[idxLevel][idxHandle].handleSymb);
                    FAlignedMemory::DeallocBytes(remoteCellGroups[idxLevel][idxHandle].ptrSymb);

                    if(remoteCellGroups[idxLevel][idxHandle].ptrUp){
                        starpu_data_unregister(remoteCellGroups[idxLevel][idxHandle].handleUp);
                        FAlignedMemory::DeallocBytes(remoteCellGroups[idxLevel][idxHandle].ptrUp);
                    }

                    if(remoteCellGroups[idxLevel][idxHandle].ptrDown){
                        starpu_data_unregister(remoteCellGroups[idxLevel][idxHandle].handleDown);
                        FAlignedMemory::DeallocBytes(remoteCellGroups[idxLevel][idxHandle].ptrDown);
                    }
                }
            }
            remoteCellGroups[idxLevel].clear();
        }
        {
            for(int idxHandle = 0 ; idxHandle < int(remoteParticleGroupss.size()) ; ++idxHandle){
                if(remoteParticleGroupss[idxHandle].ptrSymb){
                    starpu_data_unregister(remoteParticleGroupss[idxHandle].handleSymb);
                    FAlignedMemory::DeallocBytes(remoteParticleGroupss[idxHandle].ptrSymb);
                }
            }
            remoteParticleGroupss.clear();
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    /** Reset the handles array and create new ones to define
     * in a starpu way each block of data
     */
    void buildHandles(){
        cleanHandle();

        for(int idxLevel = 2 ; idxLevel < tree->getHeight() ; ++idxLevel){
            cellHandles[idxLevel].resize(tree->getNbCellGroupAtLevel(idxLevel));

            for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
                const CellContainerClass* currentCells = tree->getCellGroup(idxLevel, idxGroup);
                starpu_variable_data_register(&cellHandles[idxLevel][idxGroup].symb, 0,
                                              (uintptr_t)currentCells->getRawBuffer(), currentCells->getBufferSizeInByte());
                starpu_variable_data_register(&cellHandles[idxLevel][idxGroup].up, 0,
                                              (uintptr_t)currentCells->getRawMultipoleBuffer(), currentCells->getMultipoleBufferSizeInByte());
                starpu_variable_data_register(&cellHandles[idxLevel][idxGroup].down, 0,
                                              (uintptr_t)currentCells->getRawLocalBuffer(), currentCells->getLocalBufferSizeInByte());
                cellHandles[idxLevel][idxGroup].intervalSize = int(currentCells->getNumberOfCellsInBlock());
#ifdef STARPU_SUPPORT_ARBITER
                starpu_data_assign_arbiter(cellHandles[idxLevel][idxGroup].up, arbiterGlobal);
                starpu_data_assign_arbiter(cellHandles[idxLevel][idxGroup].down, arbiterGlobal);
#endif
            }
        }
        {
            particleHandles.resize(tree->getNbParticleGroup());
            for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
                ParticleGroupClass* containers = tree->getParticleGroup(idxGroup);
                starpu_variable_data_register(&particleHandles[idxGroup].symb, 0,
                                              (uintptr_t)containers->getRawBuffer(), containers->getBufferSizeInByte());
                starpu_variable_data_register(&particleHandles[idxGroup].down, 0,
                                              (uintptr_t)containers->getRawAttributesBuffer(), containers->getAttributesBufferSizeInByte());
#ifdef STARPU_USE_REDUX
                starpu_data_set_reduction_methods(particleHandles[idxGroup].down, &p2p_redux_perform,
                                                  &p2p_redux_init);
#else
#ifdef STARPU_SUPPORT_ARBITER
                starpu_data_assign_arbiter(particleHandles[idxGroup].down, arbiterGlobal);
#endif // STARPU_SUPPORT_ARBITER
#endif // STARPU_USE_REDUX
                particleHandles[idxGroup].intervalSize = int(containers->getNumberOfLeavesInBlock());
            }
        }
    }

    /**
     * This function is creating the interactions vector between blocks.
     * It fills externalInteractionsAllLevel and externalInteractionsLeafLevel.
     * Warning, the omp task for now are using the class attributes!
     *
     */
    void buildExternalInteractionVecs(){
        FLOG( FTic timer; FTic leafTimer; FTic cellTimer; );
        // Reset interactions
        externalInteractionsAllLevel.clear();
        externalInteractionsLeafLevel.clear();
        // One per level + leaf level
        externalInteractionsAllLevel.resize(tree->getHeight());

        // First leaf level
        {
            // We create one big vector per block
            externalInteractionsLeafLevel.resize(tree->getNbParticleGroup());

            for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
                // Create the vector
                ParticleGroupClass* containers = tree->getParticleGroup(idxGroup);

                std::vector<BlockInteractions<ParticleGroupClass>>* externalInteractions = &externalInteractionsLeafLevel[idxGroup];

#pragma omp task default(none) firstprivate(idxGroup, containers, externalInteractions)
                { // Can be a task(inout:iterCells)
                    std::vector<OutOfBlockInteraction> outsideInteractions;
                    const MortonIndex blockStartIdx = containers->getStartingIndex();
                    const MortonIndex blockEndIdx   = containers->getEndingIndex();

                    for(int leafIdx = 0 ; leafIdx < containers->getNumberOfLeavesInBlock() ; ++leafIdx){
                        const MortonIndex mindex = containers->getLeafMortonIndex(leafIdx);
                        // ParticleContainerClass particles = containers->template getLeaf<ParticleContainerClass>(leafIdx);

                        MortonIndex interactionsIndexes[26];
                        int interactionsPosition[26];
                        FTreeCoordinate coord(mindex);
                        int counter = coord.getNeighborsIndexes(tree->getHeight(),interactionsIndexes,interactionsPosition);

                        for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
                            if( blockStartIdx <= interactionsIndexes[idxInter] && interactionsIndexes[idxInter] < blockEndIdx ){
                                // Inside block interaction, do nothing
                            }
                            else if(interactionsIndexes[idxInter] < mindex){
                                OutOfBlockInteraction property;
                                property.insideIndex = mindex;
                                property.outIndex    = interactionsIndexes[idxInter];
                                property.relativeOutPosition = interactionsPosition[idxInter];
                                property.insideIdxInBlock = leafIdx;
                                property.outsideIdxInBlock = -1;
                                outsideInteractions.push_back(property);
                            }
                        }
                    }

                    // Sort to match external order
                    FQuickSort<OutOfBlockInteraction, int>::QsSequential(outsideInteractions.data(),int(outsideInteractions.size()));

                    int currentOutInteraction = 0;
                    for(int idxLeftGroup = 0 ; idxLeftGroup < idxGroup && currentOutInteraction < int(outsideInteractions.size()) ; ++idxLeftGroup){
                        ParticleGroupClass* leftContainers = tree->getParticleGroup(idxLeftGroup);
                        const MortonIndex blockStartIdxOther    = leftContainers->getStartingIndex();
                        const MortonIndex blockEndIdxOther      = leftContainers->getEndingIndex();

                        while(currentOutInteraction < int(outsideInteractions.size())
                              && (outsideInteractions[currentOutInteraction].outIndex < blockStartIdxOther
                                  || leftContainers->getLeafIndex(outsideInteractions[currentOutInteraction].outIndex) == -1)
                              && outsideInteractions[currentOutInteraction].outIndex < blockEndIdxOther){
                            currentOutInteraction += 1;
                        }

                        int lastOutInteraction = currentOutInteraction;
                        int copyExistingInteraction = currentOutInteraction;
                        while(lastOutInteraction < int(outsideInteractions.size()) && outsideInteractions[lastOutInteraction].outIndex < blockEndIdxOther){
                            const int leafPos = leftContainers->getLeafIndex(outsideInteractions[lastOutInteraction].outIndex);
                            if(leafPos != -1){
                                if(copyExistingInteraction != lastOutInteraction){
                                    outsideInteractions[copyExistingInteraction] = outsideInteractions[lastOutInteraction];
                                }
                                outsideInteractions[copyExistingInteraction].outsideIdxInBlock = leafPos;
                                copyExistingInteraction += 1;
                            }
                            lastOutInteraction += 1;
                        }

                        const int nbInteractionsBetweenBlocks = (copyExistingInteraction-currentOutInteraction);
                        if(nbInteractionsBetweenBlocks){
                            externalInteractions->emplace_back();
                            BlockInteractions<ParticleGroupClass>* interactions = &externalInteractions->back();
                            interactions->otherBlock = leftContainers;
                            interactions->otherBlockId = idxLeftGroup;
                            interactions->interactions.resize(nbInteractionsBetweenBlocks);
                            std::copy(outsideInteractions.begin() + currentOutInteraction,
                                      outsideInteractions.begin() + copyExistingInteraction,
                                      interactions->interactions.begin());
                        }

                        currentOutInteraction = lastOutInteraction;
                    }
                }
            }
        }
        FLOG( leafTimer.tac(); );
        FLOG( cellTimer.tic(); );
        {
            for(int idxLevel = tree->getHeight()-1 ; idxLevel >= 2 ; --idxLevel){
                externalInteractionsAllLevel[idxLevel].resize(tree->getNbCellGroupAtLevel(idxLevel));

                for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
                    CellContainerClass* currentCells = tree->getCellGroup(idxLevel, idxGroup);

                    std::vector<BlockInteractions<CellContainerClass>>* externalInteractions = &externalInteractionsAllLevel[idxLevel][idxGroup];

#pragma omp task default(none) firstprivate(idxGroup, currentCells, idxLevel, externalInteractions)
                    {
                        std::vector<OutOfBlockInteraction> outsideInteractions;
                        const MortonIndex blockStartIdx = currentCells->getStartingIndex();
                        const MortonIndex blockEndIdx   = currentCells->getEndingIndex();

                        for(int cellIdx = 0 ; cellIdx < currentCells->getNumberOfCellsInBlock() ; ++cellIdx){
                            const MortonIndex mindex = currentCells->getCellMortonIndex(cellIdx);

                            MortonIndex interactionsIndexes[189];
                            int interactionsPosition[189];
                            const FTreeCoordinate coord(mindex);
                            int counter = coord.getInteractionNeighbors(idxLevel,interactionsIndexes,interactionsPosition);

                            for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
                                if( blockStartIdx <= interactionsIndexes[idxInter] && interactionsIndexes[idxInter] < blockEndIdx ){
                                    // Nothing to do
                                }
                                else if(interactionsIndexes[idxInter] < mindex){
                                    OutOfBlockInteraction property;
                                    property.insideIndex = mindex;
                                    property.outIndex    = interactionsIndexes[idxInter];
                                    property.relativeOutPosition = interactionsPosition[idxInter];
                                    property.insideIdxInBlock = cellIdx;
                                    property.outsideIdxInBlock = -1;
                                    outsideInteractions.push_back(property);
                                }
                            }
                        }

                        // Manage outofblock interaction
                        FQuickSort<OutOfBlockInteraction, int>::QsSequential(outsideInteractions.data(),int(outsideInteractions.size()));

                        int currentOutInteraction = 0;
                        for(int idxLeftGroup = 0 ; idxLeftGroup < idxGroup && currentOutInteraction < int(outsideInteractions.size()) ; ++idxLeftGroup){
                            CellContainerClass* leftCells   = tree->getCellGroup(idxLevel, idxLeftGroup);
                            const MortonIndex blockStartIdxOther = leftCells->getStartingIndex();
                            const MortonIndex blockEndIdxOther   = leftCells->getEndingIndex();

                            while(currentOutInteraction < int(outsideInteractions.size())
                                  && (outsideInteractions[currentOutInteraction].outIndex < blockStartIdxOther
                                      || leftCells->getCellIndex(outsideInteractions[currentOutInteraction].outIndex) == -1)
                                  && outsideInteractions[currentOutInteraction].outIndex < blockEndIdxOther){
                                currentOutInteraction += 1;
                            }

                            int lastOutInteraction = currentOutInteraction;
                            int copyExistingInteraction = currentOutInteraction;
                            while(lastOutInteraction < int(outsideInteractions.size()) && outsideInteractions[lastOutInteraction].outIndex < blockEndIdxOther){
                                const int cellPos = leftCells->getCellIndex(outsideInteractions[lastOutInteraction].outIndex);
                                if(cellPos != -1){
                                    if(copyExistingInteraction != lastOutInteraction){
                                        outsideInteractions[copyExistingInteraction] = outsideInteractions[lastOutInteraction];
                                    }
                                    outsideInteractions[copyExistingInteraction].outsideIdxInBlock = cellPos;
                                    copyExistingInteraction += 1;
                                }
                                lastOutInteraction += 1;
                            }

                            // Create interactions
                            const int nbInteractionsBetweenBlocks = (copyExistingInteraction-currentOutInteraction);
                            if(nbInteractionsBetweenBlocks){
                                externalInteractions->emplace_back();
                                BlockInteractions<CellContainerClass>* interactions = &externalInteractions->back();
                                interactions->otherBlock = leftCells;
                                interactions->otherBlockId = idxLeftGroup;
                                interactions->interactions.resize(nbInteractionsBetweenBlocks);
                                std::copy(outsideInteractions.begin() + currentOutInteraction,
                                          outsideInteractions.begin() + copyExistingInteraction,
                                          interactions->interactions.begin());
                            }

                            currentOutInteraction = lastOutInteraction;
                        }
                    }
                }
            }
        }
        FLOG( cellTimer.tac(); );

#pragma omp taskwait

        FLOG( FLog::Controller << "\t\t Prepare in " << timer.tacAndElapsed() << "s\n" );
        FLOG( FLog::Controller << "\t\t\t Prepare at leaf level in   " << leafTimer.elapsed() << "s\n" );
        FLOG( FLog::Controller << "\t\t\t Prepare at other levels in " << cellTimer.elapsed() << "s\n" );
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Bottom Pass
    /////////////////////////////////////////////////////////////////////////////////////

    void bottomPass(){
        FLOG( FTic timer; );

        FAssertLF(cellHandles[tree->getHeight()-1].size() == particleHandles.size());

        for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
            starpu_insert_task(&p2m_cl,
                               STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                               STARPU_VALUE, &cellHandles[tree->getHeight()-1][idxGroup].intervalSize, sizeof(int),
        #ifdef SCALFMM_STARPU_USE_PRIO
                    STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2M(),
        #endif
                    STARPU_R, cellHandles[tree->getHeight()-1][idxGroup].symb,
                    STARPU_RW, cellHandles[tree->getHeight()-1][idxGroup].up,
                    STARPU_R, particleHandles[idxGroup].symb,
        #ifdef STARPU_USE_TASK_NAME
                    STARPU_NAME, p2mTaskNames.get(),
        #endif
                    0);
        }

        FLOG( FLog::Controller << "\t\t bottomPass in " << timer.tacAndElapsed() << "s\n" );
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Upward Pass
    /////////////////////////////////////////////////////////////////////////////////////

    void upwardPass(){
        FLOG( FTic timer; );
        for(int idxLevel = FMath::Min(tree->getHeight() - 2, FAbstractAlgorithm::lowerWorkingLevel - 1) ; idxLevel >= FAbstractAlgorithm::upperWorkingLevel ; --idxLevel){
            int idxSubGroup = 0;

            for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel)
                && idxSubGroup < tree->getNbCellGroupAtLevel(idxLevel+1) ; ++idxGroup){
                CellContainerClass*const currentCells = tree->getCellGroup(idxLevel, idxGroup);

                // Skip current group if needed
                if( tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex() <= (currentCells->getStartingIndex()<<3) ){
                    ++idxSubGroup;
                    FAssertLF( idxSubGroup != tree->getNbCellGroupAtLevel(idxLevel+1) );
                    FAssertLF( (tree->getCellGroup(idxLevel+1, idxSubGroup)->getStartingIndex()>>3) == currentCells->getStartingIndex() );
                }

                {

                    struct starpu_task* const task = starpu_task_create();
                    task->dyn_handles = (starpu_data_handle_t*)malloc(sizeof(starpu_data_handle_t)*20);
                    task->dyn_handles[0] = cellHandles[idxLevel][idxGroup].symb;
                    task->dyn_handles[1] = cellHandles[idxLevel][idxGroup].up;

                    task->dyn_handles[2] = cellHandles[idxLevel+1][idxSubGroup].symb;
                    task->dyn_handles[3] = cellHandles[idxLevel+1][idxSubGroup].up;

                    // put the right codelet
                    task->cl = &m2m_cl;
                    // put args values
                    char *arg_buffer;
                    size_t arg_buffer_size;
                    starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                                             STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                             STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                             STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                                             0);
                    task->cl_arg = arg_buffer;
                    task->cl_arg_size = arg_buffer_size;
                    task->cl_arg_free = 1;
#ifdef SCALFMM_STARPU_USE_PRIO
                    task->priority = PrioClass::Controller().getInsertionPosM2M(idxLevel);
#endif
#ifdef STARPU_USE_TASK_NAME
                    task->name = m2mTaskNames[idxLevel].get();
#endif
                    FAssertLF(starpu_task_submit(task) == 0);
                }

                while(tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex() <= (((currentCells->getEndingIndex()-1)<<3)+7)
                      && (idxSubGroup+1) != tree->getNbCellGroupAtLevel(idxLevel+1)
                      && tree->getCellGroup(idxLevel+1, idxSubGroup+1)->getStartingIndex() <= ((currentCells->getEndingIndex()-1)<<3)+7 ){
                    idxSubGroup += 1;

                    struct starpu_task* const task = starpu_task_create();
                    task->dyn_handles = (starpu_data_handle_t*)malloc(sizeof(starpu_data_handle_t)*20);
                    task->dyn_handles[0] = cellHandles[idxLevel][idxGroup].symb;
                    task->dyn_handles[1] = cellHandles[idxLevel][idxGroup].up;

                    task->dyn_handles[2] = cellHandles[idxLevel+1][idxSubGroup].symb;
                    task->dyn_handles[3] = cellHandles[idxLevel+1][idxSubGroup].up;

                    // put the right codelet
                    task->cl = &m2m_cl;
                    // put args values
                    char *arg_buffer;
                    size_t arg_buffer_size;
                    starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                                             STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                             STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                             STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                                             0);
                    task->cl_arg = arg_buffer;
                    task->cl_arg_size = arg_buffer_size;
                    task->cl_arg_free = 1;
#ifdef SCALFMM_STARPU_USE_PRIO
                    task->priority = PrioClass::Controller().getInsertionPosM2M(idxLevel);
#endif
#ifdef STARPU_USE_TASK_NAME
                    task->name = m2mTaskNames[idxLevel].get();
#endif
                    FAssertLF(starpu_task_submit(task) == 0);
                }

            }

            /////////////////////////////////////////////////////////////
            // Exchange for mpi
            /////////////////////////////////////////////////////////////
            // Manage the external operations
            // Find what to recv
            if(tree->getNbCellGroupAtLevel(idxLevel)){
                // Take last block at this level
                const CellContainerClass* currentCells = tree->getCellGroup(idxLevel, tree->getNbCellGroupAtLevel(idxLevel)-1);
                // Take the last cell index of the last block
                const MortonIndex myLastIdx = currentCells->getEndingIndex()-1;
                // Find the descriptor of the first block that belong to someone else at lower level
                const int firstOtherBlock = nbBlocksBeforeMinPerLevel[idxLevel+1] + tree->getNbCellGroupAtLevel(idxLevel+1);
                FAssertLF(processesBlockInfos[idxLevel+1][firstOtherBlock-1].owner == comm.processId());
                // Iterate while the block has our cell has parent
                int idxBlockToRecv = 0;
                while(firstOtherBlock + idxBlockToRecv < int(processesBlockInfos[idxLevel+1].size()) &&
                      myLastIdx == (processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].firstIndex >> 3)){

                    if(remoteCellGroups[idxLevel+1][firstOtherBlock + idxBlockToRecv].ptrSymb == nullptr){
                        const size_t nbBytesInBlockSymb = processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].bufferSizeSymb;
                        unsigned char* memoryBlockSymb = (unsigned char*)FAlignedMemory::AllocateBytes<32>(nbBytesInBlockSymb);
                        remoteCellGroups[idxLevel+1][firstOtherBlock + idxBlockToRecv].ptrSymb = memoryBlockSymb;
                        starpu_variable_data_register(&remoteCellGroups[idxLevel+1][firstOtherBlock + idxBlockToRecv].handleSymb, 0,
                                (uintptr_t)remoteCellGroups[idxLevel+1][firstOtherBlock + idxBlockToRecv].ptrSymb, nbBytesInBlockSymb);

                        const size_t nbBytesInBlockUp = processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].bufferSizeUp;
                        unsigned char* memoryBlockUp = (unsigned char*)FAlignedMemory::AllocateBytes<32>(nbBytesInBlockUp);
                        remoteCellGroups[idxLevel+1][firstOtherBlock + idxBlockToRecv].ptrUp = memoryBlockUp;
                        starpu_variable_data_register(&remoteCellGroups[idxLevel+1][firstOtherBlock + idxBlockToRecv].handleUp, 0,
                                (uintptr_t)remoteCellGroups[idxLevel+1][firstOtherBlock + idxBlockToRecv].ptrUp, nbBytesInBlockUp);
                    }

                    FLOG(FLog::Controller << "[SMpi] " << idxLevel << " Post a recv during M2M for Idx " << processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].firstIndex <<
                                                                                                                                                                         " and owner is " << processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].owner << " tag " << getTag(idxLevel, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].firstIndex, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].globalIdx, 0, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].owner) << "\n");
                    FLOG(FLog::Controller << "[SMpi] " << idxLevel << " Post a recv during M2M for Idx " << processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].firstIndex <<
                                                                                                                                                                         " and owner is " << processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].owner << " tag " << getTag(idxLevel, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].firstIndex, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].globalIdx, 1, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].owner) << "\n");
                    FLOG(FLog::Controller.flush());



                    starpu_mpi_irecv_detached ( remoteCellGroups[idxLevel+1][firstOtherBlock + idxBlockToRecv].handleSymb,
                            processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].owner,
                            getTag(idxLevel+1, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].firstIndex, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].globalIdx, 0, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].owner),
                            comm.getComm(), 0/*callback*/, 0/*arg*/ );
                    starpu_mpi_irecv_detached ( remoteCellGroups[idxLevel+1][firstOtherBlock + idxBlockToRecv].handleUp,
                            processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].owner,
                            getTag(idxLevel+1, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].firstIndex, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].globalIdx, 1, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToRecv].owner),
                            comm.getComm(), 0/*callback*/, 0/*arg*/ );


                    idxBlockToRecv += 1;
                }
                FAssertLF(idxBlockToRecv < 8);
                if(idxBlockToRecv){// Perform the work
                    // Copy at max 8 groups
                    int nbSubCellGroups = 0;
                    while(nbSubCellGroups < idxBlockToRecv){
                        struct starpu_task* const task = starpu_task_create();
                        task->dyn_handles = (starpu_data_handle_t*)malloc(sizeof(starpu_data_handle_t)*20);
                        task->dyn_handles[0] = cellHandles[idxLevel][tree->getNbCellGroupAtLevel(idxLevel)-1].symb;
                        task->dyn_handles[1] = cellHandles[idxLevel][tree->getNbCellGroupAtLevel(idxLevel)-1].up;

                        task->dyn_handles[2] = remoteCellGroups[idxLevel+1][firstOtherBlock + nbSubCellGroups].handleSymb;
                        task->dyn_handles[3] = remoteCellGroups[idxLevel+1][firstOtherBlock + nbSubCellGroups].handleUp;

                        // put the right codelet
                        task->cl = &m2m_cl;
                        // put args values
                        char *arg_buffer;
                        size_t arg_buffer_size;
                        starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                                                 STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                                 STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                                 STARPU_VALUE, &cellHandles[idxLevel][tree->getNbCellGroupAtLevel(idxLevel)-1].intervalSize, sizeof(int),
                                0);
                        task->cl_arg = arg_buffer;
                        task->cl_arg_size = arg_buffer_size;
#ifdef SCALFMM_STARPU_USE_PRIO
                        task->priority = PrioClass::Controller().getInsertionPosM2M(idxLevel);
#endif
    #ifdef STARPU_USE_TASK_NAME
                    task->name = m2mTaskNames[idxLevel].get();
    #endif
                        nbSubCellGroups += 1;
                        FAssertLF(starpu_task_submit(task) == 0);
                    }
                }
            }
            // Find what to send
            if(tree->getNbCellGroupAtLevel(idxLevel+1)
                    && nbBlocksBeforeMinPerLevel[idxLevel] != 0){
                // Take the first lower block
                const CellContainerClass* currentCells = tree->getCellGroup(idxLevel+1, 0);
                // Take its first index
                const MortonIndex myFirstChildIdx = currentCells->getStartingIndex();
                const MortonIndex missingParentIdx = (myFirstChildIdx>>3);
                // If no parent or the first parent is not the good one
                if(tree->getNbCellGroupAtLevel(idxLevel) == 0
                        || tree->getCellGroup(idxLevel, 0)->getStartingIndex() != missingParentIdx){
                    // Look if the parent is owned by another block
                    const int firstOtherBlock = nbBlocksBeforeMinPerLevel[idxLevel]-1;
                    FAssertLF(processesBlockInfos[idxLevel][firstOtherBlock].lastIndex-1 == missingParentIdx);
                    const int dest = processesBlockInfos[idxLevel][firstOtherBlock].owner;
                    int lowerIdxToSend = 0;
                    while(lowerIdxToSend != tree->getNbCellGroupAtLevel(idxLevel+1)
                          && missingParentIdx == (tree->getCellGroup(idxLevel+1, lowerIdxToSend)->getStartingIndex()>>3)){

                        FLOG(FLog::Controller << "[SMpi] " << idxLevel << " Post a send during M2M for Idx " << tree->getCellGroup(idxLevel+1, lowerIdxToSend)->getStartingIndex() <<
                             " and dest is " << dest << " tag " << getTag(idxLevel, tree->getCellGroup(idxLevel+1, lowerIdxToSend)->getStartingIndex(), nbBlocksBeforeMinPerLevel[idxLevel+1] + lowerIdxToSend, 0, dest) << "\n");
                        FLOG(FLog::Controller << "[SMpi] " << idxLevel << " Post a send during M2M for Idx " << tree->getCellGroup(idxLevel+1, lowerIdxToSend)->getStartingIndex() <<
                             " and dest is " << dest << " tag " << getTag(idxLevel, tree->getCellGroup(idxLevel+1, lowerIdxToSend)->getStartingIndex(), nbBlocksBeforeMinPerLevel[idxLevel+1] + lowerIdxToSend, 1, dest) << "\n");
                        FLOG(FLog::Controller.flush());

                        starpu_mpi_isend_detached( cellHandles[idxLevel+1][lowerIdxToSend].symb, dest,
                                getTag(idxLevel+1, tree->getCellGroup(idxLevel+1, lowerIdxToSend)->getStartingIndex(), nbBlocksBeforeMinPerLevel[idxLevel+1] + lowerIdxToSend, 0, dest),
                                comm.getComm(), 0/*callback*/, 0/*arg*/ );
                        starpu_mpi_isend_detached( cellHandles[idxLevel+1][lowerIdxToSend].up, dest,
                                getTag(idxLevel+1, tree->getCellGroup(idxLevel+1, lowerIdxToSend)->getStartingIndex(), nbBlocksBeforeMinPerLevel[idxLevel+1] + lowerIdxToSend, 1, dest),
                                comm.getComm(), 0/*callback*/, 0/*arg*/ );
                        lowerIdxToSend += 1;
                    }

                }
            }
            /////////////////////////////////////////////////////////////
        }
        FLOG( FLog::Controller << "\t\t upwardPass in " << timer.tacAndElapsed() << "s\n" );
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Transfer Pass Mpi
    /////////////////////////////////////////////////////////////////////////////////////

    void transferPassMpi(){
        FLOG( FTic timer; );
        for(int idxLevel = FAbstractAlgorithm::lowerWorkingLevel-1 ; idxLevel >= FAbstractAlgorithm::upperWorkingLevel ; --idxLevel){
            for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
                for(int idxInteraction = 0; idxInteraction < int(externalInteractionsAllLevelMpi[idxLevel][idxGroup].size()) ; ++idxInteraction){
                    const int interactionid = externalInteractionsAllLevelMpi[idxLevel][idxGroup][idxInteraction].otherBlockId;
                    const std::vector<OutOfBlockInteraction>* outsideInteractions = &externalInteractionsAllLevelMpi[idxLevel][idxGroup][idxInteraction].interactions;

                    starpu_insert_task(&m2l_cl_inout_mpi,
                                       STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                       STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                       STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                       STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                   #ifdef SCALFMM_STARPU_USE_PRIO
                                       STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                   #endif
                                       STARPU_R, cellHandles[idxLevel][idxGroup].symb,
                                       (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][idxGroup].down,
                                       STARPU_R, remoteCellGroups[idxLevel][interactionid].handleSymb,
                                       STARPU_R, remoteCellGroups[idxLevel][interactionid].handleUp,
                   #ifdef STARPU_USE_TASK_NAME
                                       STARPU_NAME, m2lOuterTaskNames[idxLevel].get(),
                   #endif
                                       0);
                }
            }
        }
        FLOG( FLog::Controller << "\t\t transferPassMpi in " << timer.tacAndElapsed() << "s\n" );
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Transfer Pass
    /////////////////////////////////////////////////////////////////////////////////////

    void transferPass(const int fromLevel, const int toLevel, const bool inner, const bool outer){
        FLOG( FTic timer; );
        FLOG( FTic timerInBlock; FTic timerOutBlock; );
        for(int idxLevel = fromLevel ; idxLevel < toLevel ; ++idxLevel){
            if(inner){
                FLOG( timerInBlock.tic() );
                for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
                    starpu_insert_task(&m2l_cl_in,
                                       STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                       STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                       STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                   #ifdef SCALFMM_STARPU_USE_PRIO
                                       STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2L(idxLevel),
                   #endif
                                       STARPU_R, cellHandles[idxLevel][idxGroup].symb,
                                       STARPU_R, cellHandles[idxLevel][idxGroup].up,
                                       (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][idxGroup].down,
                   #ifdef STARPU_USE_TASK_NAME
                                       STARPU_NAME, m2lTaskNames[idxLevel].get(),
                   #endif
                                       0);
                }
                FLOG( timerInBlock.tac() );
            }
            if(outer){
                FLOG( timerOutBlock.tic() );

                for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
                    for(int idxInteraction = 0; idxInteraction < int(externalInteractionsAllLevel[idxLevel][idxGroup].size()) ; ++idxInteraction){
                        const int interactionid = externalInteractionsAllLevel[idxLevel][idxGroup][idxInteraction].otherBlockId;
                        const std::vector<OutOfBlockInteraction>* outsideInteractions = &externalInteractionsAllLevel[idxLevel][idxGroup][idxInteraction].interactions;

                        int mode = 1;
                        starpu_insert_task(&m2l_cl_inout,
                                           STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                           STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                           STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                           STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                                           STARPU_VALUE, &mode, sizeof(int),
                   #ifdef SCALFMM_STARPU_USE_PRIO
                                           STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                   #endif
                                           STARPU_R, cellHandles[idxLevel][idxGroup].symb,
                                           (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][idxGroup].down,
                                           STARPU_R, cellHandles[idxLevel][interactionid].symb,
                                           STARPU_R, cellHandles[idxLevel][interactionid].up,
                   #ifdef STARPU_USE_TASK_NAME
                                           STARPU_NAME, m2lOuterTaskNames[idxLevel].get(),
                   #endif
                                           0);

                        mode = 2;
                        starpu_insert_task(&m2l_cl_inout,
                                           STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                           STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                           STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                           STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                                           STARPU_VALUE, &mode, sizeof(int),
                   #ifdef SCALFMM_STARPU_USE_PRIO
                                           STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                   #endif
                                           STARPU_R, cellHandles[idxLevel][interactionid].symb,
                                           (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][interactionid].down,
                                           STARPU_R, cellHandles[idxLevel][idxGroup].symb,
                                           STARPU_R, cellHandles[idxLevel][idxGroup].up,
                   #ifdef STARPU_USE_TASK_NAME
                                           STARPU_NAME, m2lOuterTaskNames[idxLevel].get(),
                   #endif
                                           0);
                    }
                }
                FLOG( timerOutBlock.tac() );
            }
        }
        FLOG( FLog::Controller << "\t\t transferPass in " << timer.tacAndElapsed() << "s\n" );
        FLOG( FLog::Controller << "\t\t\t inblock in  " << timerInBlock.elapsed() << "s\n" );
        FLOG( FLog::Controller << "\t\t\t outblock in " << timerOutBlock.elapsed() << "s\n" );
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Downard Pass
    /////////////////////////////////////////////////////////////////////////////////////

    void downardPass(){
        FLOG( FTic timer; );
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel - 1 ; ++idxLevel){
            /////////////////////////////////////////////////////////////
            // Exchange for MPI
            /////////////////////////////////////////////////////////////
            // Manage the external operations
            // Find what to send
            if(tree->getNbCellGroupAtLevel(idxLevel)){
                // Take last block at this level
                const int idxLastBlock = tree->getNbCellGroupAtLevel(idxLevel)-1;
                const CellContainerClass* currentCells = tree->getCellGroup(idxLevel, idxLastBlock);
                // Take the last cell index of the last block
                const MortonIndex myLastIdx = currentCells->getEndingIndex()-1;
                // Find the descriptor of the first block that belong to someone else at lower level
                const int firstOtherBlock = nbBlocksBeforeMinPerLevel[idxLevel+1] + tree->getNbCellGroupAtLevel(idxLevel+1);
                FAssertLF(processesBlockInfos[idxLevel+1][firstOtherBlock-1].owner == comm.processId());
                // Iterate while the block has our cell has parent
                int idxBlockToSend = 0;
                int lastProcSend   = 0;
                while(firstOtherBlock + idxBlockToSend < int(processesBlockInfos[idxLevel+1].size()) &&
                      myLastIdx == (processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].firstIndex >> 3)){

                    if(lastProcSend != processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].owner){

                        FLOG(FLog::Controller << "[SMpi] " << idxLevel << " Post a send during L2L for Idx " << tree->getCellGroup(idxLevel, idxLastBlock)->getStartingIndex() <<
                             " and dest is " << processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].owner
                                << " size " << tree->getCellGroup(idxLevel, idxLastBlock)->getBufferSizeInByte()
                                << " tag " << getTag(idxLevel, tree->getCellGroup(idxLevel, idxLastBlock)->getStartingIndex(), nbBlocksBeforeMinPerLevel[idxLevel] + idxLastBlock, 0, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].owner) << "\n");
                        FLOG(FLog::Controller << "[SMpi] " << idxLevel << " Post a send during L2L for Idx " << tree->getCellGroup(idxLevel, idxLastBlock)->getStartingIndex() <<
                             " and dest is " << processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].owner
                                << " size " << tree->getCellGroup(idxLevel, idxLastBlock)->getLocalBufferSizeInByte()
                                << " tag " << getTag(idxLevel, tree->getCellGroup(idxLevel, idxLastBlock)->getStartingIndex(), nbBlocksBeforeMinPerLevel[idxLevel] + idxLastBlock, 2, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].owner) << "\n");

                        starpu_mpi_isend_detached( cellHandles[idxLevel][idxLastBlock].symb,
                                                   processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].owner,
                                getTag(idxLevel, tree->getCellGroup(idxLevel, idxLastBlock)->getStartingIndex(), nbBlocksBeforeMinPerLevel[idxLevel] + idxLastBlock, 0, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].owner),
                                comm.getComm(), 0/*callback*/, 0/*arg*/ );
                        starpu_mpi_isend_detached( cellHandles[idxLevel][idxLastBlock].down,
                                                   processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].owner,
                                getTag(idxLevel, tree->getCellGroup(idxLevel, idxLastBlock)->getStartingIndex(), nbBlocksBeforeMinPerLevel[idxLevel] + idxLastBlock, 2, processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].owner),
                                comm.getComm(), 0/*callback*/, 0/*arg*/ );

                        lastProcSend = processesBlockInfos[idxLevel+1][firstOtherBlock + idxBlockToSend].owner;
                    }
                    idxBlockToSend += 1;
                }
            }
            // Find what to recv
            if(tree->getNbCellGroupAtLevel(idxLevel+1)
                    && nbBlocksBeforeMinPerLevel[idxLevel] != 0){
                // Take the first lower block
                const CellContainerClass* currentCells = tree->getCellGroup(idxLevel+1, 0);
                // Take its first index
                const MortonIndex myFirstChildIdx = currentCells->getStartingIndex();
                const MortonIndex missingParentIdx = (myFirstChildIdx>>3);
                // If no parent or the first parent is not the good one
                if(tree->getNbCellGroupAtLevel(idxLevel) == 0
                        || tree->getCellGroup(idxLevel, 0)->getStartingIndex() != missingParentIdx){

                    // Look if the parent is owned by another block
                    const int firstOtherBlock = nbBlocksBeforeMinPerLevel[idxLevel]-1;
                    FAssertLF(processesBlockInfos[idxLevel][firstOtherBlock].lastIndex-1 == missingParentIdx);

                    if(remoteCellGroups[idxLevel][firstOtherBlock].ptrSymb == nullptr){
                        const size_t nbBytesInBlock = processesBlockInfos[idxLevel][firstOtherBlock].bufferSizeSymb;
                        unsigned char* memoryBlock = (unsigned char*)FAlignedMemory::AllocateBytes<32>(nbBytesInBlock);
                        remoteCellGroups[idxLevel][firstOtherBlock].ptrSymb = memoryBlock;
                        starpu_variable_data_register(&remoteCellGroups[idxLevel][firstOtherBlock].handleSymb, 0,
                                                      (uintptr_t)remoteCellGroups[idxLevel][firstOtherBlock].ptrSymb, nbBytesInBlock);
                    }
                    if(remoteCellGroups[idxLevel][firstOtherBlock].ptrDown == nullptr){
                        const size_t nbBytesInBlock = processesBlockInfos[idxLevel][firstOtherBlock].bufferSizeDown;
                        unsigned char* memoryBlock = (unsigned char*)FAlignedMemory::AllocateBytes<32>(nbBytesInBlock);
                        remoteCellGroups[idxLevel][firstOtherBlock].ptrDown = memoryBlock;
                        starpu_variable_data_register(&remoteCellGroups[idxLevel][firstOtherBlock].handleDown, 0,
                                                      (uintptr_t)remoteCellGroups[idxLevel][firstOtherBlock].ptrDown, nbBytesInBlock);
                    }

                    FLOG(FLog::Controller << "[SMpi] " << idxLevel << " Post a recv during L2L for Idx " << processesBlockInfos[idxLevel][firstOtherBlock].firstIndex <<
                         " and owner " << processesBlockInfos[idxLevel][firstOtherBlock].owner
                         << " size " << processesBlockInfos[idxLevel][firstOtherBlock].bufferSizeSymb
                         << " tag " << getTag(idxLevel, processesBlockInfos[idxLevel][firstOtherBlock].firstIndex, processesBlockInfos[idxLevel][firstOtherBlock].globalIdx, 0, processesBlockInfos[idxLevel][firstOtherBlock].owner) << "\n");
                    FLOG(FLog::Controller << "[SMpi] " << idxLevel << " Post a recv during L2L for Idx " << processesBlockInfos[idxLevel][firstOtherBlock].firstIndex <<
                         " and owner " << processesBlockInfos[idxLevel][firstOtherBlock].owner
                         << " size " << processesBlockInfos[idxLevel][firstOtherBlock].bufferSizeDown
                         << " tag " << getTag(idxLevel, processesBlockInfos[idxLevel][firstOtherBlock].firstIndex, processesBlockInfos[idxLevel][firstOtherBlock].globalIdx, 2, processesBlockInfos[idxLevel][firstOtherBlock].owner) << "\n");

                    starpu_mpi_irecv_detached ( remoteCellGroups[idxLevel][firstOtherBlock].handleSymb,
                                                processesBlockInfos[idxLevel][firstOtherBlock].owner,
                                                getTag(idxLevel, processesBlockInfos[idxLevel][firstOtherBlock].firstIndex, processesBlockInfos[idxLevel][firstOtherBlock].globalIdx, 0, processesBlockInfos[idxLevel][firstOtherBlock].owner),
                                                comm.getComm(), 0/*callback*/, 0/*arg*/ );
                    starpu_mpi_irecv_detached ( remoteCellGroups[idxLevel][firstOtherBlock].handleDown,
                                                processesBlockInfos[idxLevel][firstOtherBlock].owner,
                                                getTag(idxLevel, processesBlockInfos[idxLevel][firstOtherBlock].firstIndex, processesBlockInfos[idxLevel][firstOtherBlock].globalIdx, 2, processesBlockInfos[idxLevel][firstOtherBlock].owner),
                                                comm.getComm(), 0/*callback*/, 0/*arg*/ );

                    {
                        const MortonIndex parentStartingIdx = processesBlockInfos[idxLevel][firstOtherBlock].firstIndex;
                        const MortonIndex parentEndingIdx = processesBlockInfos[idxLevel][firstOtherBlock].lastIndex;

                        int idxSubGroup = 0;
                        // Skip current group if needed
                        if( tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex() <= (parentStartingIdx<<3) ){
                            ++idxSubGroup;
                            FAssertLF( idxSubGroup != tree->getNbCellGroupAtLevel(idxLevel+1) );
                            FAssertLF( (tree->getCellGroup(idxLevel+1, idxSubGroup)->getStartingIndex()>>3) == parentStartingIdx );
                        }
                        // Copy at max 8 groups
                        int nbSubCellGroups = 0;
                        {
                            struct starpu_task* const task = starpu_task_create();
                            task->dyn_handles = (starpu_data_handle_t*)malloc(sizeof(starpu_data_handle_t)*20);
                            task->dyn_handles[0] = remoteCellGroups[idxLevel][firstOtherBlock].handleSymb;
                            task->dyn_handles[1] = remoteCellGroups[idxLevel][firstOtherBlock].handleDown;
                            task->dyn_handles[2] = cellHandles[idxLevel+1][idxSubGroup].symb;
                            task->dyn_handles[3] = cellHandles[idxLevel+1][idxSubGroup].down;

                            // put the right codelet
                            task->cl = &l2l_cl;
                            // put args values
                            char *arg_buffer;
                            size_t arg_buffer_size;
                            starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                                                     STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                                     STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                                     STARPU_VALUE, &remoteCellGroups[idxLevel][firstOtherBlock].intervalSize, sizeof(int),// TODO !
                                                     0);
                            task->cl_arg = arg_buffer;
                            task->cl_arg_size = arg_buffer_size;
#ifdef SCALFMM_STARPU_USE_PRIO
                            task->priority = PrioClass::Controller().getInsertionPosL2L(idxLevel);
#endif
    #ifdef STARPU_USE_TASK_NAME
							task->name = l2lTaskNames[idxLevel].get();
    #endif
                            FAssertLF(starpu_task_submit(task) == 0);
                        }

                        nbSubCellGroups += 1;
                        while(tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex() <= ((parentEndingIdx<<3)+7)
                              && (idxSubGroup + 1) != tree->getNbCellGroupAtLevel(idxLevel+1)
                              && tree->getCellGroup(idxLevel+1, idxSubGroup+1)->getStartingIndex() <= (parentEndingIdx<<3)+7 ){
                            idxSubGroup += 1;
                            struct starpu_task* const task = starpu_task_create();
                            task->dyn_handles = (starpu_data_handle_t*)malloc(sizeof(starpu_data_handle_t)*20);
                            task->dyn_handles[0] = remoteCellGroups[idxLevel][firstOtherBlock].handleSymb;
                            task->dyn_handles[1] = remoteCellGroups[idxLevel][firstOtherBlock].handleDown;
                            task->dyn_handles[2] = cellHandles[idxLevel+1][idxSubGroup].symb;
                            task->dyn_handles[3] = cellHandles[idxLevel+1][idxSubGroup].down;

                            // put the right codelet
                            task->cl = &l2l_cl;
                            // put args values
                            char *arg_buffer;
                            size_t arg_buffer_size;
                            starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                                                     STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                                     STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                                     STARPU_VALUE, &remoteCellGroups[idxLevel][firstOtherBlock].intervalSize, sizeof(int),// TODO !
                                                     0);
                            task->cl_arg = arg_buffer;
                            task->cl_arg_size = arg_buffer_size;
							task->cl_arg_free = 1;
#ifdef SCALFMM_STARPU_USE_PRIO
                            task->priority = PrioClass::Controller().getInsertionPosL2L(idxLevel);
#endif
    #ifdef STARPU_USE_TASK_NAME
							task->name = l2lTaskNames[idxLevel].get();
    #endif
                            FAssertLF(starpu_task_submit(task) == 0);

                            nbSubCellGroups += 1;
                            FAssertLF( nbSubCellGroups <= 9 );
                        }
                    }
                }
            }
            /////////////////////////////////////////////////////////////



            int idxSubGroup = 0;

            for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel)
                && idxSubGroup < tree->getNbCellGroupAtLevel(idxLevel+1) ; ++idxGroup){
                CellContainerClass*const currentCells = tree->getCellGroup(idxLevel, idxGroup);

                // Skip current group if needed
                if( tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex() <= (currentCells->getStartingIndex()<<3) ){
                    ++idxSubGroup;
                    FAssertLF( idxSubGroup != tree->getNbCellGroupAtLevel(idxLevel+1) );
                    FAssertLF( (tree->getCellGroup(idxLevel+1, idxSubGroup)->getStartingIndex()>>3) == currentCells->getStartingIndex() );
                }
                // Copy at max 8 groups
                {
                    struct starpu_task* const task = starpu_task_create();
                    task->dyn_handles = (starpu_data_handle_t*)malloc(sizeof(starpu_data_handle_t)*20);
                    task->dyn_handles[0] = cellHandles[idxLevel][idxGroup].symb;
                    task->dyn_handles[1] = cellHandles[idxLevel][idxGroup].down;

                    task->dyn_handles[2] = cellHandles[idxLevel+1][idxSubGroup].symb;
                    task->dyn_handles[3] = cellHandles[idxLevel+1][idxSubGroup].down;

                    // put the right codelet
                    if((noCommuteAtLastLevel && (idxLevel == FAbstractAlgorithm::lowerWorkingLevel - 2)) || noCommuteBetweenLevel){
                        task->cl = &l2l_cl_nocommute;
                    }
                    else{
                        task->cl = &l2l_cl;
                    }
                    // put args values
                    char *arg_buffer;
                    size_t arg_buffer_size;
                    starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                                             STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                             STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                             STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                                             0);
                    task->cl_arg = arg_buffer;
                    task->cl_arg_size = arg_buffer_size;
                    task->cl_arg_free = 1;
#ifdef SCALFMM_STARPU_USE_PRIO
                    task->priority = PrioClass::Controller().getInsertionPosL2L(idxLevel);
#endif
#ifdef STARPU_USE_TASK_NAME
							task->name = l2lTaskNames[idxLevel].get();
#endif
                    FAssertLF(starpu_task_submit(task) == 0);
                }
                while(tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex() <= (((currentCells->getEndingIndex()-1)<<3)+7)
                      && (idxSubGroup+1) != tree->getNbCellGroupAtLevel(idxLevel+1)
                      && tree->getCellGroup(idxLevel+1, idxSubGroup+1)->getStartingIndex() <= ((currentCells->getEndingIndex()-1)<<3)+7 ){
                    idxSubGroup += 1;

                    struct starpu_task* const task = starpu_task_create();
                    task->dyn_handles = (starpu_data_handle_t*)malloc(sizeof(starpu_data_handle_t)*20);
                    task->dyn_handles[0] = cellHandles[idxLevel][idxGroup].symb;
                    task->dyn_handles[1] = cellHandles[idxLevel][idxGroup].down;

                    task->dyn_handles[2] = cellHandles[idxLevel+1][idxSubGroup].symb;
                    task->dyn_handles[3] = cellHandles[idxLevel+1][idxSubGroup].down;

                    // put the right codelet
                    if((noCommuteAtLastLevel && (idxLevel == FAbstractAlgorithm::lowerWorkingLevel - 2)) || noCommuteBetweenLevel){
                        task->cl = &l2l_cl_nocommute;
                    }
                    else{
                        task->cl = &l2l_cl;
                    }
                    // put args values
                    char *arg_buffer;
                    size_t arg_buffer_size;
                    starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                                             STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                             STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                             STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                                             0);
                    task->cl_arg = arg_buffer;
                    task->cl_arg_size = arg_buffer_size;
                    task->cl_arg_free = 1;
#ifdef SCALFMM_STARPU_USE_PRIO
                    task->priority = PrioClass::Controller().getInsertionPosL2L(idxLevel);
#endif
#ifdef STARPU_USE_TASK_NAME
							task->name = l2lTaskNames[idxLevel].get();
#endif
                    FAssertLF(starpu_task_submit(task) == 0);
                }
            }
        }
        FLOG( FLog::Controller << "\t\t downardPass in " << timer.tacAndElapsed() << "s\n" );
    }
    /////////////////////////////////////////////////////////////////////////////////////
    /// Direct Pass MPI
    /////////////////////////////////////////////////////////////////////////////////////

    void directPassMpi(){
        FLOG( FTic timer; );

        for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
            for(int idxInteraction = 0; idxInteraction < int(externalInteractionsLeafLevelMpi[idxGroup].size()) ; ++idxInteraction){
                const int interactionid = externalInteractionsLeafLevelMpi[idxGroup][idxInteraction].otherBlockId;
                const std::vector<OutOfBlockInteraction>* outsideInteractions = &externalInteractionsLeafLevelMpi[idxGroup][idxInteraction].interactions;
                starpu_insert_task(&p2p_cl_inout_mpi,
                                   STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                   STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                   STARPU_VALUE, &particleHandles[idxGroup].intervalSize, sizeof(int),
                   #ifdef SCALFMM_STARPU_USE_PRIO
                                   STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2PExtern(),
                   #endif
                                   STARPU_R, particleHandles[idxGroup].symb,
                                   (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), particleHandles[idxGroup].down,
                                   STARPU_R, remoteParticleGroupss[interactionid].handleSymb,
                   #ifdef STARPU_USE_TASK_NAME
                                   STARPU_NAME, p2pOuterTaskNames.get(),
                   #endif
                                   0);
            }
        }

        FLOG( FLog::Controller << "\t\t directPass in MPI " << timer.tacAndElapsed() << "s\n" );
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /// Direct Pass
    /////////////////////////////////////////////////////////////////////////////////////

    void directPass(){
        FLOG( FTic timer; );
        FLOG( FTic timerInBlock; FTic timerOutBlock; );

        FLOG( timerOutBlock.tic() );
        for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
            for(int idxInteraction = 0; idxInteraction < int(externalInteractionsLeafLevel[idxGroup].size()) ; ++idxInteraction){
                const int interactionid = externalInteractionsLeafLevel[idxGroup][idxInteraction].otherBlockId;
                const std::vector<OutOfBlockInteraction>* outsideInteractions = &externalInteractionsLeafLevel[idxGroup][idxInteraction].interactions;
                starpu_insert_task(&p2p_cl_inout,
                                   STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                   STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                   STARPU_VALUE, &particleHandles[idxGroup].intervalSize, sizeof(int),
                   #ifdef SCALFMM_STARPU_USE_PRIO
                                   STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2PExtern(),
                   #endif
                                   STARPU_R, particleHandles[idxGroup].symb,
                   #ifdef STARPU_USE_REDUX
                                   STARPU_REDUX, particleHandles[idxGroup].down,
                   #else
                                   (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), particleHandles[idxGroup].down,
                   #endif
                                   STARPU_R, particleHandles[interactionid].symb,
                   #ifdef STARPU_USE_REDUX
                                   STARPU_REDUX, particleHandles[interactionid].down,
                   #else
                                   (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), particleHandles[interactionid].down,
                   #endif
                   #ifdef STARPU_USE_TASK_NAME
                                   STARPU_NAME, p2pOuterTaskNames.get(),
                   #endif
                                   0);
            }
        }
        FLOG( timerOutBlock.tac() );
        FLOG( timerInBlock.tic() );
        for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
            starpu_insert_task(&p2p_cl_in,
                               STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                               STARPU_VALUE, &particleHandles[idxGroup].intervalSize, sizeof(int),
                   #ifdef SCALFMM_STARPU_USE_PRIO
                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2P(),
                   #endif
                               STARPU_R, particleHandles[idxGroup].symb,
                   #ifdef STARPU_USE_REDUX
                               STARPU_REDUX, particleHandles[idxGroup].down,
                   #else
                               (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), particleHandles[idxGroup].down,
                   #endif
                   #ifdef STARPU_USE_TASK_NAME
                                    STARPU_NAME, p2pTaskNames.get(),
                   #endif
                               0);
        }
        FLOG( timerInBlock.tac() );

        FLOG( FLog::Controller << "\t\t directPass in " << timer.tacAndElapsed() << "s\n" );
        FLOG( FLog::Controller << "\t\t\t inblock  in " << timerInBlock.elapsed() << "s\n" );
        FLOG( FLog::Controller << "\t\t\t outblock in " << timerOutBlock.elapsed() << "s\n" );
    }
    /////////////////////////////////////////////////////////////////////////////////////
    /// Merge Pass
    /////////////////////////////////////////////////////////////////////////////////////

    void mergePass(){
        FLOG( FTic timer; );

        FAssertLF(cellHandles[tree->getHeight()-1].size() == particleHandles.size());

        for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
            starpu_insert_task(&l2p_cl,
                               STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                               STARPU_VALUE, &cellHandles[tree->getHeight()-1][idxGroup].intervalSize, sizeof(int),
        #ifdef SCALFMM_STARPU_USE_PRIO
                    STARPU_PRIORITY, PrioClass::Controller().getInsertionPosL2P(),
        #endif
                    STARPU_R, cellHandles[tree->getHeight()-1][idxGroup].symb,
                    STARPU_R, cellHandles[tree->getHeight()-1][idxGroup].down,
                    STARPU_R, particleHandles[idxGroup].symb,
        #ifdef STARPU_USE_REDUX
                    STARPU_REDUX, particleHandles[idxGroup].down,
        #else
                    (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), particleHandles[idxGroup].down,
        #endif
        #ifdef STARPU_USE_TASK_NAME
                    STARPU_NAME, l2pTaskNames.get(),
        #endif
                    0);
        }

        FLOG( FLog::Controller << "\t\t L2P in " << timer.tacAndElapsed() << "s\n" );
    }


#ifdef STARPU_USE_REDUX
    void readParticle(){
        FLOG( FTic timer; );

        FAssertLF(cellHandles[tree->getHeight()-1].size() == particleHandles.size());

        for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
            starpu_insert_task(&p2p_redux_read,
                   #ifdef SCALFMM_STARPU_USE_PRIO
                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosL2P(),
                   #endif
                               STARPU_R, particleHandles[idxGroup].down,
                   #ifdef STARPU_USE_TASK_NAME
                               STARPU_NAME, "read-particle",
                   #endif
                               0);
        }
    }
#endif
};

#endif // FGROUPTASKSTARPUMPIALGORITHM_HPP
