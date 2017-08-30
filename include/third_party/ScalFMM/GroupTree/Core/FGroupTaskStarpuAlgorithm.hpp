// Keep in private GIT
#ifndef FGROUPTASKSTARPUALGORITHM_HPP
#define FGROUPTASKSTARPUALGORITHM_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Core/FCoreCommon.hpp"
#include "../../Utils/FQuickSort.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../../Utils/FLog.hpp"
#include "../../Utils/FTic.hpp"
#include "../../Utils/FAssert.hpp"
#include "../../Utils/FEnv.hpp"

#include "FOutOfBlockInteraction.hpp"

#include <vector>
#include <memory>

#include <omp.h>

#include <starpu.h>
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
#endif

#ifdef SCALFMM_SIMGRID_TASKNAMEPARAMS
#include "../StarPUUtils/FStarPUTaskNameParams.hpp"
#endif

#include "Containers/FBoolArray.hpp"

template <class OctreeClass, class CellContainerClass, class KernelClass, class ParticleGroupClass, class StarPUCpuWrapperClass
          #ifdef SCALFMM_ENABLE_CUDA_KERNEL
          , class StarPUCudaWrapperClass = FStarPUCudaWrapper<KernelClass, FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                                              FCudaGroupOfParticles<int, 0, 0, int>, FCudaGroupAttachedLeaf<int, 0, 0, int>, FCudaEmptyKernel<int> >
          #endif
          #ifdef SCALFMM_ENABLE_OPENCL_KERNEL
          , class StarPUOpenClWrapperClass = FStarPUOpenClWrapper<KernelClass, FOpenCLDeviceWrapper<KernelClass>>
          #endif
          >
class FGroupTaskStarPUAlgorithm : public FAbstractAlgorithm {
protected:
    typedef FGroupTaskStarPUAlgorithm<OctreeClass, CellContainerClass, KernelClass, ParticleGroupClass, StarPUCpuWrapperClass
#ifdef SCALFMM_ENABLE_CUDA_KERNEL
    , StarPUCudaWrapperClass
#endif
#ifdef SCALFMM_ENABLE_OPENCL_KERNEL
    , StarPUOpenClWrapperClass
#endif
    > ThisClass;

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

    starpu_codelet p2p_cl_in;
    starpu_codelet p2p_cl_inout;

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
#ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
    std::vector<std::unique_ptr<char[]>> m2mTaskNames;
    std::vector<std::unique_ptr<char[]>> m2lTaskNames;
    std::vector<std::unique_ptr<char[]>> m2lOuterTaskNames;
    std::vector<std::unique_ptr<char[]>> l2lTaskNames;
    std::unique_ptr<char[]> p2mTaskNames;
    std::unique_ptr<char[]> l2pTaskNames;
    std::unique_ptr<char[]> p2pTaskNames;
    std::unique_ptr<char[]> p2pOuterTaskNames;
#else
    FStarPUTaskNameParams taskNames;
#endif
#endif
#ifdef SCALFMM_STARPU_USE_PRIO
    typedef FStarPUFmmPrioritiesV2 PrioClass;// FStarPUFmmPriorities
#endif

public:
    FGroupTaskStarPUAlgorithm(OctreeClass*const inTree, KernelClass* inKernels)
        : tree(inTree), originalCpuKernel(inKernels),
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
#ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
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

    ~FGroupTaskStarPUAlgorithm(){
        starpu_resume();

        cleanHandle();
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
        FLOG( FLog::Controller << "\tStart FGroupTaskStarPUAlgorithm\n" );
        const bool directOnly = (tree->getHeight() <= 2);

#ifdef STARPU_USE_CPU
        FTIME_TASKS(cpuWrapper.taskTimeRecorder.start());
#endif

        starpu_resume();
        FLOG( FTic timerSoumission; );

        if( operationsToProceed & FFmmP2P ) directPass();

        if(operationsToProceed & FFmmP2M && !directOnly) bottomPass();

        if(operationsToProceed & FFmmM2M && !directOnly) upwardPass();

        if(operationsToProceed & FFmmM2L && !directOnly) transferPass(FAbstractAlgorithm::upperWorkingLevel, FAbstractAlgorithm::lowerWorkingLevel-1 , true, true);

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
                        FTreeCoordinate coord(mindex, tree->getHeight()-1);
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
                            const FTreeCoordinate coord(mindex, idxLevel);
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
#ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                    STARPU_NAME, p2mTaskNames.get(),
#else
					//"P2M-nb_i_p"
                    STARPU_NAME, taskNames.print("P2M", "%d, %lld, %lld\n",
                                                 tree->getCellGroup(tree->getHeight()-1,idxGroup)->getNumberOfCellsInBlock(),
                                                 tree->getCellGroup(tree->getHeight()-1,idxGroup)->getSizeOfInterval(),
												 tree->getCellGroup(tree->getHeight()-1,idxGroup)->getNumberOfCellsInBlock()),
#endif
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

            for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
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
#ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                    task->name = m2mTaskNames[idxLevel].get();
#else
                    //"M2M-l_nb_i_nbc_ic_s"
                    task->name = taskNames.print("M2M", "%d, %d, %lld, %d, %lld, %lld\n",
                                                 idxLevel,
                                              tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                              tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                 tree->getCellGroup(idxLevel+1,idxSubGroup)->getNumberOfCellsInBlock(),
                                                 tree->getCellGroup(idxLevel+1,idxSubGroup)->getSizeOfInterval(),
                                                 FMath::Min(tree->getCellGroup(idxLevel,idxGroup)->getEndingIndex()-1, (tree->getCellGroup(idxLevel+1,idxSubGroup)->getEndingIndex()-1)>>3)-
                                                 FMath::Max(tree->getCellGroup(idxLevel,idxGroup)->getStartingIndex(), tree->getCellGroup(idxLevel+1,idxSubGroup)->getStartingIndex()>>3));
#endif
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
#ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                    task->name = m2mTaskNames[idxLevel].get();
#else
                    //M2M-l_nb_i_nbc_ic_s
                    task->name = taskNames.print("M2M", "%d, %d, %lld, %d, %lld, %lld\n",
                                                 idxLevel,
                                              tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                              tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                 tree->getCellGroup(idxLevel+1,idxSubGroup)->getNumberOfCellsInBlock(),
                                                 tree->getCellGroup(idxLevel+1,idxSubGroup)->getSizeOfInterval(),
                                                 FMath::Min(tree->getCellGroup(idxLevel,idxGroup)->getEndingIndex()-1, (tree->getCellGroup(idxLevel+1,idxSubGroup)->getEndingIndex()-1)>>3)-
                                                 FMath::Max(tree->getCellGroup(idxLevel,idxGroup)->getStartingIndex(), tree->getCellGroup(idxLevel+1,idxSubGroup)->getStartingIndex()>>3));
#endif
    #endif
                    FAssertLF(starpu_task_submit(task) == 0);
                }

            }
        }
        FLOG( FLog::Controller << "\t\t upwardPass in " << timer.tacAndElapsed() << "s\n" );
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
                   #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                       STARPU_NAME, m2lTaskNames[idxLevel].get(),
                   #else
									   //"M2L-l_nb_i"
                                       STARPU_NAME, taskNames.print("M2L", "%d, %d, %lld\n",
                                                                    idxLevel,
                                                                    tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                                    tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval()),
                   #endif
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
                   #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                           STARPU_NAME, m2lOuterTaskNames[idxLevel].get(),
                   #else
										   //"M2L_out-l_nb_i_nb_i_s
                                           STARPU_NAME, taskNames.print("M2L_out", "%d, %d, %lld, %d, %lld, %d\n",
                                                                        idxLevel,
                                                                        tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                                        tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                                        tree->getCellGroup(idxLevel,interactionid)->getNumberOfCellsInBlock(),
                                                                        tree->getCellGroup(idxLevel,interactionid)->getSizeOfInterval(),
                                                                        outsideInteractions->size()),
                   #endif
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
                   #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                           STARPU_NAME, m2lOuterTaskNames[idxLevel].get(),
                   #else
										   //"M2L_out-l_nb_i_nb_i_s"
                                           STARPU_NAME, taskNames.print("M2L_out", "%d, %d, %lld, %d, %lld, %d\n",
                                                                        idxLevel,
                                                                        tree->getCellGroup(idxLevel,interactionid)->getNumberOfCellsInBlock(),
                                                                        tree->getCellGroup(idxLevel,interactionid)->getSizeOfInterval(),
                                                                        tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                                        tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                                        outsideInteractions->size()),
                   #endif
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
            int idxSubGroup = 0;

            for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
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
#ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                    task->name = l2lTaskNames[idxLevel].get();
#else
                    //"L2L-l_nb_i_nbc_ic_s"
                    task->name = taskNames.print("L2L", "%d, %d, %lld, %d, %lld, %lld\n",
                                                 idxLevel,
                                              tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                              tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                 tree->getCellGroup(idxLevel+1,idxSubGroup)->getNumberOfCellsInBlock(),
                                                 tree->getCellGroup(idxLevel+1,idxSubGroup)->getSizeOfInterval(),
                                                 FMath::Min(tree->getCellGroup(idxLevel,idxGroup)->getEndingIndex()-1, (tree->getCellGroup(idxLevel+1,idxSubGroup)->getEndingIndex()-1)>>3)-
                                                 FMath::Max(tree->getCellGroup(idxLevel,idxGroup)->getStartingIndex(), tree->getCellGroup(idxLevel+1,idxSubGroup)->getStartingIndex()>>3));
#endif
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
#ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                    task->name = l2lTaskNames[idxLevel].get();
#else
                    //"L2L-l_nb_i_nbc_ic_s"
                    task->name = taskNames.print("L2L", "%d, %d, %lld, %d, %lld, %lld\n",
                                                 idxLevel,
                                              tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                              tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                 tree->getCellGroup(idxLevel+1,idxSubGroup)->getNumberOfCellsInBlock(),
                                                 tree->getCellGroup(idxLevel+1,idxSubGroup)->getSizeOfInterval(),
                                                 FMath::Min(tree->getCellGroup(idxLevel,idxGroup)->getEndingIndex()-1, (tree->getCellGroup(idxLevel+1,idxSubGroup)->getEndingIndex()-1)>>3)-
                                                 FMath::Max(tree->getCellGroup(idxLevel,idxGroup)->getStartingIndex(), tree->getCellGroup(idxLevel+1,idxSubGroup)->getStartingIndex()>>3));
#endif
    #endif
                    FAssertLF(starpu_task_submit(task) == 0);
                }
            }
        }
        FLOG( FLog::Controller << "\t\t downardPass in " << timer.tacAndElapsed() << "s\n" );
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
                   #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                   STARPU_NAME, p2pOuterTaskNames.get(),
                   #else
								   //"P2P_out-nb_i_p_nb_i_p_s"
                                   STARPU_NAME, taskNames.print("P2P_out", "%d, %lld, %lld, %d, %lld, %lld, %d\n",
                                                                tree->getParticleGroup(idxGroup)->getNumberOfLeavesInBlock(),
                                                                tree->getParticleGroup(idxGroup)->getSizeOfInterval(),
																tree->getParticleGroup(idxGroup)->getNbParticlesInGroup(),
                                                                tree->getParticleGroup(interactionid)->getNumberOfLeavesInBlock(),
                                                                tree->getParticleGroup(interactionid)->getSizeOfInterval(),
																tree->getParticleGroup(interactionid)->getNbParticlesInGroup(),
                                                                outsideInteractions->size()),
                   #endif
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
                   #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                    STARPU_NAME, p2pTaskNames.get(),
                   #else
								//"P2P-nb_i_p"
                               STARPU_NAME, taskNames.print("P2P", "%d, %lld, %lld\n",
                                                            tree->getParticleGroup(idxGroup)->getNumberOfLeavesInBlock(),
                                                            tree->getParticleGroup(idxGroup)->getSizeOfInterval(),
															tree->getParticleGroup(idxGroup)->getNbParticlesInGroup()),
                   #endif
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
        #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                    STARPU_NAME, l2pTaskNames.get(),
        #else
					//"L2P-nb_i_p"
                    STARPU_NAME, taskNames.print("L2P", "%d, %lld, %lld\n",
                                                 tree->getCellGroup(tree->getHeight()-1,idxGroup)->getNumberOfCellsInBlock(),
                                                 tree->getCellGroup(tree->getHeight()-1,idxGroup)->getSizeOfInterval(),
												 tree->getCellGroup(tree->getHeight()-1,idxGroup)->getNumberOfCellsInBlock()),
        #endif
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

#endif // FGROUPTASKSTARPUALGORITHM_HPP
