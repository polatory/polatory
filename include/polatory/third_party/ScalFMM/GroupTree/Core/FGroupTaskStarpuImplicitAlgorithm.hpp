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
#ifdef SCALFMM_USE_STARPU_EXTRACT
#include <list>
#endif

#include <omp.h>

#include <starpu.h>
#include <starpu_mpi.h>
#ifdef SCALFMM_USE_STARPU_EXTRACT
#include <algorithm>
#endif
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
#include <iostream>
#include <vector>
using namespace std;

//#define STARPU_USE_REDUX
template <class OctreeClass, class CellContainerClass, class KernelClass, class ParticleGroupClass, class StarPUCpuWrapperClass
          #ifdef SCALFMM_ENABLE_CUDA_KERNEL
          , class StarPUCudaWrapperClass = FStarPUCudaWrapper<KernelClass, FCudaEmptyCellSymb, int, int, FCudaGroupOfCells<FCudaEmptyCellSymb, int, int>,
                                                              FCudaGroupOfParticles<int, 0, 0, int>, FCudaGroupAttachedLeaf<int, 0, 0, int>, FCudaEmptyKernel<int> >
          #endif
          #ifdef SCALFMM_ENABLE_OPENCL_KERNEL
          , class StarPUOpenClWrapperClass = FStarPUOpenClWrapper<KernelClass, FOpenCLDeviceWrapper<KernelClass>>
          #endif
          >
class FGroupTaskStarPUImplicitAlgorithm : public FAbstractAlgorithm {
protected:
    typedef FGroupTaskStarPUImplicitAlgorithm<OctreeClass, CellContainerClass, KernelClass, ParticleGroupClass, StarPUCpuWrapperClass
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
#ifdef SCALFMM_USE_STARPU_EXTRACT
    std::vector< std::vector< std::vector<std::vector<int>>>> externalInteractionsAllLevelInnerIndexes;
    std::vector< std::vector< std::vector<std::vector<int>>>> externalInteractionsAllLevelOuterIndexes;
#endif
    std::vector< std::vector<BlockInteractions<ParticleGroupClass>>> externalInteractionsLeafLevel;
#ifdef SCALFMM_USE_STARPU_EXTRACT
    std::vector< std::vector<std::vector<int>>> externalInteractionsLeafLevelOuter;
    std::vector< std::vector<std::vector<int>>> externalInteractionsLeafLevelInner;
#endif
    std::list<const std::vector<OutOfBlockInteraction>*> externalInteractionsLeafLevelOpposite;

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
    FStarPUTaskNameParams* taskNames = nullptr;
#endif
#endif
#ifdef SCALFMM_STARPU_USE_PRIO
    typedef FStarPUFmmPrioritiesV2 PrioClass;// FStarPUFmmPriorities
#endif
    int mpi_rank, nproc;
    std::vector<std::vector<std::vector<MortonIndex>>> nodeRepartition;

#ifdef SCALFMM_USE_STARPU_EXTRACT
    struct ParticleExtractedHandles{
        starpu_data_handle_t symb;
        size_t size;
        std::unique_ptr<unsigned char[]> data;
        std::vector<int> leavesToExtract;
    };

    std::list<ParticleExtractedHandles> extractedParticlesBuffer;

    struct DuplicatedParticlesHandle{
        starpu_data_handle_t symb;
        size_t size;
        unsigned char* data; // Never delete it, we reuse already allocate memory here
    };

    std::list<DuplicatedParticlesHandle> duplicatedParticlesBuffer;

    starpu_codelet p2p_extract;
    starpu_codelet p2p_insert;
    starpu_codelet p2p_insert_bis;

    struct CellExtractedHandles{
        starpu_data_handle_t all;
        size_t size;
        std::unique_ptr<unsigned char[]> data;
        std::vector<int> cellsToExtract;
    };

    std::list<CellExtractedHandles> extractedCellBuffer;

    struct DuplicatedCellHandle{
        starpu_data_handle_t symb;
        size_t sizeSymb;
        unsigned char* dataSymb; // Never delete it, we reuse already allocate memory here
        starpu_data_handle_t other;
        size_t sizeOther;
        unsigned char* dataOther; // Never delete it, we reuse already allocate memory here

        std::unique_ptr<unsigned char[]> dataSymbPtr;
        std::unique_ptr<unsigned char[]> dataOtherPtr;
    };

    std::list<DuplicatedCellHandle> duplicatedCellBuffer;

    starpu_codelet cell_extract_up;
    starpu_codelet cell_insert_up;
    starpu_codelet cell_insert_up_bis;
#endif

public:
    FGroupTaskStarPUImplicitAlgorithm(OctreeClass*const inTree, KernelClass* inKernels, std::vector<MortonIndex>& distributedMortonIndex)
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
        FAssertLF(starpu_mpi_init ( 0, 0, 0 ) == 0);
        MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD,&nproc);
#ifdef STARPU_USE_TASK_NAME
#ifdef SCALFMM_SIMGRID_TASKNAMEPARAMS
        taskNames = new FStarPUTaskNameParams(mpi_rank, nproc);
#endif
#endif
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
        createMachinChose(distributedMortonIndex);
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
                if(isDataOwnedBerenger(tree->getCellGroup(idxLevel, idxHandle)->getStartingIndex(), idxLevel)) {//Clean only our data handle
                    starpu_data_acquire(cellHandles[idxLevel][idxHandle].symb, STARPU_R);
                    starpu_data_release(cellHandles[idxLevel][idxHandle].symb);
                    starpu_data_acquire(cellHandles[idxLevel][idxHandle].up, STARPU_R);
                    starpu_data_release(cellHandles[idxLevel][idxHandle].up);
                    starpu_data_acquire(cellHandles[idxLevel][idxHandle].down, STARPU_R);
                    starpu_data_release(cellHandles[idxLevel][idxHandle].down);
                }
            }
        }
        {
            for(int idxHandle = 0 ; idxHandle < int(particleHandles.size()) ; ++idxHandle){
                if(isDataOwnedBerenger(tree->getCellGroup(tree->getHeight()-1, idxHandle)->getStartingIndex(), tree->getHeight()-1)) {//Clean only our data handle
                    starpu_data_acquire(particleHandles[idxHandle].symb, STARPU_R);
                    starpu_data_release(particleHandles[idxHandle].symb);
                    starpu_data_acquire(particleHandles[idxHandle].down, STARPU_R);
                    starpu_data_release(particleHandles[idxHandle].down);
                }
            }
        }
    }

    ~FGroupTaskStarPUImplicitAlgorithm(){
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

        for(auto externalInteraction : externalInteractionsLeafLevelOpposite)
            delete externalInteraction;

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

    int getRank(void) const {
        return mpi_rank;
    }
    int getNProc(void) const {
        return nproc;
    }
    bool isDataOwnedBerenger(MortonIndex const idx, int const idxLevel) const {
        return dataMappingBerenger(idx, idxLevel) == mpi_rank;
    }
    void createMachinChose(std::vector<MortonIndex> distributedMortonIndex) {
        nodeRepartition.resize(tree->getHeight(), std::vector<std::vector<MortonIndex>>(nproc, std::vector<MortonIndex>(2)));
        for(int node_id = 0; node_id < nproc; ++node_id){
            nodeRepartition[tree->getHeight()-1][node_id][0] = distributedMortonIndex[node_id*2];
            nodeRepartition[tree->getHeight()-1][node_id][1] = distributedMortonIndex[node_id*2+1];
        }
        for(int idxLevel = tree->getHeight() - 2; idxLevel >= 0  ; --idxLevel){
            nodeRepartition[idxLevel][0][0] = nodeRepartition[idxLevel+1][0][0] >> 3;
            nodeRepartition[idxLevel][0][1] = nodeRepartition[idxLevel+1][0][1] >> 3;
            for(int node_id = 1; node_id < nproc; ++node_id){
                nodeRepartition[idxLevel][node_id][0] = FMath::Max(nodeRepartition[idxLevel+1][node_id][0] >> 3, nodeRepartition[idxLevel][node_id-1][0]+1); //Berenger phd :)
                nodeRepartition[idxLevel][node_id][1] = nodeRepartition[idxLevel+1][node_id][1] >> 3;
            }
        }
    }
    int getOppositeInterIndex(const int index) const {
        // ((( (xdiff+3) * 7) + (ydiff+3))) * 7 + zdiff + 3
        return 343-index-1;
    }
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
        p2p_cl_inout.modes[1] = starpu_data_access_mode(STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED);
#endif
        p2p_cl_inout.modes[2] = STARPU_R;
#ifdef STARPU_USE_REDUX
        p2p_cl_inout.modes[3] = STARPU_REDUX;
#else
        p2p_cl_inout.modes[3] = starpu_data_access_mode(STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED);
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

#ifdef SCALFMM_USE_STARPU_EXTRACT
        memset(&p2p_extract, 0, sizeof(p2p_extract));
        p2p_extract.nbuffers = 2;
        p2p_extract.modes[0] = STARPU_R;
        p2p_extract.modes[1] = STARPU_RW;
        p2p_extract.name = "p2p_extract";
        p2p_extract.cpu_funcs[0] = ThisClass::ExtractP2P;
        p2p_extract.where |= STARPU_CPU;

        memset(&p2p_insert, 0, sizeof(p2p_insert));
        p2p_insert.nbuffers = 2;
        p2p_insert.modes[0] = STARPU_R;
        p2p_insert.modes[1] = STARPU_RW;
        p2p_insert.name = "p2p_insert";
        p2p_insert.cpu_funcs[0] = ThisClass::InsertP2P;
        p2p_insert.where |= STARPU_CPU;

        memset(&p2p_insert_bis, 0, sizeof(p2p_insert_bis));
        p2p_insert_bis.nbuffers = 2;
        p2p_insert_bis.modes[0] = STARPU_R;
        p2p_insert_bis.modes[1] = STARPU_RW;
        p2p_insert_bis.name = "p2p_insert_bis";
        p2p_insert_bis.cpu_funcs[0] = ThisClass::InsertP2PBis;
        p2p_insert_bis.where |= STARPU_CPU;

        memset(&cell_extract_up, 0, sizeof(cell_extract_up));
        cell_extract_up.nbuffers = 3;
        cell_extract_up.modes[0] = STARPU_R;
        cell_extract_up.modes[1] = STARPU_R;
        cell_extract_up.modes[2] = STARPU_RW;
        cell_extract_up.name = "cell_extract_up";
        cell_extract_up.cpu_funcs[0] = ThisClass::ExtractCellUp;
        cell_extract_up.where |= STARPU_CPU;

        memset(&cell_insert_up, 0, sizeof(cell_insert_up));
        cell_insert_up.nbuffers = 3;
        cell_insert_up.modes[0] = STARPU_R;
        cell_insert_up.modes[1] = STARPU_RW;
        cell_insert_up.modes[2] = STARPU_RW;
        cell_insert_up.name = "cell_insert_up";
        cell_insert_up.cpu_funcs[0] = ThisClass::InsertCellUp;
        cell_insert_up.where |= STARPU_CPU;


        memset(&cell_insert_up_bis, 0, sizeof(cell_insert_up_bis));
        cell_insert_up_bis.nbuffers = 3;
        cell_insert_up_bis.modes[0] = STARPU_R;
        cell_insert_up_bis.modes[1] = STARPU_RW;
        cell_insert_up_bis.modes[2] = STARPU_RW;
        cell_insert_up_bis.name = "cell_insert_up_bis";
        cell_insert_up_bis.cpu_funcs[0] = ThisClass::InsertCellUpBis;
        cell_insert_up_bis.where |= STARPU_CPU;
#endif
    }

#ifdef SCALFMM_USE_STARPU_EXTRACT
    static void InsertP2P(void *buffers[], void *cl_arg){
        ParticleGroupClass containers((unsigned char*)STARPU_VECTOR_GET_PTR(buffers[1]),
                                      STARPU_VECTOR_GET_NX(buffers[1]),
                                      nullptr);

        ParticleExtractedHandles* interactionBufferPtr;
        starpu_codelet_unpack_args(cl_arg, &interactionBufferPtr);

        containers.restoreData(interactionBufferPtr->leavesToExtract,
                               (unsigned char*)STARPU_VECTOR_GET_PTR(buffers[0]),
                                STARPU_VECTOR_GET_NX(buffers[0]));
    }

    static void InsertP2PBis(void *buffers[], void *cl_arg){
        ParticleExtractedHandles* interactionBufferPtr;
        const unsigned char* dataPtr;
        size_t datasize;
        starpu_codelet_unpack_args(cl_arg, &interactionBufferPtr, &dataPtr, &datasize);

        memcpy((unsigned char*)STARPU_VECTOR_GET_PTR(buffers[1]), dataPtr, datasize);

        ParticleGroupClass containers((unsigned char*)STARPU_VECTOR_GET_PTR(buffers[1]),
                                      STARPU_VECTOR_GET_NX(buffers[1]),
                                      nullptr);


        containers.restoreData(interactionBufferPtr->leavesToExtract,
                               (unsigned char*)STARPU_VECTOR_GET_PTR(buffers[0]),
                                STARPU_VECTOR_GET_NX(buffers[0]));
    }

    static void ExtractP2P(void *buffers[], void *cl_arg){
        ParticleGroupClass containers((unsigned char*)STARPU_VECTOR_GET_PTR(buffers[0]),
                                      STARPU_VECTOR_GET_NX(buffers[0]),
                                      nullptr);

        ParticleExtractedHandles* interactionBufferPtr;
        starpu_codelet_unpack_args(cl_arg, &interactionBufferPtr);

        containers.extractData(interactionBufferPtr->leavesToExtract,
                               (unsigned char*)STARPU_VECTOR_GET_PTR(buffers[1]),
                               STARPU_VECTOR_GET_NX(buffers[1]));
    }

    static void InsertCellUp(void *buffers[], void *cl_arg){
        CellContainerClass currentCells((unsigned char*)STARPU_VECTOR_GET_PTR(buffers[1]),
                                        STARPU_VECTOR_GET_NX(buffers[1]),
                                        (unsigned char*)STARPU_VECTOR_GET_PTR(buffers[2]),
                                        nullptr);

        CellExtractedHandles* interactionBufferPtr;
        starpu_codelet_unpack_args(cl_arg, &interactionBufferPtr);

        currentCells.restoreDataUp(interactionBufferPtr->cellsToExtract,
                                   (unsigned char*)STARPU_VECTOR_GET_PTR(buffers[0]),
                                   STARPU_VECTOR_GET_NX(buffers[0]));
    }

    static void InsertCellUpBis(void *buffers[], void *cl_arg){
        unsigned char* ptr1;
        size_t size1;
        unsigned char* ptr2;
        size_t size2;
        CellExtractedHandles* interactionBufferPtr;
        starpu_codelet_unpack_args(cl_arg, &interactionBufferPtr, &ptr1, &size1, &ptr2, &size2);

        memcpy((unsigned char*)STARPU_VECTOR_GET_PTR(buffers[1]), ptr1, size1);
        memcpy((unsigned char*)STARPU_VECTOR_GET_PTR(buffers[2]), ptr2, size2);

        CellContainerClass currentCells((unsigned char*)STARPU_VECTOR_GET_PTR(buffers[1]),
                                        STARPU_VECTOR_GET_NX(buffers[1]),
                                        (unsigned char*)STARPU_VECTOR_GET_PTR(buffers[2]),
                                        nullptr);


        currentCells.restoreDataUp(interactionBufferPtr->cellsToExtract,
                                   (unsigned char*)STARPU_VECTOR_GET_PTR(buffers[0]),
                                   STARPU_VECTOR_GET_NX(buffers[0]));
    }

    static void ExtractCellUp(void *buffers[], void *cl_arg){
        CellContainerClass currentCells((unsigned char*)STARPU_VECTOR_GET_PTR(buffers[0]),
                                        STARPU_VECTOR_GET_NX(buffers[0]),
                                        (unsigned char*)STARPU_VECTOR_GET_PTR(buffers[1]),
                                        nullptr);

        CellExtractedHandles* interactionBufferPtr;
        starpu_codelet_unpack_args(cl_arg, &interactionBufferPtr);

        currentCells.extractDataUp(interactionBufferPtr->cellsToExtract,
                                   (unsigned char*)STARPU_VECTOR_GET_PTR(buffers[2]),
                                   STARPU_VECTOR_GET_NX(buffers[2]));
    }
#endif

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

    /** dealloc in a starpu way all the defined handles */
    void cleanHandle(){
        for(int idxLevel = 0 ; idxLevel < tree->getHeight() ; ++idxLevel){
            for(int idxHandle = 0 ; idxHandle < int(cellHandles[idxLevel].size()) ; ++idxHandle){
                if(isDataOwnedBerenger(tree->getCellGroup(idxLevel, idxHandle)->getStartingIndex(), idxLevel))//Clean only our data handle
                {
                    starpu_data_unregister(cellHandles[idxLevel][idxHandle].symb);
                    starpu_data_unregister(cellHandles[idxLevel][idxHandle].up);
                    starpu_data_unregister(cellHandles[idxLevel][idxHandle].down);
                }
            }
            cellHandles[idxLevel].clear();
        }
        {
            for(int idxHandle = 0 ; idxHandle < int(particleHandles.size()) ; ++idxHandle){
                if(isDataOwnedBerenger(tree->getCellGroup(tree->getHeight()-1, idxHandle)->getStartingIndex(), tree->getHeight()-1))//Clean only our data handle
                {
                    starpu_data_unregister(particleHandles[idxHandle].symb);
                    starpu_data_unregister(particleHandles[idxHandle].down);
                }
            }
            particleHandles.clear();
        }
#ifdef SCALFMM_USE_STARPU_EXTRACT
        for(auto& iter : extractedParticlesBuffer){
            starpu_data_unregister(iter.symb);
        }
        for(auto& iter : duplicatedParticlesBuffer){
            starpu_data_unregister(iter.symb);
        }
        for(auto& iter : extractedCellBuffer){
            starpu_data_unregister(iter.all);
        }
        for(auto& iter : duplicatedCellBuffer){
            starpu_data_unregister(iter.symb);
        }
#endif
    }

    /** Reset the handles array and create new ones to define
     * in a starpu way each block of data
     */
    int tag;
    void buildHandles(){
        cleanHandle();
        tag = 0;
        int where;
        for(int idxLevel = 2 ; idxLevel < tree->getHeight() ; ++idxLevel){
            cellHandles[idxLevel].resize(tree->getNbCellGroupAtLevel(idxLevel));
            for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
                const CellContainerClass* currentCells = tree->getCellGroup(idxLevel, idxGroup);
                int registeringNode = dataMappingBerenger(currentCells->getStartingIndex(), idxLevel);

                where = (registeringNode == mpi_rank) ? STARPU_MAIN_RAM : -1;
                starpu_variable_data_register(&cellHandles[idxLevel][idxGroup].symb, where,
                                              (uintptr_t)currentCells->getRawBuffer(), currentCells->getBufferSizeInByte());
                starpu_variable_data_register(&cellHandles[idxLevel][idxGroup].up, where,
                                              (uintptr_t)currentCells->getRawMultipoleBuffer(), currentCells->getMultipoleBufferSizeInByte());
                starpu_variable_data_register(&cellHandles[idxLevel][idxGroup].down, where,
                                              (uintptr_t)currentCells->getRawLocalBuffer(), currentCells->getLocalBufferSizeInByte());

                starpu_mpi_data_register(cellHandles[idxLevel][idxGroup].symb, tag++, registeringNode);
                starpu_mpi_data_register(cellHandles[idxLevel][idxGroup].up, tag++, registeringNode);
                starpu_mpi_data_register(cellHandles[idxLevel][idxGroup].down, tag++, registeringNode);
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
                int registeringNode = dataMappingBerenger(tree->getCellGroup(tree->getHeight()-1, idxGroup)->getStartingIndex(), tree->getHeight()-1);
                where = (registeringNode == mpi_rank) ? STARPU_MAIN_RAM : -1;
                ParticleGroupClass* containers = tree->getParticleGroup(idxGroup);
                starpu_variable_data_register(&particleHandles[idxGroup].symb, where,
                                              (uintptr_t)containers->getRawBuffer(), containers->getBufferSizeInByte());
                starpu_variable_data_register(&particleHandles[idxGroup].down, where,
                                              (uintptr_t)containers->getRawAttributesBuffer(), containers->getAttributesBufferSizeInByte());

                starpu_mpi_data_register(particleHandles[idxGroup].symb, tag++, registeringNode);
                starpu_mpi_data_register(particleHandles[idxGroup].down, tag++, registeringNode);
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
    int dataMappingBerenger(MortonIndex const idx, int const idxLevel) const {
        for(int i = 0; i < nproc; ++i)
            if(nodeRepartition[idxLevel][i][0] <= nodeRepartition[idxLevel][i][1] && idx >= nodeRepartition[idxLevel][i][0] && idx <= nodeRepartition[idxLevel][i][1])
                return i;
        if(mpi_rank == 0)
            cout << "[scalfmm][map error] idx " << idx << " on level " << idxLevel << " isn't mapped on any proccess. (Default set to 0)." << endl;
        return nproc-1;
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
#ifdef SCALFMM_USE_STARPU_EXTRACT
        externalInteractionsAllLevelInnerIndexes.clear();
        externalInteractionsAllLevelOuterIndexes.clear();
#endif
        externalInteractionsLeafLevel.clear();
#ifdef SCALFMM_USE_STARPU_EXTRACT
        externalInteractionsLeafLevelOuter.clear();
        externalInteractionsLeafLevelInner.clear();
#endif
        // One per level + leaf level
        externalInteractionsAllLevel.resize(tree->getHeight());
#ifdef SCALFMM_USE_STARPU_EXTRACT
        externalInteractionsAllLevelInnerIndexes.resize(tree->getHeight());
        externalInteractionsAllLevelOuterIndexes.resize(tree->getHeight());
#endif

        // First leaf level
        {
            // We create one big vector per block
            externalInteractionsLeafLevel.resize(tree->getNbParticleGroup());
#ifdef SCALFMM_USE_STARPU_EXTRACT
            externalInteractionsLeafLevelOuter.resize(tree->getNbParticleGroup());
            externalInteractionsLeafLevelInner.resize(tree->getNbParticleGroup());
#endif

            for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
                // Create the vector
                ParticleGroupClass* containers = tree->getParticleGroup(idxGroup);

                std::vector<BlockInteractions<ParticleGroupClass>>* externalInteractions = &externalInteractionsLeafLevel[idxGroup];
#ifdef SCALFMM_USE_STARPU_EXTRACT
                std::vector<std::vector<int>>* externalInteractionsOuter = &externalInteractionsLeafLevelOuter[idxGroup];
                std::vector<std::vector<int>>* externalInteractionsInner = &externalInteractionsLeafLevelInner[idxGroup];
#endif

#ifdef SCALFMM_USE_STARPU_EXTRACT
#pragma omp task default(none) firstprivate(idxGroup, containers, externalInteractions, externalInteractionsOuter, externalInteractionsInner)
#else
                #pragma omp task default(none) firstprivate(idxGroup, containers, externalInteractions)
#endif
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

#ifdef SCALFMM_USE_STARPU_EXTRACT
                            externalInteractionsOuter->emplace_back();
                            externalInteractionsInner->emplace_back();

                            std::vector<int>* interactionsOuter = &externalInteractionsOuter->back();
                            std::vector<int>* interactionsInner = &externalInteractionsInner->back();

                            for(int idxUnique = 0 ; idxUnique < interactions->interactions.size() ; ++idxUnique){
                                interactionsOuter->push_back(interactions->interactions[idxUnique].outsideIdxInBlock);
                                interactionsInner->push_back(interactions->interactions[idxUnique].insideIdxInBlock);
                            }
                            FQuickSort<int, int>::QsSequential(interactionsOuter->data(),int(interactionsOuter->size()));
                            FQuickSort<int, int>::QsSequential(interactionsInner->data(),int(interactionsInner->size()));

                            interactionsOuter->erase(std::unique(interactionsOuter->begin(), interactionsOuter->end()), interactionsOuter->end());
                            interactionsInner->erase(std::unique(interactionsInner->begin(), interactionsInner->end()), interactionsInner->end());
#endif
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
#ifdef SCALFMM_USE_STARPU_EXTRACT
                externalInteractionsAllLevelInnerIndexes[idxLevel].resize(tree->getNbCellGroupAtLevel(idxLevel));
                externalInteractionsAllLevelOuterIndexes[idxLevel].resize(tree->getNbCellGroupAtLevel(idxLevel));
#endif
                for(int idxGroup = 0 ; idxGroup < tree->getNbCellGroupAtLevel(idxLevel) ; ++idxGroup){
                    CellContainerClass* currentCells = tree->getCellGroup(idxLevel, idxGroup);

                    std::vector<BlockInteractions<CellContainerClass>>* externalInteractions = &externalInteractionsAllLevel[idxLevel][idxGroup];
#ifdef SCALFMM_USE_STARPU_EXTRACT
                    std::vector<std::vector<int>>* externalInteractionsInner = &externalInteractionsAllLevelInnerIndexes[idxLevel][idxGroup];
                    std::vector<std::vector<int>>* externalInteractionsOuter = &externalInteractionsAllLevelOuterIndexes[idxLevel][idxGroup];
#endif

#ifdef SCALFMM_USE_STARPU_EXTRACT
#pragma omp task default(none) firstprivate(idxGroup, currentCells, idxLevel, externalInteractions, externalInteractionsInner, externalInteractionsOuter)
#else
                    #pragma omp task default(none) firstprivate(idxGroup, currentCells, idxLevel, externalInteractions)
#endif
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

#ifdef SCALFMM_USE_STARPU_EXTRACT
                                externalInteractionsInner->emplace_back();
                                std::vector<int>* interactionsInnerIndexes = &externalInteractionsInner->back();
                                externalInteractionsOuter->emplace_back();
                                std::vector<int>* interactionsOuterIndexes = &externalInteractionsOuter->back();

                                for(int idxUnique = 0 ; idxUnique < interactions->interactions.size() ; ++idxUnique){
                                    interactionsOuterIndexes->push_back(interactions->interactions[idxUnique].outsideIdxInBlock);
                                    interactionsInnerIndexes->push_back(interactions->interactions[idxUnique].insideIdxInBlock);
                                }

                                FQuickSort<int, int>::QsSequential(interactionsOuterIndexes->data(),int(interactionsOuterIndexes->size()));
                                interactionsOuterIndexes->erase(std::unique(interactionsOuterIndexes->begin(), interactionsOuterIndexes->end()),
                                                                interactionsOuterIndexes->end());
                                FQuickSort<int, int>::QsSequential(interactionsInnerIndexes->data(),int(interactionsInnerIndexes->size()));
                                interactionsInnerIndexes->erase(std::unique(interactionsInnerIndexes->begin(), interactionsInnerIndexes->end()),
                                                                interactionsInnerIndexes->end());
#endif
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
            starpu_mpi_insert_task(MPI_COMM_WORLD,
                                   &p2m_cl,
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
                    STARPU_NAME, taskNames->print("P2M", "%d, %lld, %lld, %lld, %lld, %d\n",
                                                  tree->getCellGroup(tree->getHeight()-1,idxGroup)->getNumberOfCellsInBlock(),
                                                  tree->getCellGroup(tree->getHeight()-1,idxGroup)->getSizeOfInterval(),
                                                  tree->getCellGroup(tree->getHeight()-1,idxGroup)->getNumberOfCellsInBlock(),
                                                  tree->getParticleGroup(idxGroup)->getStartingIndex(),
                                                  tree->getParticleGroup(idxGroup)->getEndingIndex(),
                                                  starpu_mpi_data_get_rank(cellHandles[tree->getHeight()-1][idxGroup].up)),
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
                    starpu_mpi_insert_task(MPI_COMM_WORLD,
                                           &m2m_cl,
                                           STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                           STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                           STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                           STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2M(idxLevel),
                       #endif
                                           STARPU_R, cellHandles[idxLevel][idxGroup].symb, //symbolique, readonly
                                           (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][idxGroup].up, //The remaining, read/write
                                           STARPU_R, cellHandles[idxLevel+1][idxSubGroup].symb, //symbolique, readonly
                            STARPU_R, cellHandles[idxLevel+1][idxSubGroup].up, //level d'avant readonly
        #ifdef STARPU_USE_TASK_NAME
        #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                            STARPU_NAME, m2mTaskNames[idxLevel].get(),
        #else
                            //"M2M-l_nb_i_nbc_ic_s"
                            STARPU_NAME, taskNames->print("M2M", "%d, %d, %lld, %d, %lld, %lld, %lld, %lld, %lld, %lld, %d\n",
                                                          idxLevel,
                                                          tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                          tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                          tree->getCellGroup(idxLevel+1,idxSubGroup)->getNumberOfCellsInBlock(),
                                                          tree->getCellGroup(idxLevel+1,idxSubGroup)->getSizeOfInterval(),
                                                          FMath::Min(tree->getCellGroup(idxLevel,idxGroup)->getEndingIndex()-1, (tree->getCellGroup(idxLevel+1,idxSubGroup)->getEndingIndex()-1)>>3)-
                                                          FMath::Max(tree->getCellGroup(idxLevel,idxGroup)->getStartingIndex(), tree->getCellGroup(idxLevel+1,idxSubGroup)->getStartingIndex()>>3),
                                                          tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                          tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                          tree->getCellGroup(idxLevel+1, idxSubGroup)->getStartingIndex(),
                                                          tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex(),
                                                          starpu_mpi_data_get_rank(cellHandles[idxLevel][idxGroup].up)),
        #endif
        #endif
                            0);

                }

                while(tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex() <= (((currentCells->getEndingIndex()-1)<<3)+7)
                      && (idxSubGroup+1) != tree->getNbCellGroupAtLevel(idxLevel+1)
                      && tree->getCellGroup(idxLevel+1, idxSubGroup+1)->getStartingIndex() <= ((currentCells->getEndingIndex()-1)<<3)+7 ){
                    idxSubGroup += 1;

                    starpu_mpi_insert_task(MPI_COMM_WORLD,
                                           &m2m_cl,
                                           STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                           STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                           STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                           STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2M(idxLevel),
                       #endif
                                           STARPU_R, cellHandles[idxLevel][idxGroup].symb, //symbolique, readonly
                                           (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][idxGroup].up, //The remaining, read/write
                                           STARPU_R, cellHandles[idxLevel+1][idxSubGroup].symb, //symbolique, readonly
                            STARPU_R, cellHandles[idxLevel+1][idxSubGroup].up, //level d'avant readonly
        #ifdef STARPU_USE_TASK_NAME
        #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                            STARPU_NAME, m2mTaskNames[idxLevel].get(),
        #else
                            //M2M-l_nb_i_nbc_ic_s
                            STARPU_NAME, taskNames->print("M2M", "%d, %d, %lld, %d, %lld, %lld, %lld, %lld, %lld, %lld, %d\n",
                                                          idxLevel,
                                                          tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                          tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                          tree->getCellGroup(idxLevel+1,idxSubGroup)->getNumberOfCellsInBlock(),
                                                          tree->getCellGroup(idxLevel+1,idxSubGroup)->getSizeOfInterval(),
                                                          FMath::Min(tree->getCellGroup(idxLevel,idxGroup)->getEndingIndex()-1, (tree->getCellGroup(idxLevel+1,idxSubGroup)->getEndingIndex()-1)>>3)-
                                                          FMath::Max(tree->getCellGroup(idxLevel,idxGroup)->getStartingIndex(), tree->getCellGroup(idxLevel+1,idxSubGroup)->getStartingIndex()>>3),
                                                          tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                          tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                          tree->getCellGroup(idxLevel+1, idxSubGroup)->getStartingIndex(),
                                                          tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex(),
                                                          starpu_mpi_data_get_rank(cellHandles[idxLevel][idxGroup].up)),
        #endif
        #endif
                            0);
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
                    starpu_mpi_insert_task(MPI_COMM_WORLD,
                                           &m2l_cl_in,
                                           STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                           STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                           STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                           STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2L(idxLevel),
                       #endif
                                           STARPU_R, cellHandles[idxLevel][idxGroup].symb,
                                           STARPU_R, cellHandles[idxLevel][idxGroup].up,
                                           (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][idxGroup].down,
                       #ifdef STARPU_USE_TASK_NAME
                       #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                           STARPU_NAME, m2lTaskNames[idxLevel].get(),
                       #else
                                           //"M2L-l_nb_i"
                                           STARPU_NAME, taskNames->print("M2L", "%d, %d, %lld, %lld, %lld, %lld, %lld, %d\n",
                                                                         idxLevel,
                                                                         tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                                         tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                                         tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                                         tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                                         tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                                         tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                                         starpu_mpi_data_get_rank(cellHandles[idxLevel][idxGroup].down)),
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
#ifdef SCALFMM_USE_STARPU_EXTRACT
                        // On the same node -- do as usual
                        if(starpu_mpi_data_get_rank(cellHandles[idxLevel][idxGroup].symb) == starpu_mpi_data_get_rank(cellHandles[idxLevel][interactionid].symb)){
#endif
                            int mode = 1;
                            starpu_mpi_insert_task(MPI_COMM_WORLD,
                                                   &m2l_cl_inout,
                                                   STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                                   STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                                   STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                                   STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                                                   STARPU_VALUE, &mode, sizeof(int),
                           #ifdef SCALFMM_STARPU_USE_PRIO
                                                   STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                           #endif
                                                   STARPU_R, cellHandles[idxLevel][idxGroup].symb,
                                                   (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][idxGroup].down,
                                                   STARPU_R, cellHandles[idxLevel][interactionid].symb,
                                                   STARPU_R, cellHandles[idxLevel][interactionid].up,
                           #ifdef STARPU_USE_TASK_NAME
                           #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                                   STARPU_NAME, m2lOuterTaskNames[idxLevel].get(),
                           #else
                                                   //"M2L_out-l_nb_i_nb_i_s
                                                   STARPU_NAME, taskNames->print("M2L_out", "%d, %d, %lld, %d, %lld, %d, %lld, %lld, %lld, %lld, %d\n",
                                                                                 idxLevel,
                                                                                 tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                                                 tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                                                 tree->getCellGroup(idxLevel,interactionid)->getNumberOfCellsInBlock(),
                                                                                 tree->getCellGroup(idxLevel,interactionid)->getSizeOfInterval(),
                                                                                 outsideInteractions->size(),
                                                                                 tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                                                 tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                                                 tree->getCellGroup(idxLevel, interactionid)->getStartingIndex(),
                                                                                 tree->getCellGroup(idxLevel, interactionid)->getEndingIndex(),
                                                                                 starpu_mpi_data_get_rank(cellHandles[idxLevel][idxGroup].down)),
                           #endif
                           #endif
                                                   0);

                            mode = 2;
                            starpu_mpi_insert_task(MPI_COMM_WORLD,
                                                   &m2l_cl_inout,
                                                   STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                                   STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                                   STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                                   STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                                                   STARPU_VALUE, &mode, sizeof(int),
                           #ifdef SCALFMM_STARPU_USE_PRIO
                                                   STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                           #endif
                                                   STARPU_R, cellHandles[idxLevel][interactionid].symb,
                                                   (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][interactionid].down,
                                                   STARPU_R, cellHandles[idxLevel][idxGroup].symb,
                                                   STARPU_R, cellHandles[idxLevel][idxGroup].up,
                           #ifdef STARPU_USE_TASK_NAME
                           #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                                   STARPU_NAME, m2lOuterTaskNames[idxLevel].get(),
                           #else
                                                   //"M2L_out-l_nb_i_nb_i_s"
                                                   STARPU_NAME, taskNames->print("M2L_out", "%d, %d, %lld, %d, %lld, %d, %lld, %lld, %lld, %lld, %d\n",
                                                                                 idxLevel,
                                                                                 tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                                                 tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                                                 tree->getCellGroup(idxLevel,interactionid)->getNumberOfCellsInBlock(),
                                                                                 tree->getCellGroup(idxLevel,interactionid)->getSizeOfInterval(),
                                                                                 outsideInteractions->size(),
                                                                                 tree->getCellGroup(idxLevel, interactionid)->getStartingIndex(),
                                                                                 tree->getCellGroup(idxLevel, interactionid)->getEndingIndex(),
                                                                                 tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                                                 tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                                                 starpu_mpi_data_get_rank(cellHandles[idxLevel][interactionid].down)),
                           #endif
                           #endif
                                                   0);

#ifdef SCALFMM_USE_STARPU_EXTRACT
                        }
                        else{
                                {

                                    // Extract data from second group for the first one
                                    // That is copy B to B'
                                    extractedCellBuffer.emplace_back();
                                    CellExtractedHandles& interactionBuffer = extractedCellBuffer.back();
                                    interactionBuffer.cellsToExtract = externalInteractionsAllLevelOuterIndexes[idxLevel][idxGroup][idxInteraction];
                                    interactionBuffer.size = tree->getCellGroup(idxLevel,interactionid)->extractGetSizeSymbUp(interactionBuffer.cellsToExtract);
                                    // I allocate only if I will use it to extract
                                    if(starpu_mpi_data_get_rank(cellHandles[idxLevel][interactionid].symb) == mpi_rank){
                                        interactionBuffer.data.reset(new unsigned char[interactionBuffer.size]);
                                        FAssertLF(interactionBuffer.data);
                                    }
                                    else{
                                        interactionBuffer.data.reset(nullptr);
                                    }
                                    int registeringNode = starpu_mpi_data_get_rank(cellHandles[idxLevel][interactionid].symb);
                                    int where = (registeringNode == mpi_rank) ? STARPU_MAIN_RAM : -1;
                                    starpu_variable_data_register(&interactionBuffer.all, where,
                                                                  (uintptr_t)interactionBuffer.data.get(), interactionBuffer.size);
                                    starpu_mpi_data_register(interactionBuffer.all, tag++, registeringNode);

                                    CellExtractedHandles* interactionBufferPtr = &interactionBuffer;
                                    starpu_mpi_insert_task(MPI_COMM_WORLD,
                                                           &cell_extract_up,
                                                           STARPU_VALUE, &interactionBufferPtr, sizeof(CellExtractedHandles*),
                                   #ifdef SCALFMM_STARPU_USE_PRIO
                                                           STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                                   #endif
                                                           STARPU_R, cellHandles[idxLevel][interactionid].symb,
                                                           STARPU_R, cellHandles[idxLevel][interactionid].up,
                                                           STARPU_RW, interactionBuffer.all, 0);

                                    // Move to a new memory block that is on the same node as A
                                    // B' to B'''
                                    duplicatedCellBuffer.emplace_back();
                                    DuplicatedCellHandle& duplicateB = duplicatedCellBuffer.back();
                                    duplicateB.sizeSymb = tree->getCellGroup(idxLevel,interactionid)->getBufferSizeInByte();
                                    duplicateB.sizeOther = tree->getCellGroup(idxLevel,interactionid)->getMultipoleBufferSizeInByte();
                                    if(starpu_mpi_data_get_rank(cellHandles[idxLevel][idxGroup].symb) == mpi_rank){
                                        // Reuse block but just to perform the send
                                        duplicateB.dataSymbPtr.reset(new unsigned char[duplicateB.sizeSymb]);// = const_cast<unsigned char*>(tree->getCellGroup(idxLevel,interactionid)->getRawBuffer());
                                        duplicateB.dataOtherPtr.reset(new unsigned char[duplicateB.sizeOther]);// = reinterpret_cast<unsigned char*>(tree->getCellGroup(idxLevel,interactionid)->getRawMultipoleBuffer());
                                    }
                                    duplicateB.dataSymb = nullptr;
                                    duplicateB.dataOther = nullptr;

                                    registeringNode = starpu_mpi_data_get_rank(cellHandles[idxLevel][idxGroup].symb);
                                    where = (registeringNode == mpi_rank) ? STARPU_MAIN_RAM : -1;
                                    starpu_variable_data_register(&duplicateB.symb, where,
                                                                  (uintptr_t)duplicateB.dataSymbPtr.get(), duplicateB.sizeSymb);
                                    starpu_mpi_data_register(duplicateB.symb, tag++, registeringNode);
                                    starpu_variable_data_register(&duplicateB.other, where,
                                                                  (uintptr_t)duplicateB.dataOtherPtr.get(), duplicateB.sizeOther);
                                    starpu_mpi_data_register(duplicateB.other, tag++, registeringNode);

                                    const unsigned char* ptr1 = const_cast<unsigned char*>(tree->getCellGroup(idxLevel,interactionid)->getRawBuffer());
                                    size_t size1 = duplicateB.sizeSymb;
                                    const unsigned char* ptr2 = reinterpret_cast<unsigned char*>(tree->getCellGroup(idxLevel,interactionid)->getRawMultipoleBuffer());
                                    size_t size2 = duplicateB.sizeOther;

                                    starpu_mpi_insert_task(MPI_COMM_WORLD,
                                                           &cell_insert_up_bis,
                                                           STARPU_VALUE, &interactionBufferPtr, sizeof(CellExtractedHandles*),
                                                           STARPU_VALUE, &ptr1, sizeof(ptr1),
                                                           STARPU_VALUE, &size1, sizeof(size1),
                                                           STARPU_VALUE, &ptr2, sizeof(ptr2),
                                                           STARPU_VALUE, &size2, sizeof(size2),
                                   #ifdef SCALFMM_STARPU_USE_PRIO
                                                           STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                                   #endif
                                                           STARPU_R, interactionBuffer.all,
                                                           STARPU_RW, duplicateB.symb,
                                                           STARPU_RW, duplicateB.other, 0);


                                int mode = 1;
                                starpu_mpi_insert_task(MPI_COMM_WORLD,
                                                       &m2l_cl_inout,
                                                       STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                                       STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                                       STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                                       STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                                                       STARPU_VALUE, &mode, sizeof(int),
                               #ifdef SCALFMM_STARPU_USE_PRIO
                                                       STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                               #endif
                                                       STARPU_R, cellHandles[idxLevel][idxGroup].symb,
                                                       (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][idxGroup].down,
                                                       STARPU_R, duplicateB.symb,
                                                       STARPU_R, duplicateB.other,
                               #ifdef STARPU_USE_TASK_NAME
                               #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                                       STARPU_NAME, m2lOuterTaskNames[idxLevel].get(),
                               #else
                                                       //"M2L_out-l_nb_i_nb_i_s
                                                       STARPU_NAME, taskNames->print("M2L_out", "%d, %d, %lld, %d, %lld, %d, %lld, %lld, %lld, %lld, %d\n",
                                                                                     idxLevel,
                                                                                     tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                                                     tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                                                     tree->getCellGroup(idxLevel,interactionid)->getNumberOfCellsInBlock(),
                                                                                     tree->getCellGroup(idxLevel,interactionid)->getSizeOfInterval(),
                                                                                     outsideInteractions->size(),
                                                                                     tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                                                     tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                                                     tree->getCellGroup(idxLevel, interactionid)->getStartingIndex(),
                                                                                     tree->getCellGroup(idxLevel, interactionid)->getEndingIndex(),
                                                                                     starpu_mpi_data_get_rank(cellHandles[idxLevel][idxGroup].down)),
                               #endif
                               #endif
                                                       0);
                            }
                            {
                                // Extract data from second group for the first one
                                // That is copy A to A'
                                extractedCellBuffer.emplace_back();
                                CellExtractedHandles& interactionBuffer = extractedCellBuffer.back();
                                interactionBuffer.cellsToExtract = externalInteractionsAllLevelInnerIndexes[idxLevel][idxGroup][idxInteraction];
                                interactionBuffer.size = tree->getCellGroup(idxLevel,idxGroup)->extractGetSizeSymbUp(interactionBuffer.cellsToExtract);
                                // I allocate only if I will use it to extract
                                if(starpu_mpi_data_get_rank(cellHandles[idxLevel][idxGroup].symb) == mpi_rank){
                                    interactionBuffer.data.reset(new unsigned char[interactionBuffer.size]);
                                }
                                else{
                                    interactionBuffer.data.reset(nullptr);
                                }
                                int registeringNode = starpu_mpi_data_get_rank(cellHandles[idxLevel][idxGroup].symb);
                                int where = (registeringNode == mpi_rank) ? STARPU_MAIN_RAM : -1;
                                starpu_variable_data_register(&interactionBuffer.all, where,
                                                              (uintptr_t)interactionBuffer.data.get(), interactionBuffer.size);
                                starpu_mpi_data_register(interactionBuffer.all, tag++, registeringNode);

                                CellExtractedHandles* interactionBufferPtr = &interactionBuffer;
                                starpu_mpi_insert_task(MPI_COMM_WORLD,
                                                       &cell_extract_up,
                                                       STARPU_VALUE, &interactionBufferPtr, sizeof(CellExtractedHandles*),
                               #ifdef SCALFMM_STARPU_USE_PRIO
                                                       STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                               #endif
                                                       STARPU_R, cellHandles[idxLevel][idxGroup].symb,
                                                       STARPU_R, cellHandles[idxLevel][idxGroup].up,
                                                       STARPU_RW, interactionBuffer.all, 0);

                                // Move to a new memory block that is on the same node as A
                                // B' to B'''
                                duplicatedCellBuffer.emplace_back();
                                DuplicatedCellHandle& duplicateB = duplicatedCellBuffer.back();
                                duplicateB.sizeSymb = tree->getCellGroup(idxLevel,idxGroup)->getBufferSizeInByte();
                                duplicateB.sizeOther = tree->getCellGroup(idxLevel,idxGroup)->getMultipoleBufferSizeInByte();
                                if(starpu_mpi_data_get_rank(cellHandles[idxLevel][interactionid].symb) == mpi_rank){
                                    // Reuse block but just to perform the send
                                    duplicateB.dataSymbPtr.reset(new unsigned char[duplicateB.sizeSymb]);//const_cast<unsigned char*>(tree->getCellGroup(idxLevel,idxGroup)->getRawBuffer());
                                    //memcpy(duplicateB.dataSymbPtr.get(), tree->getCellGroup(idxLevel,idxGroup)->getRawBuffer(), duplicateB.sizeSymb);
                                    duplicateB.dataOtherPtr.reset(new unsigned char[duplicateB.sizeOther]);//reinterpret_cast<unsigned char*>(tree->getCellGroup(idxLevel,idxGroup)->getRawMultipoleBuffer());
                                    //memcpy(duplicateB.dataOtherPtr.get(), tree->getCellGroup(idxLevel,idxGroup)->getRawMultipoleBuffer(), duplicateB.sizeOther);
                                }
                                duplicateB.dataSymb = nullptr;
                                duplicateB.dataOther = nullptr;

                                registeringNode = starpu_mpi_data_get_rank(cellHandles[idxLevel][interactionid].symb);
                                where = (registeringNode == mpi_rank) ? STARPU_MAIN_RAM : -1;
                                starpu_variable_data_register(&duplicateB.symb, where,
                                                              (uintptr_t)duplicateB.dataSymbPtr.get(), duplicateB.sizeSymb);
                                starpu_mpi_data_register(duplicateB.symb, tag++, registeringNode);
                                starpu_variable_data_register(&duplicateB.other, where,
                                                              (uintptr_t)duplicateB.dataOtherPtr.get(), duplicateB.sizeOther);
                                starpu_mpi_data_register(duplicateB.other, tag++, registeringNode);

                                const unsigned char* ptr1 = const_cast<unsigned char*>(tree->getCellGroup(idxLevel,idxGroup)->getRawBuffer());
                                size_t size1 = duplicateB.sizeSymb;
                                const unsigned char* ptr2 = reinterpret_cast<unsigned char*>(tree->getCellGroup(idxLevel,idxGroup)->getRawMultipoleBuffer());
                                size_t size2 = duplicateB.sizeOther;
                                starpu_mpi_insert_task(MPI_COMM_WORLD,
                                                       &cell_insert_up_bis,
                                                       STARPU_VALUE, &interactionBufferPtr, sizeof(CellExtractedHandles*),
                                                       STARPU_VALUE, &ptr1, sizeof(ptr1),
                                                       STARPU_VALUE, &size1, sizeof(size1),
                                                       STARPU_VALUE, &ptr2, sizeof(ptr2),
                                                       STARPU_VALUE, &size2, sizeof(size2),
                               #ifdef SCALFMM_STARPU_USE_PRIO
                                                       STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                               #endif
                                                       STARPU_R, interactionBuffer.all,
                                                       STARPU_RW, duplicateB.symb,
                                                       STARPU_RW, duplicateB.other, 0);

                                int mode = 2;
                                starpu_mpi_insert_task(MPI_COMM_WORLD,
                                                       &m2l_cl_inout,
                                                       STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                                       STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                                       STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                                       STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                                                       STARPU_VALUE, &mode, sizeof(int),
                               #ifdef SCALFMM_STARPU_USE_PRIO
                                                       STARPU_PRIORITY, PrioClass::Controller().getInsertionPosM2LExtern(idxLevel),
                               #endif
                                                       STARPU_R, cellHandles[idxLevel][interactionid].symb,
                                                       (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel][interactionid].down,
                                                       STARPU_R, duplicateB.symb,
                                                       STARPU_R, duplicateB.other,
                               #ifdef STARPU_USE_TASK_NAME
                               #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                                       STARPU_NAME, m2lOuterTaskNames[idxLevel].get(),
                               #else
                                                       //"M2L_out-l_nb_i_nb_i_s"
                                                       STARPU_NAME, taskNames->print("M2L_out", "%d, %d, %lld, %d, %lld, %d, %lld, %lld, %lld, %lld, %d\n",
                                                                                     idxLevel,
                                                                                     tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                                                     tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                                                     tree->getCellGroup(idxLevel,interactionid)->getNumberOfCellsInBlock(),
                                                                                     tree->getCellGroup(idxLevel,interactionid)->getSizeOfInterval(),
                                                                                     outsideInteractions->size(),
                                                                                     tree->getCellGroup(idxLevel, interactionid)->getStartingIndex(),
                                                                                     tree->getCellGroup(idxLevel, interactionid)->getEndingIndex(),
                                                                                     tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                                                     tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                                                     starpu_mpi_data_get_rank(cellHandles[idxLevel][interactionid].down)),
                               #endif
                               #endif
                                                       0);
                            }
                        }
#endif
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
                    // put the right codelet
                    if((noCommuteAtLastLevel && (idxLevel == FAbstractAlgorithm::lowerWorkingLevel - 2)) || noCommuteBetweenLevel){
                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                               &l2l_cl_nocommute,
                                               STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                               STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                               STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosL2L(idxLevel),
                       #endif
                                               STARPU_R, cellHandles[idxLevel][idxGroup].symb, //symbolique, readonly
                                               STARPU_R, cellHandles[idxLevel][idxGroup].down, //The remaining, read/write
                                               STARPU_R, cellHandles[idxLevel+1][idxSubGroup].symb, //symbolique, readonly
                                STARPU_RW, cellHandles[idxLevel+1][idxSubGroup].down, //level d'avant readonly
        #ifdef STARPU_USE_TASK_NAME
        #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                STARPU_NAME, l2lTaskNames[idxLevel].get(),
        #else
                                //"L2L-l_nb_i_nbc_ic_s"
                                STARPU_NAME, taskNames->print("L2L", "%d, %d, %lld, %d, %lld, %lld, %lld, %lld, %lld, %lld, %d\n",
                                                              idxLevel,
                                                              tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                              tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                              tree->getCellGroup(idxLevel+1,idxSubGroup)->getNumberOfCellsInBlock(),
                                                              tree->getCellGroup(idxLevel+1,idxSubGroup)->getSizeOfInterval(),
                                                              FMath::Min(tree->getCellGroup(idxLevel,idxGroup)->getEndingIndex()-1, (tree->getCellGroup(idxLevel+1,idxSubGroup)->getEndingIndex()-1)>>3)-
                                                              FMath::Max(tree->getCellGroup(idxLevel,idxGroup)->getStartingIndex(), tree->getCellGroup(idxLevel+1,idxSubGroup)->getStartingIndex()>>3),
                                                              tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                              tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                              tree->getCellGroup(idxLevel+1, idxSubGroup)->getStartingIndex(),
                                                              tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex(),
                                                              starpu_mpi_data_get_rank(cellHandles[idxLevel+1][idxSubGroup].down)),
        #endif
        #endif
                                0);
                    }
                    else{
                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                               &l2l_cl,
                                               STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                               STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                               STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosL2L(idxLevel),
                       #endif
                                               STARPU_R, cellHandles[idxLevel][idxGroup].symb, //symbolique, readonly
                                               STARPU_R, cellHandles[idxLevel][idxGroup].down, //The remaining, read/write
                                               STARPU_R, cellHandles[idxLevel+1][idxSubGroup].symb, //symbolique, readonly
                                (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel+1][idxSubGroup].down, //level d'avant readonly
        #ifdef STARPU_USE_TASK_NAME
        #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                STARPU_NAME, l2lTaskNames[idxLevel].get(),
        #else
                                //"L2L-l_nb_i_nbc_ic_s"
                                STARPU_NAME, taskNames->print("L2L", "%d, %d, %lld, %d, %lld, %lld, %lld, %lld, %lld, %lld, %d\n",
                                                              idxLevel,
                                                              tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                              tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                              tree->getCellGroup(idxLevel+1,idxSubGroup)->getNumberOfCellsInBlock(),
                                                              tree->getCellGroup(idxLevel+1,idxSubGroup)->getSizeOfInterval(),
                                                              FMath::Min(tree->getCellGroup(idxLevel,idxGroup)->getEndingIndex()-1, (tree->getCellGroup(idxLevel+1,idxSubGroup)->getEndingIndex()-1)>>3)-
                                                              FMath::Max(tree->getCellGroup(idxLevel,idxGroup)->getStartingIndex(), tree->getCellGroup(idxLevel+1,idxSubGroup)->getStartingIndex()>>3),
                                                              tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                              tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                              tree->getCellGroup(idxLevel+1, idxSubGroup)->getStartingIndex(),
                                                              tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex(),
                                                              starpu_mpi_data_get_rank(cellHandles[idxLevel+1][idxSubGroup].down)),
        #endif
        #endif
                                0);
                    }

                }
                while(tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex() <= (((currentCells->getEndingIndex()-1)<<3)+7)
                      && (idxSubGroup+1) != tree->getNbCellGroupAtLevel(idxLevel+1)
                      && tree->getCellGroup(idxLevel+1, idxSubGroup+1)->getStartingIndex() <= ((currentCells->getEndingIndex()-1)<<3)+7 ){
                    idxSubGroup += 1;

                    // put the right codelet
                    if((noCommuteAtLastLevel && (idxLevel == FAbstractAlgorithm::lowerWorkingLevel - 2)) || noCommuteBetweenLevel){
                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                               &l2l_cl_nocommute,
                                               STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                               STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                               STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosL2L(idxLevel),
                       #endif
                                               STARPU_R, cellHandles[idxLevel][idxGroup].symb, //symbolique, readonly
                                               STARPU_R, cellHandles[idxLevel][idxGroup].down, //The remaining, read/write
                                               STARPU_R, cellHandles[idxLevel+1][idxSubGroup].symb, //symbolique, readonly
                                STARPU_RW, cellHandles[idxLevel+1][idxSubGroup].down, //level d'avant readonly
        #ifdef STARPU_USE_TASK_NAME
        #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                STARPU_NAME, l2lTaskNames[idxLevel].get(),
        #else
                                //"L2L-l_nb_i_nbc_ic_s"
                                STARPU_NAME, taskNames->print("L2L", "%d, %d, %lld, %d, %lld, %lld, %lld, %lld, %lld, %lld, %d\n",
                                                              idxLevel,
                                                              tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                              tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                              tree->getCellGroup(idxLevel+1,idxSubGroup)->getNumberOfCellsInBlock(),
                                                              tree->getCellGroup(idxLevel+1,idxSubGroup)->getSizeOfInterval(),
                                                              FMath::Min(tree->getCellGroup(idxLevel,idxGroup)->getEndingIndex()-1, (tree->getCellGroup(idxLevel+1,idxSubGroup)->getEndingIndex()-1)>>3)-
                                                              FMath::Max(tree->getCellGroup(idxLevel,idxGroup)->getStartingIndex(), tree->getCellGroup(idxLevel+1,idxSubGroup)->getStartingIndex()>>3),
                                                              tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                              tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                              tree->getCellGroup(idxLevel+1, idxSubGroup)->getStartingIndex(),
                                                              tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex(),
                                                              starpu_mpi_data_get_rank(cellHandles[idxLevel+1][idxSubGroup].down)),
        #endif
        #endif
                                0);
                    }
                    else{
                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                               &l2l_cl,
                                               STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                               STARPU_VALUE, &idxLevel, sizeof(idxLevel),
                                               STARPU_VALUE, &cellHandles[idxLevel][idxGroup].intervalSize, sizeof(int),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosL2L(idxLevel),
                       #endif
                                               STARPU_R, cellHandles[idxLevel][idxGroup].symb, //symbolique, readonly
                                               STARPU_R, cellHandles[idxLevel][idxGroup].down, //The remaining, read/write
                                               STARPU_R, cellHandles[idxLevel+1][idxSubGroup].symb, //symbolique, readonly
                                (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), cellHandles[idxLevel+1][idxSubGroup].down, //level d'avant readonly
        #ifdef STARPU_USE_TASK_NAME
        #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                STARPU_NAME, l2lTaskNames[idxLevel].get(),
        #else
                                //"L2L-l_nb_i_nbc_ic_s"
                                STARPU_NAME, taskNames->print("L2L", "%d, %d, %lld, %d, %lld, %lld, %lld, %lld, %lld, %lld, %d\n",
                                                              idxLevel,
                                                              tree->getCellGroup(idxLevel,idxGroup)->getNumberOfCellsInBlock(),
                                                              tree->getCellGroup(idxLevel,idxGroup)->getSizeOfInterval(),
                                                              tree->getCellGroup(idxLevel+1,idxSubGroup)->getNumberOfCellsInBlock(),
                                                              tree->getCellGroup(idxLevel+1,idxSubGroup)->getSizeOfInterval(),
                                                              FMath::Min(tree->getCellGroup(idxLevel,idxGroup)->getEndingIndex()-1, (tree->getCellGroup(idxLevel+1,idxSubGroup)->getEndingIndex()-1)>>3)-
                                                              FMath::Max(tree->getCellGroup(idxLevel,idxGroup)->getStartingIndex(), tree->getCellGroup(idxLevel+1,idxSubGroup)->getStartingIndex()>>3),
                                                              tree->getCellGroup(idxLevel, idxGroup)->getStartingIndex(),
                                                              tree->getCellGroup(idxLevel, idxGroup)->getEndingIndex(),
                                                              tree->getCellGroup(idxLevel+1, idxSubGroup)->getStartingIndex(),
                                                              tree->getCellGroup(idxLevel+1, idxSubGroup)->getEndingIndex(),
                                                              starpu_mpi_data_get_rank(cellHandles[idxLevel+1][idxSubGroup].down)),
        #endif
        #endif
                                0);
                    }
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
                if(starpu_mpi_data_get_rank(particleHandles[idxGroup].down) == starpu_mpi_data_get_rank(particleHandles[interactionid].down))
                {
                    starpu_mpi_insert_task(MPI_COMM_WORLD,
                                           &p2p_cl_inout,
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
                                           (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), particleHandles[idxGroup].down,
                       #endif
                                           STARPU_R, particleHandles[interactionid].symb,
                       #ifdef STARPU_USE_REDUX
                                           STARPU_REDUX, particleHandles[interactionid].down,
                       #else
                                           (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), particleHandles[interactionid].down,
                                           STARPU_EXECUTE_ON_DATA, particleHandles[interactionid].down,
                       #endif
                       #ifdef STARPU_USE_TASK_NAME
                       #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                           STARPU_NAME, p2pOuterTaskNames.get(),
                       #else
                                           //"P2P_out-nb_i_p_nb_i_p_s"
                                           STARPU_NAME, taskNames->print("P2P_out", "%d, %lld, %lld, %d, %lld, %lld, %d, %lld, %lld, %lld, %lld, %d\n",
                                                                         tree->getParticleGroup(idxGroup)->getNumberOfLeavesInBlock(),
                                                                         tree->getParticleGroup(idxGroup)->getSizeOfInterval(),
                                                                         tree->getParticleGroup(idxGroup)->getNbParticlesInGroup(),
                                                                         tree->getParticleGroup(interactionid)->getNumberOfLeavesInBlock(),
                                                                         tree->getParticleGroup(interactionid)->getSizeOfInterval(),
                                                                         tree->getParticleGroup(interactionid)->getNbParticlesInGroup(),
                                                                         outsideInteractions->size(),
                                                                         tree->getParticleGroup(idxGroup)->getStartingIndex(),
                                                                         tree->getParticleGroup(idxGroup)->getEndingIndex(),
                                                                         tree->getParticleGroup(interactionid)->getStartingIndex(),
                                                                         tree->getParticleGroup(interactionid)->getEndingIndex(),
                                                                         starpu_mpi_data_get_rank(particleHandles[interactionid].down)),
                       #endif
                       #endif
                                           0);
                }
                else
                {

#ifdef SCALFMM_USE_STARPU_EXTRACT
                    {
                        // Extract data from second group for the first one
                        // That is copy B to B'
                        extractedParticlesBuffer.emplace_back();
                        ParticleExtractedHandles& interactionBuffer = extractedParticlesBuffer.back();
                        interactionBuffer.leavesToExtract = externalInteractionsLeafLevelOuter[idxGroup][idxInteraction];

                        interactionBuffer.size = tree->getParticleGroup(interactionid)->getExtractBufferSize(interactionBuffer.leavesToExtract);
                        // I allocate only if I will use it to extract
                        if(starpu_mpi_data_get_rank(particleHandles[interactionid].symb) == mpi_rank){
                            interactionBuffer.data.reset(new unsigned char[interactionBuffer.size]);
                        }
                        else{
                            interactionBuffer.data.reset(nullptr);
                        }

                        int registeringNode = starpu_mpi_data_get_rank(particleHandles[interactionid].symb);
                        int where = (registeringNode == mpi_rank) ? STARPU_MAIN_RAM : -1;
                        starpu_variable_data_register(&interactionBuffer.symb, where,
                                                      (uintptr_t)interactionBuffer.data.get(), interactionBuffer.size);
                        starpu_mpi_data_register(interactionBuffer.symb, tag++, registeringNode);

                        ParticleExtractedHandles* interactionBufferPtr = &interactionBuffer;
                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                               &p2p_extract,
                                               STARPU_VALUE, &interactionBufferPtr, sizeof(ParticleExtractedHandles*),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2PExtern(),
                       #endif
                                               STARPU_R, particleHandles[interactionid].symb,
                                               STARPU_RW, interactionBuffer.symb, 0);

                        // Move to a new memory block that is on the same node as A
                        // B' to B'''
                        duplicatedParticlesBuffer.emplace_back();
                        DuplicatedParticlesHandle& duplicateB = duplicatedParticlesBuffer.back();
                        duplicateB.size = tree->getParticleGroup(interactionid)->getBufferSizeInByte();
                        if(starpu_mpi_data_get_rank(particleHandles[idxGroup].symb) == mpi_rank){
                            // Reuse block but just to perform the send
                            duplicateB.data = (unsigned char*) FAlignedMemory::AllocateBytes<64>(duplicateB.size);// = const_cast<unsigned char*>(tree->getParticleGroup(interactionid)->getRawBuffer());
                        }
                        else{
                            duplicateB.data = nullptr;
                        }

                        registeringNode = starpu_mpi_data_get_rank(particleHandles[idxGroup].symb);
                        where = (registeringNode == mpi_rank) ? STARPU_MAIN_RAM : -1;
                        starpu_variable_data_register(&duplicateB.symb, where,
                                                      (uintptr_t)duplicateB.data, duplicateB.size);
                        starpu_mpi_data_register(duplicateB.symb, tag++, registeringNode);

                        const unsigned char* dataPtr = const_cast<unsigned char*>(tree->getParticleGroup(interactionid)->getRawBuffer());
                        size_t sizeData = duplicateB.size;

                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                               &p2p_insert_bis,
                                               STARPU_VALUE, &interactionBufferPtr, sizeof(ParticleExtractedHandles*),
                                               STARPU_VALUE, &dataPtr, sizeof(dataPtr),
                                               STARPU_VALUE, &sizeData, sizeof(sizeData),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2PExtern(),
                       #endif
                                               STARPU_R, interactionBuffer.symb,
                                               STARPU_RW, duplicateB.symb,
                                               0);

                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                               &p2p_cl_inout_mpi,
                                               STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                               STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                               STARPU_VALUE, &particleHandles[idxGroup].intervalSize, sizeof(int),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2PExtern(),
                       #endif
                                               STARPU_R, particleHandles[idxGroup].symb,
                                               (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), particleHandles[idxGroup].down,
                                               STARPU_R, duplicateB.symb,
                       #ifdef STARPU_USE_TASK_NAME
                       #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                               STARPU_NAME, p2pOuterTaskNames.get(),
                       #else
                                               //"P2P_out-nb_i_p_nb_i_p_s"
                                               STARPU_NAME, taskNames->print("P2P_out", "%d, %lld, %lld, %d, %lld, %lld, %d, %lld, %lld, %lld, %lld, %d\n",
                                                                             tree->getParticleGroup(idxGroup)->getNumberOfLeavesInBlock(),
                                                                             tree->getParticleGroup(idxGroup)->getSizeOfInterval(),
                                                                             tree->getParticleGroup(idxGroup)->getNbParticlesInGroup(),
                                                                             tree->getParticleGroup(interactionid)->getNumberOfLeavesInBlock(),
                                                                             tree->getParticleGroup(interactionid)->getSizeOfInterval(),
                                                                             tree->getParticleGroup(interactionid)->getNbParticlesInGroup(),
                                                                             outsideInteractions->size(),
                                                                             tree->getParticleGroup(idxGroup)->getStartingIndex(),
                                                                             tree->getParticleGroup(idxGroup)->getEndingIndex(),
                                                                             tree->getParticleGroup(interactionid)->getStartingIndex(),
                                                                             tree->getParticleGroup(interactionid)->getEndingIndex(),
                                                                             starpu_mpi_data_get_rank(particleHandles[idxGroup].down)),
                       #endif
                       #endif
                                               0);
                    }
                    {
                        std::vector<OutOfBlockInteraction>* outsideInteractionsOpposite = new std::vector<OutOfBlockInteraction>(externalInteractionsLeafLevel[idxGroup][idxInteraction].interactions);
                        for(unsigned int i = 0; i < outsideInteractionsOpposite->size(); ++i)
                        {
                            MortonIndex tmp = outsideInteractionsOpposite->at(i).outIndex;
                            outsideInteractionsOpposite->at(i).outIndex = outsideInteractionsOpposite->at(i).insideIndex;
                            outsideInteractionsOpposite->at(i).insideIndex = tmp;
                            int tmp2 = outsideInteractionsOpposite->at(i).insideIdxInBlock;
                            outsideInteractionsOpposite->at(i).insideIdxInBlock = outsideInteractionsOpposite->at(i).outsideIdxInBlock;
                            outsideInteractionsOpposite->at(i).outsideIdxInBlock = tmp2;
                            outsideInteractionsOpposite->at(i).relativeOutPosition = getOppositeInterIndex(outsideInteractionsOpposite->at(i).relativeOutPosition);
                        }
                        externalInteractionsLeafLevelOpposite.push_front(outsideInteractionsOpposite);


                        // Extract data from second group for the first one
                        // That is copy A to A'
                        extractedParticlesBuffer.emplace_back();
                        ParticleExtractedHandles& interactionBuffer = extractedParticlesBuffer.back();
                        interactionBuffer.leavesToExtract = externalInteractionsLeafLevelInner[idxGroup][idxInteraction];

                        interactionBuffer.size = tree->getParticleGroup(idxGroup)->getExtractBufferSize(interactionBuffer.leavesToExtract);
                        // I allocate only if I will use it to extract
                        if(starpu_mpi_data_get_rank(particleHandles[idxGroup].down) == mpi_rank){
                            interactionBuffer.data.reset(new unsigned char[interactionBuffer.size]);
                        }
                        else{
                            interactionBuffer.data.reset(nullptr);
                        }

                        int registeringNode = starpu_mpi_data_get_rank(particleHandles[idxGroup].down);
                        int where = (registeringNode == mpi_rank) ? STARPU_MAIN_RAM : -1;
                        starpu_variable_data_register(&interactionBuffer.symb, where,
                                                      (uintptr_t)interactionBuffer.data.get(), interactionBuffer.size);
                        starpu_mpi_data_register(interactionBuffer.symb, tag++, registeringNode);

                        ParticleExtractedHandles* interactionBufferPtr = &interactionBuffer;
                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                               &p2p_extract,
                                               STARPU_VALUE, &interactionBufferPtr, sizeof(ParticleExtractedHandles*),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2PExtern(),
                       #endif
                                               STARPU_R, particleHandles[idxGroup].symb,
                                               STARPU_RW, interactionBuffer.symb, 0);

                        // Move to a new memory block that is on the same node as A
                        // B' to B'''
                        duplicatedParticlesBuffer.emplace_back();
                        DuplicatedParticlesHandle& duplicateA = duplicatedParticlesBuffer.back();
                        duplicateA.size = tree->getParticleGroup(idxGroup)->getBufferSizeInByte();
                        if(starpu_mpi_data_get_rank(particleHandles[interactionid].down) == mpi_rank){
                            // Reuse block but just to perform the send
                            duplicateA.data = (unsigned char*) FAlignedMemory::AllocateBytes<64>(duplicateA.size);// = const_cast<unsigned char*>(tree->getParticleGroup(idxGroup)->getRawBuffer());
                        }
                        else{
                            duplicateA.data = nullptr;
                        }

                        registeringNode = starpu_mpi_data_get_rank(particleHandles[interactionid].down);
                        where = (registeringNode == mpi_rank) ? STARPU_MAIN_RAM : -1;
                        starpu_variable_data_register(&duplicateA.symb, where,
                                                      (uintptr_t)duplicateA.data, duplicateA.size);
                        starpu_mpi_data_register(duplicateA.symb, tag++, registeringNode);

                        const unsigned char* dataPtr = const_cast<unsigned char*>(tree->getParticleGroup(idxGroup)->getRawBuffer());
                        size_t sizeData = duplicateA.size;

                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                               &p2p_insert_bis,
                                               STARPU_VALUE, &interactionBufferPtr, sizeof(ParticleExtractedHandles*),
                                               STARPU_VALUE, &dataPtr, sizeof(dataPtr),
                                               STARPU_VALUE, &sizeData, sizeof(sizeData),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2PExtern(),
                       #endif
                                               STARPU_R, interactionBuffer.symb,
                                               STARPU_RW, duplicateA.symb, 0);


                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                               &p2p_cl_inout_mpi,
                                               STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                               STARPU_VALUE, &outsideInteractionsOpposite, sizeof(outsideInteractionsOpposite),
                                               STARPU_VALUE, &particleHandles[idxGroup].intervalSize, sizeof(int),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                               STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2PExtern(),
                       #endif
                                               STARPU_R, particleHandles[interactionid].symb,
                                               (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), particleHandles[interactionid].down,
                                               STARPU_R, duplicateA.symb,
                       #ifdef STARPU_USE_TASK_NAME
                       #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                               STARPU_NAME, p2pOuterTaskNames.get(),
                       #else
                                               //"P2P_out-nb_i_p_nb_i_p_s"
                                               STARPU_NAME, taskNames->print("P2P_out", "%d, %lld, %lld, %d, %lld, %lld, %d, %lld, %lld, %lld, %lld, %d\n",
                                                                             tree->getParticleGroup(interactionid)->getNumberOfLeavesInBlock(),
                                                                             tree->getParticleGroup(interactionid)->getSizeOfInterval(),
                                                                             tree->getParticleGroup(interactionid)->getNbParticlesInGroup(),
                                                                             tree->getParticleGroup(idxGroup)->getNumberOfLeavesInBlock(),
                                                                             tree->getParticleGroup(idxGroup)->getSizeOfInterval(),
                                                                             tree->getParticleGroup(idxGroup)->getNbParticlesInGroup(),
                                                                             outsideInteractions->size(),
                                                                             tree->getParticleGroup(interactionid)->getStartingIndex(),
                                                                             tree->getParticleGroup(interactionid)->getEndingIndex(),
                                                                             tree->getParticleGroup(idxGroup)->getStartingIndex(),
                                                                             tree->getParticleGroup(idxGroup)->getEndingIndex(),
                                                                             starpu_mpi_data_get_rank(particleHandles[interactionid].down)),
                       #endif
                       #endif
                                               0);
                    }
#else
                    {
                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                           &p2p_cl_inout_mpi,
                                           STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                           STARPU_VALUE, &outsideInteractions, sizeof(outsideInteractions),
                                           STARPU_VALUE, &particleHandles[idxGroup].intervalSize, sizeof(int),
                           #ifdef SCALFMM_STARPU_USE_PRIO
                                           STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2PExtern(),
                           #endif
                                           STARPU_R, particleHandles[idxGroup].symb,
                                           (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), particleHandles[idxGroup].down,
                                           STARPU_R, particleHandles[interactionid].symb,
                           #ifdef STARPU_USE_TASK_NAME
                           #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                           STARPU_NAME, p2pOuterTaskNames.get(),
                           #else
                                           //"P2P_out-nb_i_p_nb_i_p_s"
                                           STARPU_NAME, taskNames->print("P2P_out", "%d, %lld, %lld, %d, %lld, %lld, %d, %lld, %lld, %lld, %lld, %d\n",
                                                        tree->getParticleGroup(idxGroup)->getNumberOfLeavesInBlock(),
                                                        tree->getParticleGroup(idxGroup)->getSizeOfInterval(),
                                                        tree->getParticleGroup(idxGroup)->getNbParticlesInGroup(),
                                                        tree->getParticleGroup(interactionid)->getNumberOfLeavesInBlock(),
                                                        tree->getParticleGroup(interactionid)->getSizeOfInterval(),
                                                        tree->getParticleGroup(interactionid)->getNbParticlesInGroup(),
                                                        outsideInteractions->size(),
                                                        tree->getParticleGroup(idxGroup)->getStartingIndex(),
                                                        tree->getParticleGroup(idxGroup)->getEndingIndex(),
                                                        tree->getParticleGroup(interactionid)->getStartingIndex(),
                                                        tree->getParticleGroup(interactionid)->getEndingIndex(),
                                                        starpu_mpi_data_get_rank(particleHandles[idxGroup].down)),
                           #endif
                           #endif
                                           0);
                        std::vector<OutOfBlockInteraction>* outsideInteractionsOpposite = new std::vector<OutOfBlockInteraction>(externalInteractionsLeafLevel[idxGroup][idxInteraction].interactions);
                        for(unsigned int i = 0; i < outsideInteractionsOpposite->size(); ++i)
                        {
                            MortonIndex tmp = outsideInteractionsOpposite->at(i).outIndex;
                            outsideInteractionsOpposite->at(i).outIndex = outsideInteractionsOpposite->at(i).insideIndex;
                            outsideInteractionsOpposite->at(i).insideIndex = tmp;
                            int tmp2 = outsideInteractionsOpposite->at(i).insideIdxInBlock;
                            outsideInteractionsOpposite->at(i).insideIdxInBlock = outsideInteractionsOpposite->at(i).outsideIdxInBlock;
                            outsideInteractionsOpposite->at(i).outsideIdxInBlock = tmp2;
                            outsideInteractionsOpposite->at(i).relativeOutPosition = getOppositeInterIndex(outsideInteractionsOpposite->at(i).relativeOutPosition);
                        }
                        externalInteractionsLeafLevelOpposite.push_front(outsideInteractionsOpposite);
                        starpu_mpi_insert_task(MPI_COMM_WORLD,
                                           &p2p_cl_inout_mpi,
                                           STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                           STARPU_VALUE, &outsideInteractionsOpposite, sizeof(outsideInteractionsOpposite),
                                           STARPU_VALUE, &particleHandles[idxGroup].intervalSize, sizeof(int),
                           #ifdef SCALFMM_STARPU_USE_PRIO
                                           STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2PExtern(),
                           #endif
                                           STARPU_R, particleHandles[interactionid].symb,
                                           (STARPU_RW|STARPU_COMMUTE_IF_SUPPORTED), particleHandles[interactionid].down,
                                           STARPU_R, particleHandles[idxGroup].symb,
                           #ifdef STARPU_USE_TASK_NAME
                           #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                           STARPU_NAME, p2pOuterTaskNames.get(),
                           #else
                                           //"P2P_out-nb_i_p_nb_i_p_s"
                                           STARPU_NAME, taskNames->print("P2P_out", "%d, %lld, %lld, %d, %lld, %lld, %d, %lld, %lld, %lld, %lld, %d\n",
                                                        tree->getParticleGroup(interactionid)->getNumberOfLeavesInBlock(),
                                                        tree->getParticleGroup(interactionid)->getSizeOfInterval(),
                                                        tree->getParticleGroup(interactionid)->getNbParticlesInGroup(),
                                                        tree->getParticleGroup(idxGroup)->getNumberOfLeavesInBlock(),
                                                        tree->getParticleGroup(idxGroup)->getSizeOfInterval(),
                                                        tree->getParticleGroup(idxGroup)->getNbParticlesInGroup(),
                                                        outsideInteractions->size(),
                                                        tree->getParticleGroup(interactionid)->getStartingIndex(),
                                                        tree->getParticleGroup(interactionid)->getEndingIndex(),
                                                        tree->getParticleGroup(idxGroup)->getStartingIndex(),
                                                        tree->getParticleGroup(idxGroup)->getEndingIndex(),
                                                        starpu_mpi_data_get_rank(particleHandles[interactionid].down)),
                           #endif
                           #endif
                                           0);
                    }
#endif
                }
            }
        }
        FLOG( timerOutBlock.tac() );
        FLOG( timerInBlock.tic() );
        for(int idxGroup = 0 ; idxGroup < tree->getNbParticleGroup() ; ++idxGroup){
            starpu_mpi_insert_task(MPI_COMM_WORLD,
                                   &p2p_cl_in,
                                   STARPU_VALUE, &wrapperptr, sizeof(wrapperptr),
                                   STARPU_VALUE, &particleHandles[idxGroup].intervalSize, sizeof(int),
                       #ifdef SCALFMM_STARPU_USE_PRIO
                                   STARPU_PRIORITY, PrioClass::Controller().getInsertionPosP2P(),
                       #endif
                                   STARPU_R, particleHandles[idxGroup].symb,
                       #ifdef STARPU_USE_REDUX
                                   STARPU_REDUX, particleHandles[idxGroup].down,
                       #else
                                   (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), particleHandles[idxGroup].down,
                       #endif
                       #ifdef STARPU_USE_TASK_NAME
                       #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                                   STARPU_NAME, p2pTaskNames.get(),
                       #else
                                   //"P2P-nb_i_p"
                                   STARPU_NAME, taskNames->print("P2P", "%d, %lld, %lld, %lld, %lld, %lld, %lld, %d\n",
                                                                 tree->getParticleGroup(idxGroup)->getNumberOfLeavesInBlock(),
                                                                 tree->getParticleGroup(idxGroup)->getSizeOfInterval(),
                                                                 tree->getParticleGroup(idxGroup)->getNbParticlesInGroup(),
                                                                 tree->getParticleGroup(idxGroup)->getStartingIndex(),
                                                                 tree->getParticleGroup(idxGroup)->getEndingIndex(),
                                                                 tree->getParticleGroup(idxGroup)->getStartingIndex(),
                                                                 tree->getParticleGroup(idxGroup)->getEndingIndex(),
                                                                 starpu_mpi_data_get_rank(particleHandles[idxGroup].down)),
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
            starpu_mpi_insert_task(MPI_COMM_WORLD,
                                   &l2p_cl,
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
                    (STARPU_RW | STARPU_COMMUTE_IF_SUPPORTED), particleHandles[idxGroup].down,
        #endif
        #ifdef STARPU_USE_TASK_NAME
        #ifndef SCALFMM_SIMGRID_TASKNAMEPARAMS
                    STARPU_NAME, l2pTaskNames.get(),
        #else
                    //"L2P-nb_i_p"
                    STARPU_NAME, taskNames->print("L2P", "%d, %lld, %lld, %lld, %lld, %d\n",
                                                  tree->getCellGroup(tree->getHeight()-1,idxGroup)->getNumberOfCellsInBlock(),
                                                  tree->getCellGroup(tree->getHeight()-1,idxGroup)->getSizeOfInterval(),
                                                  tree->getCellGroup(tree->getHeight()-1,idxGroup)->getNumberOfCellsInBlock(),
                                                  tree->getParticleGroup(idxGroup)->getStartingIndex(),
                                                  tree->getParticleGroup(idxGroup)->getEndingIndex(),
                                                  starpu_mpi_data_get_rank(particleHandles[idxGroup].down)),
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
            starpu_mpi_insert_task(MPI_COMM_WORLD,
                                   &p2p_redux_read,
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
