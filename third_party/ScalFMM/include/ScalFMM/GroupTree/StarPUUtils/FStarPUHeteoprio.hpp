// See LICENCE file at project root

#ifndef FSTARPUHETEOPRIO_HPP
#define FSTARPUHETEOPRIO_HPP

/**
 * \brief Heteroprio
 *
 * \author Berenger Bramas (berenger.bramas@inria.fr)
 *
 * Heteroprio is a scheduler wich support different priorities for the
 * different architectures and with a critical end management based on
 * the speedup of one architecture over the others.
 * However the user needs to provide some information using a callback:
 * initialize_heteroprio_center_policy_callback
 * (this callbacj should set before any scheduler creation)
 * The scheduler proposes also prefetching and work stealing.
 *
 * The scheduler has HETEROPRIO_MAX_PRIO different buckets.
 * A bucket (_starpu_heteroprio_bucket) here is a n-uple of priorities,
 * and has a queue of tasks.
 * The architectures that access a bucket must be flaged in valide_archs.
 * And any task pushed to a bucket must be computable by these architectures.
 * For example this is not allowed :
 * a_bucket.valide_archs = STARPU_CPU | STARPU_CUDA
 * and push a task to this bucket that does not support STARPU_CUDA.
 * But it is possible to push a task that may be computed by more architecture,
 * like pushing a task that has a support on STARPU_CPU | STARPU_CUDA | STARPU_OPENCL.
 *
 * To enable the critical end support on a bucket, the user should say
 * which architecture is the fastest by setting factor_base_arch_index.
 * For example factor_base_arch_index = STARPU_CPU
 * Then for all the architectures related to this bucket (valide_archs)
 * the user should tell how slow are each architecture compare to the fastest one
 * in slow_factors_per_index.
 * For example slow_factors_per_index[FSTARPU_CUDA_IDX] = 10.0f if it is 10 times slower.
 * Then Heteroprio may forbid some tasks if the worker that requested it may not be
 * the best one to compute it.
 *
 * The mapping between priorities of the different architectures:
 * the prio_mapping_per_arch_index array should contains the mapping
 * for the priorities and nb_prio_per_arch_index should tells how many
 * priorities there are per architectures.
 *
 * Finally Heteroprio is prefetching HETEROPRIO_MAX_PREFETCH tasks per worker.
 * Remark if HETEROPRIO_MAX_PREFETCH=1 then there is no prefetching
 * because the single task will be consummed directly.
 * When no tasks are found a worker will try to steal one but this will
 * happen only on the worker of the same type.
 * For example, Cuda workers will steal work to Cuda workers only.
 *
 *
 * Example of callback function that set the correct variables:
 * #include "../../Src/GroupTree/StarPUUtils/FStarPUHeteoprio.hpp"
 *
 * void initSchedulerCallback(unsigned sched_ctx_id,
 *     struct _starpu_heteroprio_center_policy_heteroprio *heteroprio){
 *     // CPU uses 3 buckets
 *     starpu_heteroprio_set_nb_prios(heteroprio, FSTARPU_CPU_IDX, 3);
 *     // It uses direct mapping idx => idx
 *     for(unsigned idx = 0 ; idx < 3 ; ++idx){
 *         starpu_heteroprio_set_mapping(heteroprio, FSTARPU_CPU_IDX, idx, idx);
 *         starpu_heteroprio_set_faster_arch(heteroprio, FSTARPU_CPU_IDX, idx);
 *     }
 * #ifdef STARPU_USE_OPENCL
 *     // OpenCL is enabled and uses 2 buckets
 *     starpu_heteroprio_set_nb_prios(heteroprio, FSTARPU_OPENCL_IDX, 2);
 *     // OpenCL will first look to priority 2
 *     starpu_heteroprio_set_mapping(heteroprio, FSTARPU_OPENCL_IDX, 0, 2);
 *     // For this bucket OpenCL is the fastest
 *     starpu_heteroprio_set_faster_arch(heteroprio, FSTARPU_OPENCL_IDX, 2);
 *     // And CPU is 4 times slower
 *     starpu_heteroprio_set_arch_slow_factor(heteroprio, FSTARPU_CPU_IDX, 2, 4.0f);
 *
 *     starpu_heteroprio_set_mapping(heteroprio, FSTARPU_OPENCL_IDX, 1, 1);
 *     // We let the CPU as the fastest and tell that OpenCL is 1.7 times slower
 *     starpu_heteroprio_set_arch_slow_factor(heteroprio, FSTARPU_OPENCL_IDX, 1, 1.7f);
 * #endif
 * }
 *
 * In this example, the CPU can compute 3 buckets and its mapping is
 * {0,1,2} => {0,1,2} so nb_prio_per_arch_index = 3
 * So it will compute the tasks with priority 0, then with 1, and finally with 2.
 * We inform that the CPU is involve in these buckets by setting valide_archs.
 *
 * The Opencl workers use 2 buckets (numbers 1 and 2).
 * So the mapping is {0,1} => {2,1},
 * the opencl workers will compute first the tasks with priority 2 and then 1.
 * We set valide_archs to tell that OpenCL workers use these buckets.
 * We say that for the bucket with index 2, the OpenCL is faster:
 *     heteroprio->buckets[2].factor_base_arch_index = FSTARPU_OPENCL_IDX;
 * And we say that the CPU is 4 times slower.
 * For the bucket one we keep the CPU has being the fastest but tell that the OpenCL
 * is 1.7 slower.
 *
 * It is advised to run the scheduler in debug first (to enable assert)
 * since an important number of checks are done and a wrong configuration
 * or priorities is directly fired to the user.
 *
 * In case of using this code outside ScalFMM, notice that we use:
 * enum FStarPUTypes{
 *     // First will be zero
 * #ifdef STARPU_USE_CPU
 *     FSTARPU_CPU_IDX, // = 0
 * #endif
 * #ifdef STARPU_USE_CUDA
 *     FSTARPU_CUDA_IDX,
 * #endif
 * #ifdef STARPU_USE_OPENCL
 *     FSTARPU_OPENCL_IDX,
 * #endif
 *     // This will be the number of archs
 *     FSTARPU_NB_TYPES
 * };
 *
 * const unsigned FStarPUTypesToArch[FSTARPU_NB_TYPES+1] = {
 *     #ifdef STARPU_USE_CPU
 *         STARPU_CPU,
 *     #endif
 *     #ifdef STARPU_USE_CUDA
 *         STARPU_CUDA,
 *     #endif
 *     #ifdef STARPU_USE_OPENCL
 *         STARPU_OPENCL,
 *     #endif
 *         0
 * };
 *
 * The test to let a worker taking a task from a bucket is :
 * // There is at least one task in the list
 * (_starpu_fifo_empty(bucket->tasks_queue) == 0
 * // And the worker has not taken all the task he needs
 * && nb_tasks_to_prefetch
 * // And no faster arch has been set
 * && (bucket->factor_base_arch_index == 0
 * // Or the worker is the faster arch
 * || worker->arch_index == bucket->factor_base_arch_index
 * // OR the number of task/ number of faster worker >= slow factor
 * (float(_starpu_fifo_size(bucket->tasks_queue))/float(heteroprio->nb_workers_per_arch_index[bucket->factor_base_arch_index])
 *              >= bucket->slow_factors_per_index[worker->arch_index])))
 *
 * For example there is n task, w faster worker, and the slow factor is s:
 * n = 10, w = 3 => 10/3 = 3.33 >= s (so a task will be given if s is no more than 3 times slower)
 */

#include "../../Utils/FGlobal.hpp"
#include "FStarPUUtils.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

#include <starpu_scheduler.h>
#include <starpu_bitmap.h>

#include <assert.h>

/********************************************************************************/
/********************************************************************************/

/* Out of starpu build, cannot access : #include <sched_policies/fifo_queues.h> */
struct _starpu_fifo_taskq_node{
    struct _starpu_fifo_taskq_node* next;
    struct starpu_task* task;
};

struct _starpu_fifo_taskq{
    struct _starpu_fifo_taskq_node* head;
    struct _starpu_fifo_taskq_node* tail;
    unsigned ntasks;
};

struct _starpu_fifo_taskq* _starpu_create_fifo(){
    struct _starpu_fifo_taskq* fifo = (struct _starpu_fifo_taskq*)malloc(sizeof(struct _starpu_fifo_taskq));
    memset(fifo, 0, sizeof(struct _starpu_fifo_taskq));
    return fifo;
}

void _starpu_destroy_fifo(struct _starpu_fifo_taskq* fifo){
    assert(fifo);
    assert(fifo->head == NULL && fifo->tail == NULL && fifo->ntasks == 0);
    free(fifo);
}

int _starpu_fifo_empty(struct _starpu_fifo_taskq *fifo){
    return fifo->ntasks == 0;
}

struct starpu_task* _starpu_fifo_pop_local_task(struct _starpu_fifo_taskq *fifo){
    assert(fifo);
    assert(fifo->ntasks);
    assert(fifo->head);
    struct starpu_task* task = fifo->head->task;
    struct _starpu_fifo_taskq_node* to_remove = fifo->head;
    if(fifo->tail == fifo->head){
        fifo->tail = NULL;
    }
    fifo->head = fifo->head->next;
    free(to_remove);
    fifo->ntasks -= 1;
    return task;
}

void _starpu_task_list_push_back(struct _starpu_fifo_taskq *fifo, struct starpu_task* task){
    assert(fifo);
    struct _starpu_fifo_taskq_node* new_node = (struct _starpu_fifo_taskq_node*)malloc(sizeof(struct _starpu_fifo_taskq_node));
    new_node->task = task;
    new_node->next = NULL;
    if(fifo->tail != NULL){
        fifo->tail->next = new_node;
        fifo->tail = new_node;
    }
    else{
        fifo->head = new_node;
        fifo->tail = new_node;
    }
    fifo->ntasks += 1;
}

/* Cannot find a function that give the nb of task in a fifo so provide it */
int _starpu_fifo_size(struct _starpu_fifo_taskq *fifo){
    return fifo->ntasks;
}

/********************************************************************************/
/********************************************************************************/

/* A bucket corresponds to a Pair of priorities
 * When a task is pushed with a priority X, it will be stored
 * into the bucket X.
 * All the tasks stored in the fifo should be computable by the arch
 * in valide_archs.
 * For example if valide_archs = (STARPU_CPU|STARPU_CUDA)
 * Then task->task->cl->where should be at least (STARPU_CPU|STARPU_CUDA)
 */
struct _starpu_heteroprio_bucket{
    /* The task of the current bucket */
    struct _starpu_fifo_taskq* tasks_queue;
    /* The correct arch for the current bucket */
    unsigned valide_archs;
    /* The slow factors for any archs */
    float slow_factors_per_index[FSTARPU_NB_TYPES];
    /* The base arch for the slow factor (the fatest arch for the current task in the bucket */
    unsigned factor_base_arch_index;
};

/* Init a bucket */
static void _starpu_heteroprio_bucket_init(struct _starpu_heteroprio_bucket* bucket){
    memset(bucket, 0, sizeof(*bucket));
    bucket->tasks_queue =  _starpu_create_fifo();
}

/* Release a bucket */
static void _starpu_heteroprio_bucket_release(struct _starpu_heteroprio_bucket* bucket){
    assert(_starpu_fifo_empty(bucket->tasks_queue) != 0);
    _starpu_destroy_fifo(bucket->tasks_queue);
}

/********************************************************************************/
/********************************************************************************/


/* HETEROPRIO_MAX_PREFETCH Represent the number of task stored in each worker queue if possible */
#define HETEROPRIO_MAX_PREFETCH 1
#if HETEROPRIO_MAX_PREFETCH <= 0
#error HETEROPRIO_MAX_PREFETCH == 1 means no prefetch so HETEROPRIO_MAX_PREFETCH must >= 1
#endif

/* A worker is mainly composed of a fifo for the tasks
 * and some direct access to worker properties.
 * The fifo is implemented with any array,
 * to read a task, access tasks_queue[tasks_queue_index]
 * to write a task, access tasks_queue[(tasks_queue_index+tasks_queue_size)%HETEROPRIO_MAX_PREFETCH]
 */
struct _starpu_heteroprio_worker{
    unsigned arch_type;
    unsigned arch_index;
    struct starpu_task* tasks_queue[HETEROPRIO_MAX_PREFETCH];
    unsigned tasks_queue_size;
    unsigned tasks_queue_index;
    starpu_pthread_mutex_t ws_prefetch_mutex;
};

/* Init a worker by setting every thing to zero */
static void _starpu_heteroprio_worker_init(struct _starpu_heteroprio_worker* worker){
    memset(worker, 0, sizeof(*worker));
    worker->tasks_queue_index = 0;
    STARPU_PTHREAD_MUTEX_INIT(&worker->ws_prefetch_mutex, NULL);
}

/* Release a worker */
static void _starpu_heteroprio_worker_release(struct _starpu_heteroprio_worker* worker){
    assert(worker->tasks_queue_size == 0);
    STARPU_PTHREAD_MUTEX_DESTROY(&worker->ws_prefetch_mutex);
}

/********************************************************************************/
/********************************************************************************/


/* HETEROPRIO_MAX_PRIO is the maximum prio/buckets available */
#define HETEROPRIO_MAX_PRIO 100

/* This is the core part of the scheduler.
 * It contains the buckets, the worker information
 * and several counters to avoid useless iteration
 * Also it contains the mapping for each arch to the correct buckets.
 * For example a worker of type CUDA has index FSTARPU_CUDA_IDX
 * It has nb_prio_per_arch_index[FSTARPU_CUDA_IDX] buckets to check (<= HETEROPRIO_MAX_PRIO).
 * It will access the correct bucket using the mapping prio_mapping_per_arch_index[FSTARPU_CUDA_IDX][idx].
 */
struct _starpu_heteroprio_center_policy_heteroprio
{
    starpu_pthread_mutex_t policy_mutex;
    struct starpu_bitmap *waiters;
    /* The bucket to store the tasks */
    struct _starpu_heteroprio_bucket buckets[HETEROPRIO_MAX_PRIO];
    /* The number of buckets for each arch */
    unsigned nb_prio_per_arch_index[FSTARPU_NB_TYPES];
    /* The mapping to the corresponding buckets */
    unsigned prio_mapping_per_arch_index[FSTARPU_NB_TYPES][HETEROPRIO_MAX_PRIO];
    /* The number of available tasks for a given arch (not prefetched) */
    unsigned nb_remaining_tasks_per_arch_index[FSTARPU_NB_TYPES];
    /* The total number of tasks in the bucket (not prefetched) */
    unsigned total_tasks_in_buckets;
    /* The total number of prefetched tasks for a given arch */
    unsigned nb_prefetched_tasks_per_arch_index[FSTARPU_NB_TYPES];
    /* The information for all the workers */
    struct _starpu_heteroprio_worker workers_heteroprio[STARPU_NMAXWORKERS];
    /* The number of workers */
    unsigned nb_workers;
    /* The number of workers for a given arch */
    unsigned nb_workers_per_arch_index[FSTARPU_NB_TYPES];
};

/********************************************************************************/
/********************************************************************************/

/* This is the callback that must init the scheduler buckets */
typedef void (*Heteroprio_callback_type)(unsigned sched_ctx_id, void* heteroprio);
/*extern*/ Heteroprio_callback_type initialize_heteroprio_center_policy_callback = NULL;


inline void starpu_heteroprio_set_callback(Heteroprio_callback_type user_callback){
    initialize_heteroprio_center_policy_callback = user_callback;
}

/** Tell how many prio there are for a given arch */
inline void starpu_heteroprio_set_nb_prios(void* heterodata, const FStarPUTypes arch, const unsigned max_prio){
    assert(max_prio < HETEROPRIO_MAX_PRIO);
    ((struct _starpu_heteroprio_center_policy_heteroprio*)heterodata)->nb_prio_per_arch_index[arch] = max_prio;
}

/** Set the mapping for a given arch prio=>bucket */
inline void starpu_heteroprio_set_mapping(void* heterodata, const FStarPUTypes arch, const unsigned source_prio, const unsigned dest_bucket_id){
    assert(dest_bucket_id < HETEROPRIO_MAX_PRIO);
    ((struct _starpu_heteroprio_center_policy_heteroprio*)heterodata)->prio_mapping_per_arch_index[arch][source_prio] = dest_bucket_id;
    ((struct _starpu_heteroprio_center_policy_heteroprio*)heterodata)->buckets[dest_bucket_id].valide_archs |= FStarPUTypesToArch[arch];
}

/** Tell which arch is the faster for the tasks of a bucket (optional) */
inline void starpu_heteroprio_set_faster_arch(void* heterodata, const FStarPUTypes arch, const unsigned bucket_id){
    assert(bucket_id < HETEROPRIO_MAX_PRIO);
    ((struct _starpu_heteroprio_center_policy_heteroprio*)heterodata)->buckets[bucket_id].factor_base_arch_index = arch;
    ((struct _starpu_heteroprio_center_policy_heteroprio*)heterodata)->buckets[bucket_id].slow_factors_per_index[arch] = 0;
}

/** Tell how slow is a arch for the tasks of a bucket (optional) */
inline void starpu_heteroprio_set_arch_slow_factor(void* heterodata, const FStarPUTypes arch, const unsigned bucket_id, const float slow_factor){
    assert(bucket_id < HETEROPRIO_MAX_PRIO);
    ((struct _starpu_heteroprio_center_policy_heteroprio*)heterodata)->buckets[bucket_id].slow_factors_per_index[arch] = slow_factor;
}

/********************************************************************************/
/********************************************************************************/

/* Init the scheduler - This will call the init callback! */
static void initialize_heteroprio_center_policy(unsigned sched_ctx_id)
{
    /* Copy of eager */
//#ifdef STARPU_HAVE_HWLOC
//    starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_TREE);
//#else
//    starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
//#endif
    /* Alloc the scheduler data  */
    struct _starpu_heteroprio_center_policy_heteroprio* heteroprio = (struct _starpu_heteroprio_center_policy_heteroprio*)malloc(sizeof(struct _starpu_heteroprio_center_policy_heteroprio));
    memset(heteroprio, 0, sizeof(*heteroprio));
    heteroprio->waiters = starpu_bitmap_create();
    starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)heteroprio);
    STARPU_PTHREAD_MUTEX_INIT(&heteroprio->policy_mutex, NULL);
    /* End copy of eager */

    /* Init the buckets */
    for(unsigned idx_prio = 0 ; idx_prio < HETEROPRIO_MAX_PRIO ; ++idx_prio){
        _starpu_heteroprio_bucket_init(&heteroprio->buckets[idx_prio]);
    }

    /* Init the worker information */
    heteroprio->nb_workers = starpu_worker_get_count();
    for(unsigned idx_worker = 0 ; idx_worker < starpu_worker_get_count() ; ++idx_worker){
        _starpu_heteroprio_worker_init(&heteroprio->workers_heteroprio[idx_worker]);
        switch(starpu_worker_get_type(idx_worker)){
#ifdef STARPU_USE_CPU
            case STARPU_CPU_WORKER:
                heteroprio->workers_heteroprio[idx_worker].arch_type = STARPU_CPU;
                heteroprio->workers_heteroprio[idx_worker].arch_index = FSTARPU_CPU_IDX;
                break;
#endif
#ifdef STARPU_USE_CUDA
            case STARPU_CUDA_WORKER:
                heteroprio->workers_heteroprio[idx_worker].arch_type = STARPU_CUDA;
                heteroprio->workers_heteroprio[idx_worker].arch_index = FSTARPU_CUDA_IDX;
                break;
#endif
#ifdef STARPU_USE_OPENCL
            case STARPU_OPENCL_WORKER:
                heteroprio->workers_heteroprio[idx_worker].arch_type = STARPU_OPENCL;
                heteroprio->workers_heteroprio[idx_worker].arch_index = FSTARPU_OPENCL_IDX;
                break;
#endif
        default:
            assert(0);
        }
        heteroprio->nb_workers_per_arch_index[heteroprio->workers_heteroprio[idx_worker].arch_index] += 1;
    }

    /* Ask the user to fill the bucket information */
    assert(initialize_heteroprio_center_policy_callback != NULL);
    (*initialize_heteroprio_center_policy_callback)(sched_ctx_id, heteroprio);

    /* Ensure that information have been correctly filled */
    unsigned check_all_archs[HETEROPRIO_MAX_PRIO];
    memset(check_all_archs, 0, sizeof(unsigned)*HETEROPRIO_MAX_PRIO);
    for(unsigned arch_index = 0 ; arch_index < FSTARPU_NB_TYPES ; ++arch_index){
        assert(heteroprio->nb_prio_per_arch_index[arch_index] <= HETEROPRIO_MAX_PRIO);
        unsigned check_archs[HETEROPRIO_MAX_PRIO];
        memset(check_archs, 0, sizeof(unsigned)*HETEROPRIO_MAX_PRIO);

        for(unsigned idx_prio = 0 ; idx_prio < heteroprio->nb_prio_per_arch_index[arch_index] ; ++idx_prio){
            const unsigned mapped_prio = heteroprio->prio_mapping_per_arch_index[arch_index][idx_prio];
            assert(mapped_prio <= HETEROPRIO_MAX_PRIO);
            assert(heteroprio->buckets[mapped_prio].slow_factors_per_index[arch_index] >= 0.0);
            assert(heteroprio->buckets[mapped_prio].valide_archs & FStarPUTypesToArch[arch_index]);
            check_archs[mapped_prio]      = 1;
            check_all_archs[mapped_prio] += 1;
        }
        for(unsigned idx_prio = 0 ; idx_prio < HETEROPRIO_MAX_PRIO ; ++idx_prio){
            /* Ensure the current arch use a bucket or someone else can use it */
            assert(check_archs[idx_prio] == 1 || heteroprio->buckets[idx_prio].valide_archs == 0
                   || (heteroprio->buckets[idx_prio].valide_archs & ~FStarPUTypesToArch[arch_index]) != 0);
        }
    }
    /* Ensure that if a valide_archs = (STARPU_CPU|STARPU_CUDA) then check_all_archs[] = 2 for example */
    for(unsigned idx_prio = 0 ; idx_prio < HETEROPRIO_MAX_PRIO ; ++idx_prio){
        unsigned nb_arch_on_bucket = 0;
        for(unsigned arch_index = 0 ; arch_index < FSTARPU_NB_TYPES ; ++arch_index){
            if(heteroprio->buckets[idx_prio].valide_archs & FStarPUTypesToArch[arch_index]){
                nb_arch_on_bucket += 1;
            }
        }
        assert(check_all_archs[idx_prio] == nb_arch_on_bucket);
    }
}

/* Release a scheduler */
static void deinitialize_heteroprio_center_policy(unsigned sched_ctx_id)
{
    struct _starpu_heteroprio_center_policy_heteroprio *heteroprio = (struct _starpu_heteroprio_center_policy_heteroprio*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

    /* Ensure there are no more tasks */
    assert(heteroprio->total_tasks_in_buckets == 0);
    for(unsigned arch_index = 0 ; arch_index < FSTARPU_NB_TYPES ; ++arch_index){
        assert(heteroprio->nb_remaining_tasks_per_arch_index[arch_index] == 0);
        assert(heteroprio->nb_prefetched_tasks_per_arch_index[arch_index] == 0);
    }

    for(unsigned idx_prio = 0 ; idx_prio < HETEROPRIO_MAX_PRIO ; ++idx_prio){
        _starpu_heteroprio_bucket_release(&heteroprio->buckets[idx_prio]);
    }

    for(unsigned idx_worker = 0 ; idx_worker < heteroprio->nb_workers ; ++idx_worker){
        _starpu_heteroprio_worker_release(&heteroprio->workers_heteroprio[idx_worker]);
    }

    /* Copy of eager */
    starpu_bitmap_destroy(heteroprio->waiters);

    //starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
    STARPU_PTHREAD_MUTEX_DESTROY(&heteroprio->policy_mutex);
    free(heteroprio);
    /* End copy of eager */
}

/* Push a new task (simply store it and update counters) */
static int push_task_heteroprio_policy(struct starpu_task *task)
{
    unsigned sched_ctx_id = task->sched_ctx;
    struct _starpu_heteroprio_center_policy_heteroprio *heteroprio = (struct _starpu_heteroprio_center_policy_heteroprio*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

    /* One worker at a time use heteroprio */
    STARPU_PTHREAD_MUTEX_LOCK(&heteroprio->policy_mutex);

    /* Retrieve the correct bucket */
    assert(task->priority < HETEROPRIO_MAX_PRIO);
    struct _starpu_heteroprio_bucket* bucket = &heteroprio->buckets[task->priority];
    /* Ensure that any worker that check that list can compute the task */
    assert(bucket->valide_archs
           && ((bucket->valide_archs ^ task->cl->where) & bucket->valide_archs) == 0);
    /* save the task */
    _starpu_task_list_push_back(bucket->tasks_queue,task);

    /* Inc counters */
    for(unsigned arch_index = 0 ; arch_index < FSTARPU_NB_TYPES ; ++arch_index){
        /* We test the archs on the bucket and not on task->cl->where since it is restrictive */
        if(bucket->valide_archs & FStarPUTypesToArch[arch_index]){
            heteroprio->nb_remaining_tasks_per_arch_index[arch_index] += 1;
        }
    }
    heteroprio->total_tasks_in_buckets += 1;

    starpu_push_task_end(task);

    /* Copy of eager */
    /*if there are no tasks_queue block */
    /* wake people waiting for a task */
    unsigned worker = 0;
    struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

    struct starpu_sched_ctx_iterator it;
#ifndef STARPU_NON_BLOCKING_DRIVERS
    char dowake[STARPU_NMAXWORKERS] = { 0 };
#endif

    workers->init_iterator(workers, &it);
    while(workers->has_next(workers, &it))
    {
        worker = workers->get_next(workers, &it);

#ifdef STARPU_NON_BLOCKING_DRIVERS
        if (!starpu_bitmap_get(heteroprio->waiters, worker))
            /* This worker is not waiting for a task */
            continue;
#endif

        if (starpu_worker_can_execute_task_first_impl(worker, task, NULL))
        {
            /* It can execute this one, tell him! */
#ifdef STARPU_NON_BLOCKING_DRIVERS
            starpu_bitmap_unset(heteroprio->waiters, worker);
            /* We really woke at least somebody, no need to wake somebody else */
            break;
#else
            dowake[worker] = 1;
#endif
        }
    }
    /* Let the task free */
    STARPU_PTHREAD_MUTEX_UNLOCK(&heteroprio->policy_mutex);

#ifndef STARPU_NON_BLOCKING_DRIVERS
    /* Now that we have a list of potential workers, try to wake one */

    workers->init_iterator(workers, &it);
    while(workers->has_next(workers, &it))
    {
        worker = workers->get_next(workers, &it);
        if (dowake[worker])
            if (starpu_wake_worker(worker))
                break; // wake up a single worker
    }
#endif
    /* End copy of eager */

    return 0;
}


static struct starpu_task *pop_task_heteroprio_policy(unsigned sched_ctx_id)
{
    const unsigned workerid = starpu_worker_get_id();
    struct _starpu_heteroprio_center_policy_heteroprio *heteroprio = (struct _starpu_heteroprio_center_policy_heteroprio*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
    struct _starpu_heteroprio_worker* worker = &heteroprio->workers_heteroprio[workerid];

    /* If not tasks available, not tasks in worker queue or some arch worker queue just return NULL */
    if ((heteroprio->total_tasks_in_buckets == 0 || heteroprio->nb_remaining_tasks_per_arch_index[worker->arch_index] == 0)
            && worker->tasks_queue_size == 0 && heteroprio->nb_prefetched_tasks_per_arch_index[worker->arch_index] == 0){
        return NULL;
    }

#ifdef STARPU_NON_BLOCKING_DRIVERS
    if (starpu_bitmap_get(heteroprio->waiters, workerid)){
        /* Nobody woke us, avoid bothering the mutex */
        return NULL;
    }
#endif

    STARPU_PTHREAD_MUTEX_LOCK(&heteroprio->policy_mutex);

    /* keep track of the new added task to perfom real prefetch on node */
    unsigned nb_added_tasks = 0;

    /* Check that some tasks are available for the current worker arch */
    if( heteroprio->nb_remaining_tasks_per_arch_index[worker->arch_index] != 0 ){
        /* Ideally we would like to fill the prefetch array */
        unsigned nb_tasks_to_prefetch = (HETEROPRIO_MAX_PREFETCH-worker->tasks_queue_size);
        /* But there are maybe less tasks than that! */
        if(nb_tasks_to_prefetch > heteroprio->nb_remaining_tasks_per_arch_index[worker->arch_index]){
            nb_tasks_to_prefetch = heteroprio->nb_remaining_tasks_per_arch_index[worker->arch_index];
        }
        /* But in case there are less tasks than worker we take the minimum */
        if(heteroprio->nb_remaining_tasks_per_arch_index[worker->arch_index] < heteroprio->nb_workers){
            if(worker->tasks_queue_size == 0) nb_tasks_to_prefetch = 1;
            else nb_tasks_to_prefetch = 0;
        }

        nb_added_tasks = nb_tasks_to_prefetch;

        /* We iterate until we found all the tasks we need */
        for(unsigned idx_prio = 0 ; nb_tasks_to_prefetch && idx_prio < heteroprio->nb_prio_per_arch_index[worker->arch_index] ; ++idx_prio){
            /* Retrieve the bucket using the mapping */
            struct _starpu_heteroprio_bucket* bucket = &heteroprio->buckets[heteroprio->prio_mapping_per_arch_index[worker->arch_index][idx_prio]];
            /* Ensure we can compute task from this bucket */
            assert(bucket->valide_archs & worker->arch_type);
            /* Take nb_tasks_to_prefetch tasks if possible */
            while(_starpu_fifo_empty(bucket->tasks_queue) == 0 && nb_tasks_to_prefetch
                 && (bucket->factor_base_arch_index == 0 || worker->arch_index == bucket->factor_base_arch_index
                  || (float(_starpu_fifo_size(bucket->tasks_queue))/float(heteroprio->nb_workers_per_arch_index[bucket->factor_base_arch_index])
                        >= bucket->slow_factors_per_index[worker->arch_index]))){
                struct starpu_task* task = _starpu_fifo_pop_local_task(bucket->tasks_queue);
                assert(starpu_worker_can_execute_task(workerid, task, 0));
                /* Save the task */
                worker->tasks_queue[(worker->tasks_queue_index+worker->tasks_queue_size)%HETEROPRIO_MAX_PREFETCH] = task;
                worker->tasks_queue_size += 1;
                /* Update general counter */
                heteroprio->nb_prefetched_tasks_per_arch_index[worker->arch_index] += 1;
                heteroprio->total_tasks_in_buckets    -= 1;
                for(unsigned arch_index = 0 ; arch_index < FSTARPU_NB_TYPES ; ++arch_index){
                    /* We test the archs on the bucket and not on task->cl->where since it is restrictive */
                    if(bucket->valide_archs & FStarPUTypesToArch[arch_index]){
                        heteroprio->nb_remaining_tasks_per_arch_index[arch_index] -= 1;
                    }
                }
                /* Decrease the number of tasks to found */
                nb_tasks_to_prefetch -= 1;
                // TODO starpu_prefetch_task_input_on_node(task, workerid);
            }
        }
        // This is not sure any more assert(nb_tasks_to_prefetch == 0);
    }

    struct starpu_task* task = NULL;

    /* The worker has some tasks in its queue */
    if(worker->tasks_queue_size){
        task = worker->tasks_queue[worker->tasks_queue_index];
        worker->tasks_queue_index = (worker->tasks_queue_index+1)%HETEROPRIO_MAX_PREFETCH;
        worker->tasks_queue_size -= 1;
        heteroprio->nb_prefetched_tasks_per_arch_index[worker->arch_index] -= 1;
    }
    /* Otherwise look if we can steal some work */
    else if(heteroprio->nb_prefetched_tasks_per_arch_index[worker->arch_index]){
        /* If HETEROPRIO_MAX_PREFETCH==1 it should not be possible to steal work */
        assert(HETEROPRIO_MAX_PREFETCH != 1);
        /* Each worker starts from its own index and do a turn */
        for(unsigned idx_worker_it = 1 ; idx_worker_it < heteroprio->nb_workers ; ++idx_worker_it){
            const unsigned idx_worker = ((workerid+idx_worker_it)%heteroprio->nb_workers);
            /* we must never test on ourself */
            assert(idx_worker != workerid);
            /* If it is the same arch and there is a task to steal */
            if(heteroprio->workers_heteroprio[idx_worker].arch_index == worker->arch_index
                    && heteroprio->workers_heteroprio[idx_worker].tasks_queue_size){
                /* ensure the worker is not currently prefetching its data */
                STARPU_PTHREAD_MUTEX_LOCK(&heteroprio->workers_heteroprio[idx_worker].ws_prefetch_mutex);
                if(heteroprio->workers_heteroprio[idx_worker].arch_index == worker->arch_index
                        && heteroprio->workers_heteroprio[idx_worker].tasks_queue_size){
                    /* steal the last added task */
                    task = heteroprio->workers_heteroprio[idx_worker].tasks_queue[(heteroprio->workers_heteroprio[idx_worker].tasks_queue_index+heteroprio->workers_heteroprio[idx_worker].tasks_queue_size-1)
                                                                                    % HETEROPRIO_MAX_PREFETCH];
                    /* update the worker by saying we steal a task */
                    heteroprio->workers_heteroprio[idx_worker].tasks_queue_size -= 1;
                    /* we steal a task update global counter */
                    heteroprio->nb_prefetched_tasks_per_arch_index[heteroprio->workers_heteroprio[idx_worker].arch_index] -= 1;
                    STARPU_PTHREAD_MUTEX_UNLOCK(&heteroprio->workers_heteroprio[idx_worker].ws_prefetch_mutex);
                    break;
                }
                STARPU_PTHREAD_MUTEX_UNLOCK(&heteroprio->workers_heteroprio[idx_worker].ws_prefetch_mutex);
            }
        }
    }

    /* Copy of eager */
    if (!task){
        /* Tell pushers that we are waiting for tasks_queue for us */
        starpu_bitmap_set(heteroprio->waiters, workerid);
    }
    STARPU_PTHREAD_MUTEX_UNLOCK(&heteroprio->policy_mutex);

    if(task){
        unsigned child_sched_ctx = starpu_sched_ctx_worker_is_master_for_child_ctx(workerid, sched_ctx_id);
        if(child_sched_ctx != STARPU_NMAX_SCHED_CTXS){
            starpu_sched_ctx_revert_task_counters(sched_ctx_id, task->flops);
#if (STARPU_MAJOR_VERSION >= 1) && (STARPU_MINOR_VERSION >= 3)
            starpu_sched_ctx_move_task_to_ctx(task, child_sched_ctx, 1, 1);
#else
            starpu_sched_ctx_move_task_to_ctx(task, child_sched_ctx);
#endif
            return NULL;
        }
    }
    /* End copy of eager */

    /* if we have task (task) me way have some in the queue (worker->tasks_queue_size) that was freshly addeed (nb_added_tasks) */
    if(task && worker->tasks_queue_size && nb_added_tasks && starpu_get_prefetch_flag()){
        const unsigned memory_node = starpu_worker_get_memory_node(workerid);

        /* prefetch the new task that I own but protecte my node from work stealing during the prefetch */
        STARPU_PTHREAD_MUTEX_LOCK(&worker->ws_prefetch_mutex);

        /* prefetch task but stop in case we now some one may steal a task from us */
        while(nb_added_tasks && heteroprio->nb_remaining_tasks_per_arch_index[worker->arch_index] != 0){
            /* prefetch from closest to end task */
            starpu_prefetch_task_input_on_node(worker->tasks_queue[(worker->tasks_queue_index+worker->tasks_queue_size-nb_added_tasks)%HETEROPRIO_MAX_PREFETCH], memory_node);
            nb_added_tasks -= 1;
        }

        STARPU_PTHREAD_MUTEX_UNLOCK(&worker->ws_prefetch_mutex);
    }

    return task;
}

/********************************************************************************/
/********************************************************************************/


struct starpu_sched_policy _starpu_sched_heteroprio_policy_build(){
    struct starpu_sched_policy policy;
    memset(&policy, 0, sizeof(policy));
    policy.init_sched         = initialize_heteroprio_center_policy;
    policy.deinit_sched       = deinitialize_heteroprio_center_policy;
    policy.add_workers        = NULL;
    policy.remove_workers     = NULL;
    policy.push_task          = push_task_heteroprio_policy;
    policy.pop_task           = pop_task_heteroprio_policy;
    policy.pre_exec_hook      = NULL;
    policy.post_exec_hook     = NULL;
    policy.pop_every_task     = NULL;
    policy.policy_name        = "heteroprio";
    policy.policy_description = "heteroprio policy from scalfmm";
#if (STARPU_MAJOR_VERSION >= 1) && (STARPU_MINOR_VERSION >= 3)
#ifdef STARPU_HAVE_HWLOC
    policy.worker_type = STARPU_WORKER_TREE;
#else
    policy.worker_type = STARPU_WORKER_LIST;
#endif
#endif
    return policy;
}

struct starpu_sched_policy _starpu_sched_heteroprio_policy = _starpu_sched_heteroprio_policy_build();


#ifdef __cplusplus
}
#endif

#endif // FSTARPUHETEOPRIO_HPP
