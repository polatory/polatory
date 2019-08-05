// See LICENCE file at project root
#ifndef FQUICKSORT_HPP
#define FQUICKSORT_HPP

#include <omp.h>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <vector> // For parallel without task
#include <utility> // For move
#include <functional> // To have std::function

#include "FGlobal.hpp"
#include "FMemUtils.hpp"


/** This class is parallel quick sort
  * It hold a mpi version
  * + 2 openmp versions (one on tasks and the other like mpi)
  * + a sequential version
  *
  * The task based algorithm is easy to undestand,
  * for the mpi/openmp2nd please see
  * Introduction to parallel computing (Grama Gupta Karypis Kumar)
  *
  * In the previous version Intel Compiler was not using the QS with task
  * defined(__ICC) || defined(__INTEL_COMPILER) defined the FQS_TASKS_ARE_DISABLED
  *
  * If needed one can define FQS_FORCE_NO_TASKS to ensure that the no task version is used.
  */

template <class SortType, class IndexType = size_t>
class FQuickSort {
protected:

#if _OPENMP < 200805 || defined(FQS_FORCE_NO_TASKS)
#define FQS_TASKS_ARE_DISABLED
    class TaskInterval{
        IndexType left;
        IndexType right;
        int deep;
    public:
        TaskInterval(const IndexType inLeft, const IndexType inRight, const int inDeep)
            : left(inLeft), right(inRight), deep(inDeep){
        }

        IndexType getLeft() const{
            return left;
        }
        IndexType getRight() const{
            return right;
        }
        int getDeep() const{
            return deep;
        }
    };
#endif

    /** swap to value */
    template <class NumType>
    static inline void Swap(NumType& value, NumType& other){
        const NumType temp = std::move(value);
        value = std::move(other);
        other = std::move(temp);
    }

    typedef bool (*infOrEqualPtr)(const SortType&, const SortType&);

    ////////////////////////////////////////////////////////////
    // Quick sort
    ////////////////////////////////////////////////////////////

    /* Use in the sequential qs */
    static IndexType QsPartition(SortType array[], IndexType left, IndexType right, const infOrEqualPtr infOrEqual){
        Swap(array[right],array[((right - left ) / 2) + left]);

        for( IndexType idx = left; idx < right ; ++idx){
            if( infOrEqual(array[idx],array[right]) ){
                Swap(array[idx],array[left]);
                left += 1;
            }
        }

        Swap(array[left],array[right]);

        return left;
    }


    /* The sequential qs */
    static void QsSequentialStep(SortType array[], const IndexType left, const IndexType right, const infOrEqualPtr infOrEqual){
        if(left < right){
            const IndexType part = QsPartition(array, left, right, infOrEqual);
            QsSequentialStep(array,part + 1,right, infOrEqual);
            if(part) QsSequentialStep(array,left,part - 1, infOrEqual);
        }
    }

#ifndef FQS_TASKS_ARE_DISABLED
    /** A task dispatcher */
    static void QsOmpTask(SortType array[], const IndexType left, const IndexType right, const int deep, const infOrEqualPtr infOrEqual){
        if(left < right){
            const IndexType part = QsPartition(array, left, right, infOrEqual);
            if( deep ){
                // default(none) has been removed for clang compatibility
                #pragma omp task firstprivate(array, part, right, deep, infOrEqual)
                QsOmpTask(array,part + 1,right, deep - 1, infOrEqual);
                // #pragma omp task default(none) firstprivate(array, part, right, deep, infOrEqual) // not needed
                if(part) QsOmpTask(array,left,part - 1, deep - 1, infOrEqual);
            }
            else {
                QsSequentialStep(array,part + 1,right, infOrEqual);
                if(part) QsSequentialStep(array,left,part - 1, infOrEqual);
            }
        }
    }
#endif

public:
    /* a sequential qs */
    static void QsSequential(SortType array[], const IndexType size, const infOrEqualPtr infOrEqual){
        QsSequentialStep(array, 0, size-1, infOrEqual);
    }

    static void QsSequential(SortType array[], const IndexType size){
        QsSequential(array, size, [](const SortType& v1, const SortType& v2){
            return v1 <= v2;
        });
    }

#ifdef FQS_TASKS_ARE_DISABLED
    static void QsOmp(SortType elements[], const int nbElements, const infOrEqualPtr infOrEqual){
        const int nbTasksRequiere = (omp_get_max_threads() * 5);
        int deep = 0;
        while( (1 << deep) < nbTasksRequiere ) deep += 1;

        std::vector<TaskInterval> tasks;
        tasks.push_back(TaskInterval(0, nbElements-1, deep));
        int numberOfThreadProceeding = 0;
        omp_lock_t mutexShareVariable;
        omp_init_lock(&mutexShareVariable);

        #pragma omp parallel
        {
            bool hasWorkToDo = true;
            while(hasWorkToDo){
                // Ask for the mutex
                omp_set_lock(&mutexShareVariable);
                if(tasks.size()){
                    // There is tasks to proceed
                    const TaskInterval ts(tasks.back());
                    tasks.pop_back();

                    // Does this task should create some other?
                    if(ts.getDeep() == 0){
                        // No release the mutex and run in seq
                        omp_unset_lock(&mutexShareVariable);
                        QsSequentialStep(elements , ts.getLeft(), ts.getRight(), infOrEqual);
                    }
                    else{
                        // Yes so inform other and release the mutex
                        numberOfThreadProceeding += 1;
                        omp_unset_lock(&mutexShareVariable);

                        // Partition
                        const IndexType part = QsPartition(elements, ts.getLeft(), ts.getRight(), infOrEqual);

                        // Push the new task in the vector
                        omp_set_lock(&mutexShareVariable);
                        tasks.push_back(TaskInterval(part+1, ts.getRight(), ts.getDeep()-1));
                        if(part) tasks.push_back(TaskInterval(ts.getLeft(), part-1, ts.getDeep()-1));
                        // We create new task but we are not working so inform other
                        numberOfThreadProceeding -= 1;
                        omp_unset_lock(&mutexShareVariable);
                    }
                }
                else{
                    // There is not task in the vector
                    #pragma omp flush(numberOfThreadProceeding)
                    if(numberOfThreadProceeding == 0){
                        // And there is no thread that may create some tasks so stop here
                        hasWorkToDo = false;
                    }
                    // Release mutex
                    omp_unset_lock(&mutexShareVariable);
                }
            }
        }

        omp_destroy_lock(&mutexShareVariable);
    }
#else
    /** The openmp quick sort */
    static void QsOmp(SortType array[], const IndexType size, const infOrEqualPtr infOrEqual){
        const int nbTasksRequiere = (omp_get_max_threads() * 5);
        int deep = 0;
        while( (1 << deep) < nbTasksRequiere ) deep += 1;

        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                QsOmpTask(array, 0, size - 1 , deep, infOrEqual);
            }
        }
    }
#endif

    static void QsOmp(SortType array[], const IndexType size){
        QsOmp(array, size, [](const SortType& v1, const SortType& v2){
            return v1 <= v2;
        });
    }
};

#endif // FQUICKSORT_HPP
