#ifndef FPAROBJECT_HPP
#define FPAROBJECT_HPP

#include <omp.h>
#include <functional>

/**
 * This class is a way to hide concurent access to the user.
 * Especially with the FParForEachOctree functions.
 * Please refer to the testOctreeParallelFuncteur to see an example.
 */
template <class SharedClass>
class FParObject{
    const int nbThreads;
    SharedClass* objects;

    FParObject(const FParObject&) = delete;
    FParObject& operator=(const FParObject&) = delete;

public:

    template <class... ArgsClass>
    explicit FParObject(ArgsClass... args) : nbThreads(omp_get_max_threads()), objects(nullptr) {
        objects = reinterpret_cast<SharedClass*>(new unsigned char[sizeof(SharedClass)*nbThreads]);
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            new (&objects[idxThread]) SharedClass(args...);
        }
    }

    ~FParObject(){
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            objects[idxThread].~SharedClass();
        }
        delete[] reinterpret_cast<unsigned char*>(objects);
    }

    SharedClass& getMine(){
        return objects[omp_get_thread_num()];
    }

    const SharedClass& getMine() const {
        return objects[omp_get_thread_num()];
    }

    void applyToAll(std::function<void(SharedClass&)> func){
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            func(objects[idxThread]);
        }
    }

    void applyToAllPar(std::function<void(SharedClass&)> func){
        #pragma omp parallel for num_threads(nbThreads) schedule(static)
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            func(objects[idxThread]);
        }
    }

    SharedClass reduce(std::function<SharedClass(const SharedClass&, const SharedClass&)> func){
        if(nbThreads == 0){
            return SharedClass();
        }
        else if(nbThreads == 1){
            return objects[0];
        }
        else{
            SharedClass res = func(objects[0], objects[1]);
            for(int idxThread = 2 ; idxThread < nbThreads ; ++idxThread){
                res = func(res, objects[idxThread]);
            }
            return res;
        }
    }

    void set(const SharedClass& val){
        #pragma omp parallel for num_threads(nbThreads) schedule(static)
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            objects[idxThread] = val;
        }
    }

    FParObject& operator=(const SharedClass& val){
        set(val);
        return *this;
    }
};


template <class SharedClass>
class FParArray{
    const int nbThreads;
    SharedClass** arrays;
    size_t arraySize;

    FParArray(const FParArray&) = delete;
    FParArray& operator=(const FParArray&) = delete;

public:

    explicit FParArray(const size_t inArraySize = 0)
        : nbThreads(omp_get_max_threads()), arrays(nullptr), arraySize(inArraySize) {
        arrays = new SharedClass*[nbThreads];
        #pragma omp parallel for schedule(static)
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            arrays[idxThread] = new SharedClass[inArraySize];
            if(std::is_pod<SharedClass>::value){
                memset(arrays[idxThread], 0, sizeof(SharedClass)*inArraySize);
            }
        }
    }

    ~FParArray(){
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            delete[] arrays[idxThread];
        }
        delete[] arrays;
    }

    void resize(const size_t inArraySize){
        if(inArraySize != arraySize){
            for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
                delete[] arrays[idxThread];
            }

            arraySize = inArraySize;

            #pragma omp parallel for schedule(static)
            for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
                arrays[idxThread] = new SharedClass[inArraySize];
                if(std::is_pod<SharedClass>::value){
                    memset(arrays[idxThread], 0, sizeof(SharedClass)*inArraySize);
                }
            }
        }
    }

    SharedClass* getMine(){
        return arrays[omp_get_thread_num()];
    }

    const SharedClass* getMine() const {
        return arrays[omp_get_thread_num()];
    }

    void applyToAll(std::function<void(SharedClass*)> func){
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            func(arrays[idxThread]);
        }
    }

    void applyToAllPar(std::function<void(SharedClass*)> func){
        #pragma omp parallel for num_threads(nbThreads) schedule(static)
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            func(arrays[idxThread]);
        }
    }

    void reduceArray(SharedClass* resArray, std::function<SharedClass(const SharedClass&, const SharedClass&)> func){
        if(nbThreads == 0){
        }
        else if(nbThreads == 1){
            if(std::is_pod<SharedClass>::value){
                memcpy(resArray, arrays[0], sizeof(SharedClass)*arraySize);
            }
            else{
                for(int idxVal = 0 ; idxVal < arraySize ; ++idxVal){
                    resArray = arrays[0][idxVal];
                }
            }
        }
        else{
            for(int idxVal = 0 ; idxVal < arraySize ; ++idxVal){
                resArray[idxVal] = func(arrays[0][idxVal], arrays[1][idxVal]);
            }
            for(int idxThread = 2 ; idxThread < nbThreads ; ++idxThread){
                for(int idxVal = 0 ; idxVal < arraySize ; ++idxVal){
                    resArray[idxVal] = func(resArray[idxVal], arrays[idxThread][idxVal]);
                }
            }
        }
    }

    void copyArray(const SharedClass* array){
        #pragma omp parallel for num_threads(nbThreads) schedule(static)
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            if(std::is_pod<SharedClass>::value){
                memcpy(arrays[idxThread], array, sizeof(SharedClass)*arraySize);
            }
            else{
                for(int idxVal = 0 ; idxVal < arraySize ; ++idxVal){
                    arrays[idxThread][idxVal] = array[idxVal];
                }
            }
        }
    }
};

#endif // FPAROBJECT_HPP
