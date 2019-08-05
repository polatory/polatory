#ifndef FCUDADATA_HPP
#define FCUDADATA_HPP

#include "FCudaGlobal.hpp"

template <class ObjectClass>
class FCudaData{
protected:
    ObjectClass* cudaPtr;
    FSize nbElements;

    void allocAndCopy(const ObjectClass* inCpuPtr, const FSize inNbElements){
        FCudaCheck( cudaMalloc(&cudaPtr,inNbElements*sizeof(ObjectClass)) );
        FCudaCheck( cudaMemcpy( cudaPtr, inCpuPtr, inNbElements*sizeof(ObjectClass),
                    cudaMemcpyHostToDevice ) );
    }

    void dealloc(){
        FCudaCheck(cudaFree(cudaPtr));
        cudaPtr = nullptr;
        nbElements = 0;
    }
public:
    FCudaData(const FCudaData&) = delete;
    FCudaData& operator=(const FCudaData&) = delete;

    FCudaData(const ObjectClass* inCpuPtr, const FSize inNbElements)
        : cudaPtr(nullptr), nbElements(0){
        allocAndCopy(inCpuPtr, inNbElements);
    }


    FCudaData(FCudaData&& other)
        : cudaPtr(nullptr), nbElements(0){
        this->cudaPtr = other->cudaPtr;
        this->nbElements = other->nbElements;
        other->cudaPtr = nullptr;
        other->nbElements = 0;
    }

    FCudaData& operator=(FCudaData&& other){
        dealloc();
        this->cudaPtr = other->cudaPtr;
        this->nbElements = other->nbElements;
        other->cudaPtr = nullptr;
        other->nbElements = 0;
    }


    ~FCudaData(){
        dealloc();
    }

    void release(){
        dealloc();
    }

    ObjectClass* get() {
        return cudaPtr;
    }

    const ObjectClass* get() const {
        return cudaPtr;
    }

    FSize getSize() const {
        return nbElements;
    }

    void copyToHost(ObjectClass* inCpuPtr){
        FCudaCheck( cudaMemcpy( inCpuPtr, cudaPtr, nbElements*sizeof(ObjectClass),
                    cudaMemcpyDeviceToHost ) );
    }
};

#endif // FCUDADATA_HPP

