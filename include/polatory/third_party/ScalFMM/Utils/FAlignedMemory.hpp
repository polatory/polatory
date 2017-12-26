// See LICENCE file at project root
#ifndef FALIGNEDMEMORY_HPP
#define FALIGNEDMEMORY_HPP

#include <cstdint>


/**
 * This should be used to allocate and deallocate aligned memory.
 */
namespace FAlignedMemory {

template <std::size_t AlignementValue>
inline void* AllocateBytes(const std::size_t inSize){
    if(inSize == 0){
        return nullptr;
    }

    // Ensure it is a power of 2
    static_assert(AlignementValue != 0 && ((AlignementValue-1)&AlignementValue) == 0, "Alignement must be a power of 2");
    // We will need to store the adress of the real blocks
    const std::size_t sizeForAddress = (AlignementValue < sizeof(unsigned char*)? sizeof(unsigned char*) : AlignementValue);

    unsigned char* allocatedMemory      = new unsigned char[inSize + AlignementValue-1 + sizeForAddress];
    unsigned char* alignedMemoryAddress = reinterpret_cast<unsigned char*>((reinterpret_cast<std::size_t>(allocatedMemory) + AlignementValue-1 + sizeForAddress) & ~static_cast<std::size_t>(AlignementValue-1));
    unsigned char* ptrForAddress        = (alignedMemoryAddress - sizeof(unsigned char*));

    // Save allocated adress
    *reinterpret_cast<unsigned char**>(ptrForAddress) = allocatedMemory;
    // Return aligned address
    return reinterpret_cast<void*>(alignedMemoryAddress);
}

inline void DeallocBytes(const void* ptrToFree){
    if( ptrToFree ){
        const unsigned char*const* storeRealAddress = reinterpret_cast<const unsigned char*const *>(reinterpret_cast<const unsigned char*>(ptrToFree) - sizeof(unsigned char*));
        delete[] reinterpret_cast<const unsigned char*>(*storeRealAddress);
    }
}

template <std::size_t AlignementValue, class ArrayType>
inline ArrayType* AllocateArray(const std::size_t inNbElementsInArray){
    if(inNbElementsInArray == 0){
        return nullptr;
    }
    const std::size_t inSize = (inNbElementsInArray*sizeof(ArrayType));

    // Ensure it is a power of 2
    static_assert(AlignementValue != 0 && ((AlignementValue-1)&AlignementValue) == 0, "Alignement must be a power of 2");
    // We will need to store the adress of the real blocks
    const std::size_t sizeForAddressAndNumber = (AlignementValue < (sizeof(unsigned char*)+sizeof(std::size_t))? (sizeof(unsigned char*)+sizeof(std::size_t)) : AlignementValue);

    unsigned char* allocatedMemory      = new unsigned char[inSize + AlignementValue-1 + sizeForAddressAndNumber];
    unsigned char* alignedMemoryAddress = reinterpret_cast<unsigned char*>((reinterpret_cast<std::size_t>(allocatedMemory) + AlignementValue-1 + sizeForAddressAndNumber) & ~static_cast<std::size_t>(AlignementValue-1));
    unsigned char* ptrForAddress        = (alignedMemoryAddress - sizeof(unsigned char*));
    unsigned char* ptrForNumber         = (ptrForAddress - sizeof(std::size_t));

    // Save allocated adress
    *reinterpret_cast<unsigned char**>(ptrForAddress) = allocatedMemory;
    *reinterpret_cast<std::size_t*>(ptrForNumber) = inNbElementsInArray;
    // Return aligned address
    ArrayType* array = reinterpret_cast<ArrayType*>(alignedMemoryAddress);

    for(std::size_t idx = 0 ; idx < inNbElementsInArray ; ++idx){
        new (&array[idx]) ArrayType();
    }
    return array;
}

template <class ArrayType>
inline void DeallocArray(const ArrayType* ptrToFree){
    if( ptrToFree ){
        const std::size_t numberOfElements = (*reinterpret_cast<const std::size_t*>(reinterpret_cast<const unsigned char*>(ptrToFree) - sizeof(unsigned char*) - sizeof(std::size_t)));

        for(std::size_t idx = 0 ; idx < numberOfElements ; ++idx){
            ptrToFree[idx].~ArrayType();
        }

        const unsigned char*const* storeRealAddress = reinterpret_cast<const unsigned char*const *>(reinterpret_cast<const unsigned char*>(ptrToFree) - sizeof(unsigned char*));
        delete[] reinterpret_cast<const unsigned char*>(*storeRealAddress);
    }
}

}


#endif // FALIGNEDMEMORY_HPP
