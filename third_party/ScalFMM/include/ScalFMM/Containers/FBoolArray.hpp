// See LICENCE file at project root
#ifndef FBOOLARRAY_HPP
#define FBOOLARRAY_HPP

#include "../Utils/FGlobal.hpp"
#include "../Utils/FAssert.hpp"
// To get memcpy
#include <cstring>

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FBoolArray
* Please read the license
*
* A bool array is a dynamique allocated array that used 1 bit per value.
* Under the wood, it use bit operations to acess/set value in an array of
* native type.
*/
class FBoolArray{
    /** Size of a unsigned long */
    const static FSize BytesInBlock = sizeof(unsigned long);
    const static FSize SizeOfBlock = BytesInBlock * 8;

    /** The array to store bits */
    unsigned long* array;
    /** Size of the memory allocated */
    FSize memSize;
    /** Size of the array => number of real elements */
    FSize size;

    /** get size to number of long */
    FSize LongFromSize(const FSize inSize){
        return ((inSize + SizeOfBlock - 1) / SizeOfBlock);
    }

    /** Alloc an array */
    unsigned long * AllocArray(const FSize inSize){
        return new unsigned long[LongFromSize(inSize)];
    }

public :
    /** Constructor with size */
    explicit FBoolArray(const FSize inSize = 0) : array(AllocArray(inSize)), memSize(LongFromSize(inSize)*BytesInBlock), size(inSize) {
        setToZeros();
    }

    /** Constructor form another array */
    FBoolArray(const FBoolArray& other): array(AllocArray(other.size)), memSize(other.memSize), size(other.size){
        *this = other;
    }

    /** Move the data */
    FBoolArray(FBoolArray&& other): array(nullptr), memSize(0), size(0){
        array   = other.array;
        memSize = other.memSize;
        size    = other.size;
        other.array   = nullptr;
        other.memSize = 0;
        other.size    = 0;
    }

    /** remove all values and allocate new array */
    void reset(const FSize inSize){
        delete [] array;
        array   = (AllocArray(inSize));
        memSize = (LongFromSize(inSize)*BytesInBlock);
        size    = (inSize);
        setToZeros();
    }

    /** Destructor */
    ~FBoolArray(){
        delete [] array;
    }

    /**
    * Operator =
    * Array must have the same size
    */
    FBoolArray& operator=(const FBoolArray& other){
        FAssertLF(size == other.size);
        memcpy(array, other.array, memSize);
        return *this;
    }

    /**
     * Move the data from one array to the other
     */
    FBoolArray& operator=(FBoolArray&& other){
        delete [] array;
        array   = other.array;
        memSize = other.memSize;
        size    = other.size;
        other.array   = nullptr;
        other.memSize = 0;
        other.size    = 0;
        return *this;
    }

    /**
    * Operator ==
    * Array must have the same size
    */
    bool operator==(const FBoolArray& other){
        return memcmp(array, other.array, memSize) == 0;
    }

    /**
    * Operator !=
    * Array must have the same size
    */
    bool operator!=(const FBoolArray& other){
        return !(*this == other);
    }

    /** To get a value */
    bool get(const FSize inPos) const {
        const FSize posInArray = inPos / SizeOfBlock;
        const FSize bytePosition = inPos - (posInArray * 8);
        return (array[posInArray] >> bytePosition) & 1;
    }

    /** To set a value */
    void set(const FSize inPos, const bool inVal){
        const FSize posInArray = inPos / SizeOfBlock;
        const FSize bytePosition = inPos - (posInArray * 8);
        if(inVal) array[posInArray] |= (1UL << bytePosition);
        else array[posInArray] &= ~(1UL << bytePosition);
    }

    /** To get the size of the array */
    FSize getSize() const {
        return size;
    }

    /** Set all the memory to 0 */
    void setToZeros() const {
        memset( array, 0, memSize);
    }

    /** Set all the memory to 1 */
    void setToOnes() const {
        memset( array, (unsigned char)0xFF, memSize);
    }
};


#endif //FBOOLARRAY_HPP


