// See LICENCE file at project root
#ifndef FMPIBUFFERREADER_HPP
#define FMPIBUFFERREADER_HPP

#include <memory>
#include "../Utils/FMpi.hpp"
#include "FAbstractBuffer.hpp"
#include "../Utils/FAssert.hpp"

/** @author Cyrille Piacibello, Berenger Bramas
 * This class provide the same features as FBufferWriter
 *
 * Put some data
 * then insert back if needed
 * finally use data pointer as you like
 */
class FMpiBufferReader : public FAbstractBufferReader {
    FSize arrayCapacity;        //< Allocated space
    std::unique_ptr<char[]> array;  //< Allocated Array
    FSize currentIndex;

public :
    /*Constructor with a default arrayCapacity of 512 bytes */
    explicit FMpiBufferReader(const FSize inDefaultCapacity = 512):
        arrayCapacity(inDefaultCapacity),
        array(new char[inDefaultCapacity]),
        currentIndex(0){
        FAssertLF(array, "Cannot allocate array");
    }

    /** To change the capacity (but reset the head to 0) */
    void cleanAndResize(const FSize newCapacity){
        if(newCapacity != arrayCapacity){
            arrayCapacity = newCapacity;
            array.reset(new char[newCapacity]);
        }
        currentIndex = 0;
    }

    /** Destructor
   */
    virtual ~FMpiBufferReader(){
    }

    /** Get allocated memory pointer */
    char* data() override {
        return array.get();
    }

    /** Get allocated memory pointer */
    const char* data() const override  {
        return array.get();
    }

    /** get the filled space */
    FSize getSize() const override {
        return currentIndex;
    }

    /** Size of the memory initialized */
    FSize getCapacity() const{
        return arrayCapacity;
    }

    /** Move the read index to a position */
    void seek(const FSize inIndex) override {
        FAssertLF(inIndex <= arrayCapacity, "FMpiBufferReader :: Aborting :: Can't move index because buffer isn't long enough ",inIndex," ",arrayCapacity);
        currentIndex = inIndex;
    }

    /** Get the read position */
    FSize tell() const override  {
        return currentIndex;
    }

    /** Get a value with memory cast */
    template <class ClassType>
    ClassType getValue(){
        FAssertLF(currentIndex + FSize(sizeof(ClassType)) <= arrayCapacity );
        ClassType value;
        memcpy(&value, &array[currentIndex], sizeof(ClassType));
        currentIndex += sizeof(ClassType);
        return value;
    }

    /** Get a value with memory cast at a specified index */
    template <class ClassType>
    ClassType getValue(const FSize ind){
        currentIndex = ind;
        return getValue<ClassType>();
    }

    /** Fill a value with memory cast */
    template <class ClassType>
    void fillValue(ClassType* const inValue){
        FAssertLF(currentIndex + FSize(sizeof(ClassType)) <= arrayCapacity );
        memcpy(inValue, &array[currentIndex], sizeof(ClassType));
        currentIndex += sizeof(ClassType);
    }

    /** Fill one/many value(s) with memcpy */
    template <class ClassType>
    void fillArray(ClassType* const inArray, const FSize inSize){
        FAssertLF(currentIndex + FSize(sizeof(ClassType))*inSize <= arrayCapacity );
        memcpy(inArray, &array[currentIndex], sizeof(ClassType)*inSize);
        currentIndex += sizeof(ClassType)*inSize;
    }

    /** Same as fillValue */
    template <class ClassType>
    FMpiBufferReader& operator>>(ClassType& object){
        fillValue(&object);
        return *this;
    }

};
#endif

