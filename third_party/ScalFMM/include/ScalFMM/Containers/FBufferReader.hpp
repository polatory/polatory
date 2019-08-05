// See LICENCE file at project root
#ifndef FBUFFERREADER_HPP
#define FBUFFERREADER_HPP

#include "FVector.hpp"
#include "FAbstractBuffer.hpp"

/** @author Berenger Bramas
  * This class provide a fast way to manage a memory and convert
  * the content to basic type.
  *
  * Specifie the needed space with reserve, then fill it with data
  * finaly read and convert.
  */
class FBufferReader : public FAbstractBufferReader {
    FVector<char> buffer;   //< The memory buffer
    FSize index;              //< The current index reading position

public:
    /** Construct with a memory size init to 0 */
    explicit FBufferReader(const FSize inCapacity = 0) : buffer(inCapacity), index(0) {
        if(inCapacity){
            reserve(inCapacity);
        }
    }

    /** Destructor */
    virtual ~FBufferReader() override {
    }

    /** Get the memory area */
    char* data() override {
        return buffer.data();
    }

    /** Get the memory area */
    const char* data() const  override {
        return buffer.data();
    }

    /** Size of the meomry initialzed */
    FSize getSize() const override {
        return buffer.getSize();
    }

    /** Move the read index to a position */
    void seek(const FSize inIndex) override {
        index = inIndex;
    }

    /** Get the read position */
    FSize tell() const  override {
        return index;
    }

    /** Reset and allocate nbBytes memory filled with 0 */
    void reserve(const FSize nbBytes){
        reset();
        buffer.set( 0, nbBytes);
    }

    /** Move the read index to 0 */
    void reset(){
        buffer.clear();
        index = 0;
    }

    /** Get a value with memory cast */
    template <class ClassType>
    ClassType getValue(){
        ClassType value = (*reinterpret_cast<ClassType*>(&buffer[index]));
        index += FSize(sizeof(ClassType));
        return value;
    }

    /** Fill a value with memory cast */
    template <class ClassType>
    void fillValue(ClassType* const inValue){
        (*inValue) = (*reinterpret_cast<ClassType*>(&buffer[index]));
        index += FSize(sizeof(ClassType));
    }

    /** Fill one/many value(s) with memcpy */
    template <class ClassType>
    void fillArray(ClassType* const inArray, const FSize inSize){
        memcpy( inArray, &buffer[index], sizeof(ClassType) * inSize);
        index += FSize(sizeof(ClassType) * inSize);
    }

    /** Same as fillValue */
    template <class ClassType>
    FBufferReader& operator>>(ClassType& object){
        fillValue(&object);
        return *this;
    }
};


#endif // FBUFFERREADER_HPP
