// See LICENCE file at project root
#ifndef FBUFFERWRITER_HPP
#define FBUFFERWRITER_HPP

#include "FVector.hpp"
#include "FAbstractBuffer.hpp"

/** @author Berenger Bramas
  * This class provide a fast way to manage a memory and fill it
  *
  * Put some data
  * then insert back if needed
  * finaly use data pointer as you like
  */
class FBufferWriter : public FAbstractBufferWriter {
private:
    FVector<char> buffer; //< The buffer

public:
    /** Constructor with a default capacity of 512 bytes */
    explicit FBufferWriter(const FSize inCapacity = 512) : buffer(inCapacity) {
    }

    /** Destructor */
    virtual ~FBufferWriter(){
    }

    /** Get allocated memory pointer */
    char* data(){
        return buffer.data();
    }

    /** Get allocated memory pointer */
    const char* data() const {
        return buffer.data();
    }

    /** Get the filled space */
    FSize getSize() const {
        return buffer.getSize();
    }

    /** Write data by mem cpy */
    template <class ClassType>
    void write(const ClassType& object){
        buffer.memocopy(reinterpret_cast<const char*>(&object), FSize(sizeof(ClassType)));
    }

    /** Write back, position + sizeof(object) has to be < size */
    template <class ClassType>
    void writeAt(const FSize position, const ClassType& object){
        (*reinterpret_cast<ClassType*>(&buffer[position])) = object;
    }

    /** Write an array */
    template <class ClassType>
    void write(const ClassType* const objects, const FSize inSize){
        buffer.memocopy(reinterpret_cast<const char*>(objects), FSize(sizeof(ClassType)) * inSize);
    }

    /** Equivalent to write */
    template <class ClassType>
    FBufferWriter& operator<<(const ClassType& object){
        write(object);
        return *this;
    }

    /** Reset the writing index, but do not change the capacity */
    void reset(){
        buffer.clear();
    }
};


#endif // FBUFFERWRITER_HPP
