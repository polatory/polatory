// ===================================================================================
// Copyright ScalFmm 2016 INRIA, Olivier Coulaud, Bérenger Bramas,
// Matthias Messner olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the
// FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
// An extension to the license is given to allow static linking of scalfmm
// inside a proprietary application (no matter its license).
// See the main license file for more details.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================
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
