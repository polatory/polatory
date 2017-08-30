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
#ifndef FMPIBUFFERWRITER_HPP
#define FMPIBUFFERWRITER_HPP

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
class FMpiBufferWriter : public FAbstractBufferWriter {
    FSize arrayCapacity;              //< Allocated Space
    std::unique_ptr<char[]> array;  //< Allocated Array
    FSize currentIndex;               //< Currently filled space

    /** Test and exit if not enought space */
    void expandIfNeeded(const FSize requestedSpace) {
        if( arrayCapacity < currentIndex + requestedSpace){
            arrayCapacity = FSize(double(currentIndex + requestedSpace + 1) * 1.5);
            char* arrayTmp = new char[arrayCapacity];
            memcpy(arrayTmp, array.get(), sizeof(char)*currentIndex);
            array.reset(arrayTmp);
        }
    }

public:
    /** Constructor with a default arrayCapacity of 512 bytes */
    explicit FMpiBufferWriter(const FSize inDefaultCapacity = 1024):
        arrayCapacity(inDefaultCapacity),
        array(new char[inDefaultCapacity]),
        currentIndex(0)
    {}


    /** To change the capacity (but reset the head to 0 if size if lower) */
    void resize(const FSize newCapacity){
        if(newCapacity != arrayCapacity){
            arrayCapacity = newCapacity;
            char* arrayTmp = new char[arrayCapacity];
            currentIndex = (currentIndex < arrayCapacity ? currentIndex : arrayCapacity-1);
            memcpy(arrayTmp, array.get(), sizeof(char)*currentIndex);
            array.reset(arrayTmp);
        }
    }

    /** Destructor */
    virtual ~FMpiBufferWriter(){
    }

    /** Get allocated memory pointer */
    char* data() override {
        return array.get();
    }

    /** Get allocated memory pointer */
    const char* data() const override  {
        return array.get();
    }

    /** Get the filled space */
    FSize getSize() const override  {
        return currentIndex;
    }

    /** Get the allocated space */
    FSize getCapacity() const {
        return arrayCapacity;
    }

    /** Write data by packing cpy */
    template <class ClassType>
    void write(const ClassType& object){
        expandIfNeeded(sizeof(ClassType));
        memcpy(&array[currentIndex], &object, sizeof(ClassType));
        currentIndex += sizeof(ClassType);
    }

    /**
   * Allow to pass rvalue to write
   */
    template <class ClassType>
    void write(const ClassType&& object){
        expandIfNeeded(sizeof(ClassType));
        memcpy(&array[currentIndex], &object, sizeof(ClassType));
        currentIndex += sizeof(ClassType);
    }

    /** Write back, position + sizeof(object) has to be < size */
    template <class ClassType>
    void writeAt(const FSize position, const ClassType& object){
        FAssertLF(position+FSize(sizeof(ClassType)) <= currentIndex);
        memcpy(&array[position], &object, sizeof(ClassType));
    }

    /** Write an array
   * Warning : inSize is a number of ClassType object to write, not a size in bytes
   */
    template <class ClassType>
    void write(const ClassType* const objects, const FSize inSize){
        expandIfNeeded(sizeof(ClassType) * inSize);
        memcpy(&array[currentIndex], objects, sizeof(ClassType)*inSize);
        currentIndex += sizeof(ClassType)*inSize;
    }

    /** Equivalent to write */
    template <class ClassType>
    FMpiBufferWriter& operator<<(const ClassType& object){
        write(object);
        return *this;
    }

    /** Reset the writing index, but do not change the arrayCapacity */
    void reset() override {
        currentIndex = 0;
    }
};


#endif // FBUFFERWRITER_HPP
