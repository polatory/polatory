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
#ifndef FABSTRACTBUFFER_HPP
#define FABSTRACTBUFFER_HPP

/**
 * @brief The FAbstractBufferReader class defines what is an abstract buffer reader.
 * The buffer used by the mpi algorithm for example should defines this methods.
 */
class FAbstractBufferReader {
public:
    virtual ~FAbstractBufferReader(){
    }

    virtual char*       data()      = 0;
    virtual const char* data()      const  = 0;
    virtual FSize         getSize()   const = 0;
    virtual void        seek(const FSize inIndex) = 0;
    virtual FSize         tell()      const  = 0;

    template <class ClassType>
    ClassType getValue(){
        static_assert(sizeof(ClassType) == 0, "Your Buffer should implement getValue.");
        return ClassType();
    }
    template <class ClassType>
    void fillValue(ClassType* const){
        static_assert(sizeof(ClassType) == 0, "Your Buffer should implement fillValue.");
    }
    template <class ClassType>
    void fillArray(ClassType* const , const FSize ){
        static_assert(sizeof(ClassType) == 0, "Your Buffer should implement fillArray.");
    }
    template <class ClassType>
    FAbstractBufferReader& operator>>(ClassType& ){
        static_assert(sizeof(ClassType) == 0, "Your Buffer should implement operator>>.");
        return *this;
    }
};


/**
 * @brief The FAbstractBufferWriter class defines what is an abstract buffer writer.
 * The buffer used by the mpi algorithm for example should defines this methods.
 */
class FAbstractBufferWriter {
public:
    virtual ~FAbstractBufferWriter(){
    }

    virtual char*       data()  = 0;
    virtual const char* data()  const = 0;
    virtual FSize         getSize() const = 0;
    virtual void        reset() = 0;

    template <class ClassType>
    void write(const ClassType& object){
        static_assert(sizeof(ClassType) == 0, "Your Buffer should implement write.");
    }
    template <class ClassType>
    void writeAt(const FSize position, const ClassType& object){
        static_assert(sizeof(ClassType) == 0, "Your Buffer should implement writeAt.");
    }
    template <class ClassType>
    void write(const ClassType* const objects, const FSize inSize){
        static_assert(sizeof(ClassType) == 0, "Your Buffer should implement write.");
    }
    template <class ClassType>
    FAbstractBufferWriter& operator<<(const ClassType& ){
        static_assert(sizeof(ClassType) == 0, "Your Buffer should implement operator<<.");
        return *this;
    }
};

#endif // FABSTRACTBUFFER_HPP
