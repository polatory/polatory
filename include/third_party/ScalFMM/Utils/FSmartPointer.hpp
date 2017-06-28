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
#ifndef FSMARTPOINTER_HPP
#define FSMARTPOINTER_HPP


enum FSmartPointerType{
    FSmartArrayMemory,
    FSmartPointerMemory
};

template <FSmartPointerType MemoryType, class ClassType>
inline typename std::enable_if<(MemoryType == FSmartArrayMemory), void>::type FSmartDeletePointer(ClassType* ptr){
    delete[] ptr;
}

template <FSmartPointerType MemoryType, class ClassType>
inline typename std::enable_if<(MemoryType == FSmartPointerMemory), void>::type FSmartDeletePointer(ClassType* ptr){
    delete ptr;
}

/** This class is a basic smart pointer class
  * Use as FSmartPointer<int> array = new int[5];
  * FSmartPointer<int, PointerMemory> pt = new int;
  */
template <class ClassType, FSmartPointerType MemoryType = FSmartArrayMemory>
class FSmartPointer {
    ClassType* pointer; //< The pointer to the memory area
    int* counter;       //< Reference counter

public:
    // The type of data managed by the pointer
    typedef ClassType ValueType;

    /** Empty constructor */
    FSmartPointer() : pointer(nullptr), counter(nullptr) {
    }

    /** Constructor from the memory pointer */
    FSmartPointer(ClassType* const inPointer) : pointer(nullptr), counter(nullptr) {
        assign(inPointer);
    }

    /** Constructor from a smart pointer */
    FSmartPointer(const FSmartPointer& inPointer) : pointer(nullptr), counter(nullptr) {
        assign(inPointer);
    }

    /** Destructor, same as release */
    ~FSmartPointer(){
        release();
    }

    /** Assign operator, same as assign */
    void operator=(ClassType* const inPointer){
        assign(inPointer);
    }

    /** Assign operator, same as assign */
    void operator=(const FSmartPointer& inPointer){
        assign(inPointer);
    }

    /** Point to a new pointer, release if needed */
    void assign(ClassType* const inPointer){
        release();
        if( inPointer ){
            pointer = inPointer;
            counter = new int;
            (*counter) = 1;
        }
    }

    /** Point to a new pointer, release if needed */
    void assign(const FSmartPointer& inPointer){
        release();
        pointer = inPointer.pointer;
        counter = inPointer.counter;
        if(counter) (*counter) += 1;
    }

    /** Dec counter and Release the memory last */
    void release(){
        if(counter){
            (*counter) -= 1;
            if( (*counter) == 0 ){
                FSmartDeletePointer<MemoryType>(pointer);
                delete counter;
            }
            pointer = nullptr;
            counter = nullptr;
        }
    }

    /** To know if the smart pointer is pointing to a memory */
    bool isAssigned() const{
        return pointer != nullptr;
    }

    /** Is last => counter == 0 */
    bool isLast() const{
        return counter && (*counter) == 1;
    }

    /** Get the direct pointer */
    ClassType* getPtr(){
        return pointer;
    }

    /** Get the direct pointer */
    const ClassType* getPtr() const {
        return pointer;
    }

    /** Operator [] */
    ClassType& operator[](const int& index){
        return pointer[index];
    }

    /** Operator [] */
    const ClassType& operator[](const int& index) const {
        return pointer[index];
    }

    /** Operator * return a reference to the pointer object */
    ClassType& operator*(){
        return (*pointer);
    }

    /** Operator * return a reference to the pointer object */
    const ClassType& operator*() const {
        return (*pointer);
    }

    /** Return the pointing address */
    ClassType* operator->(){
        return pointer;
    }

    /** Return the pointing address */
    const ClassType* operator->() const {
        return pointer;
    }

    /** To cast to class type pointer */
    operator const ClassType*() const {
        return pointer;
    }

    /** To cast to class type pointer */
    operator ClassType*() {
        return pointer;
    }
};

#endif // FSMARTPOINTER_HPP
