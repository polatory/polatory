// See LICENCE file at project root
#ifndef FVECTOR_HPP
#define FVECTOR_HPP


#include "../Utils/FGlobal.hpp"

#include <cstring>

/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * @class FVector
 * Please read the license
 *
 * This class is a vector container.
 * It is a very basic vector to enable strong performance.
 *
 * Please refere to unit test utestVector.cpp to know more.
 */
template<class ObjectType>
class FVector {
protected:
    static const FSize DefaultSize = 10;      /**< Default size */

    ObjectType* array;        /**< memory area*/

    FSize capacity;             /**< memory capacity, the size of array */
    FSize index;                /**< index in array, the current position to insert */

public:
    typedef ObjectType ValueType; /**< data type of data in FVector */

    /**
    * @brief Constructor
    * Create a vector with a default size capacity
    */
    FVector() : array(nullptr), capacity(DefaultSize), index(0) {
        array = reinterpret_cast< ObjectType* >( new char[sizeof(ObjectType) * DefaultSize] );
    }

    /**
    * @brief Constructor
    * @param inCapacity the memory to allocate
    */
    explicit FVector(const FSize inCapacity): array(nullptr), capacity(inCapacity), index(0) {
        if( inCapacity ){
            array = reinterpret_cast< ObjectType* >( new char[sizeof(ObjectType) * inCapacity]);
        }
    }

    /**
     * Copy constructor
     * @param other original vector
     * object must have an copy constructor
     */
    FVector(const FVector& other): array(nullptr), capacity(other.capacity), index(other.index) {
        if( other.capacity ){
            array = reinterpret_cast< ObjectType* >( new char[sizeof(ObjectType) * other.capacity]);
            // Copy each element
            for(FSize idx = 0 ; idx < other.index ; ++idx){
                new((void*)&array[idx]) ObjectType(other.array[idx]);
            }
        }
    }

    /**
     * Copy constructor
     * @param other original vector
     * object must have an copy constructor
     */
    FVector(FVector&& other): array(nullptr), capacity(0), index(0) {
        array    = other.array;
        capacity = other.capacity;
        index    = other.index;
        other.array    = nullptr;
        other.capacity = 0;
        other.index    = 0;
    }

    /** Copy operator
      * @param other the original vector
      * @return this after copying data
      * Objects of the current vector are deleted then
      * objects from other are copied using copy constructor.
      * The capacity is not copied.
      */
    FVector& operator=(const FVector& other){
        if(&other != this){
            // clear current element
            clear();
            // alloc bigger if needed
            if(capacity < other.getSize()){
                delete [] reinterpret_cast< char* >(array);
                capacity = FSize(double(other.getSize()) * 1.5);
                array = reinterpret_cast< ObjectType* >( new char[sizeof(ObjectType) * capacity]);
            }

            index = other.index;
            for(FSize idx = 0 ; idx < other.index ; ++idx){
                new((void*)&array[idx]) ObjectType(other.array[idx]);
            }
        }
        return *this;
    }

    /**
    *@brief Copy operator
    */
    FVector& operator=(FVector&& other){
        if(&other != this){
            clear();
            delete [] reinterpret_cast< char* >(array);
            array    = other.array;
            capacity = other.capacity;
            index    = other.index;
            other.array    = nullptr;
            other.capacity = 0;
            other.index    = 0;
        }
        return (*this);
    }

    /**
    *@brief destructor
    */
    virtual ~FVector(){
        clear();
        delete [] reinterpret_cast< char* >(array);
    }

    /**
    * @brief Get the buffer capacity
    * @return the buffer capacity
    * The capacity is the current memory size allocated.
    */
    FSize getCapacity() const{
        return capacity;
    }

    /**
    *@brief Set the buffer capacity
    *@param inCapacity to change the capacity
    * If capacity given is lower than size elements after capacity are removed
    */
    void setCapacity(const FSize inCapacity) {
        if( inCapacity != capacity ){
            while(inCapacity < index){
                (&array[--index])->~ObjectType();
            }

            // Copy elements
            ObjectType* const nextArray = reinterpret_cast< ObjectType* >( inCapacity ? new char[sizeof(ObjectType) * inCapacity] : nullptr);
            for(FSize idx = 0 ; idx < index ; ++idx){
                new((void*)&nextArray[idx]) ObjectType(std::move(array[idx]));
                (&array[idx])->~ObjectType();
            }
            delete [] reinterpret_cast< char* >(array);

            array    = nextArray;
            capacity = inCapacity;
        }
    }

    /** Resize the vector (and change the capacity if needed) */
    template <typename... Args>
 //  [[gnu::deprecated]]
    void resize(const FSize newSize, Args... args){
        if(index < newSize){
            if(capacity < newSize){
                setCapacity(FSize(double(newSize)*1.5));
            }
            while(index != newSize){
                new((void*)&array[index]) ObjectType(args...);
                ++index ;
            }
        }
        else{
            index = newSize;
        }
    }


    /**
    * @return Last inserted object
    * This function return the data at the last position
    */
    const ObjectType& head() const {
        return array[index - 1];
    }

    /**
    * @return Last inserted object
    * This function return the data at the last position
    */
    ObjectType& head() {
        return array[index - 1];
    }

    /**
    * @brief Delete all, then size = 0 but capacity is unchanged
    */
    void clear() {
        while(0 < index){
            (&array[--index])->~ObjectType();
        }
    }

    /**
    * @return The number of element added into the vector
    * This is not the capcity
    */
    FSize getSize() const{
        return index;
    }

    /**
    * @brief pop the first value
    * Warning, FVector do not check that there is an element before poping
    * The capacity is unchanged
    */
    void pop(){
        (&array[--index])->~ObjectType();
    }

    /**
    * Add a value at the end, resize the capacity if needed
    * @param inValue the new value
    */
    void push( const ObjectType & inValue ){
        // if needed, increase the vector
        if( index == capacity ){
            setCapacity(static_cast<FSize>(double(capacity+1) * 1.5));
        }
        // add the new element
        new((void*)&array[index++]) ObjectType(inValue);
    }

    /**
     * To Create a new object
     */
    template <typename... Args>
    void pushNew(Args... args){
        // if needed, increase the vector
        if( index == capacity ){
            setCapacity(static_cast<FSize>(double(capacity+1) * 1.5));
        }
        // add the new element
        new((void*)&array[index++]) ObjectType(args...);
    }

    /**
    * Add one value multiple time
    * @param inValue the new value
    * @param inRepeat the number of time the value is inserted
    */
    void set( const ObjectType & inValue, const FSize inRepeat){
        // if needed, increase the vector
        if( capacity < index + inRepeat ){
            setCapacity(FSize(double(index + inRepeat) * 1.5));
        }
        // add the new element
        for( FSize idx = 0 ; idx < inRepeat ; ++idx){
            new((void*)&array[index++]) ObjectType(inValue);
        }
    }

    /**
    *@brief Get a reference of a given value
    *@param inPosition the query position
    *@return the value
    */
    ObjectType& operator[](const FSize inPosition ) {
            return array[inPosition];
    }

    /**
    * @brief Get a const reference of a given value
    * @param inPosition the query position
    * @return the value
    */
    const ObjectType& operator[](const FSize inPosition ) const {
            return array[inPosition];
    }

    /** To get the entire array memory
      * @return the array allocated by the vector
      */
    ObjectType* data(){
        return array;
    }

    /** To get the entire array memory
      * @return the array allocated by the vector
      */
    const ObjectType* data() const{
        return array;
    }

    /** To take values from C array but copy with = operator
      * @param inArray the array to copy values
      * @param inSize the size of the array
      */
    void extractValues(const ObjectType*const inArray, const FSize inSize){
        // Check available memory
        if(capacity < index + inSize){
            setCapacity( FSize(double(index + inSize) * 1.5) );
        }
        // Copy values
        for(FSize idx = 0 ; idx < inSize ; ++idx){
            new((void*)&array[index++]) ObjectType(inArray[idx]);
        }
    }

    /** To take values from C array but copy with memcpy
      * @param inArray the array to copie values
      * @param inSize the size of the array
      */
    void memocopy(const ObjectType*const inArray, const FSize inSize){
        // Check available memory
        if(capacity < index + inSize){
            setCapacity( FSize(double(index + inSize) * 1.5) );
        }
        // Copy values
        memcpy(&array[index], inArray, inSize * sizeof(ObjectType));
        index += inSize;
    }

    /** Remove a values by shifting all the next values */
    void removeOne(const FSize idxToRemove){
        for(FSize idxMove = idxToRemove + 1; idxMove < index ; ++idxMove){
            array[idxMove - 1] = array[idxMove];
        }
        index -= 1;
    }

    /** This class is a basic iterator
      * <code>
      *  typename FVector<FSize>::ConstBasicIterator iter(myVector);<br>
      *  while( iter.hasNotFinished() ){<br>
      *      printf("%d\n",iter.data());<br>
      *      iter.gotoNext();<br>
      *  } <br>
      * </code>
      */
    class BasicIterator {
    protected:
        FVector* const vector;  /**< the vector to work on*/
        FSize index;              /**< the current node*/

    public:
        /** Empty destructor */
        virtual ~BasicIterator(){}

        /** Constructor need a vector */
        explicit BasicIterator(FVector<ObjectType>& inVector) : vector(&inVector), index(0){}

        /** Go to next vector element */
        void gotoNext() {
            ++index;
        }

        /** is it over
          * @return true if we are over the vector
          */
        bool hasNotFinished() const {
            return index < vector->index;
        }

        /** Get current data */
        ObjectType& data(){
            return vector->array[index];
        }

        /** Get current data */
        const ObjectType& data() const{
            return vector->array[index];
        }

        /** Set the data */
        void setData(const ObjectType& inData){
            vector->array[index] = inData;
        }

        /** Remove current data
          * It will move all the data after to their previous position
          */
        void remove(){
            if( hasNotFinished() ){
                for(FSize idxMove = index + 1; idxMove < vector->index ; ++idxMove){
                    vector->array[idxMove - 1] = vector->array[idxMove];
                }
                vector->index -= 1;
            }
        }

    };
    friend class BasicIterator;

    /** This class is a basic const iterator
      * it uses a const vector to work on
      */
    class ConstBasicIterator {
    protected:
        const FVector* const vector;  /**< the vector to work on*/
        FSize index;                    /**< the current node*/

    public:
        /** Empty destructor */
        virtual ~ConstBasicIterator(){}

        /** Constructor need a vector */
        explicit ConstBasicIterator(const FVector<ObjectType>& inVector) : vector(&inVector), index(0){}

        /** Go to next vector element */
        void gotoNext(){
            ++index;
        }

        /** is it over
          * @return true if we are over the vector
          */
        bool hasNotFinished() const{
            return index < vector->index;
        }

        /** Get current data */
        const ObjectType& data() const{
            return vector->array[index];
        }

    };
    friend class ConstBasicIterator;

};



#endif // FVECTOR_HPP

