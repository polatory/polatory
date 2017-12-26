// See LICENCE file at project root
#ifndef FLIST_HPP
#define FLIST_HPP


#include "../Utils/FGlobal.hpp"


/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * @class FList
 * Please read the license
 *
 * This class is a linked list container.
 * It is a very basic list to enable strong performance.
 *
 * Please refer to unit test utestList.cpp to know more.
 */
template< class Object >
class FList {
    /** A list node */
    struct Node {
        Object target;	//< Object of the node
        Node* next;	//< Next node
    };

    Node* root; //< Root node, nullptr if size is 0
    int size;   //< Elements in the list

    /**
        * Copy a list into current object
        * The current list has to be empty when this function is called
        */
    void copy(const FList& other){
        // on iterator on the current list, another for the original list
        const Node* FRestrict  otherRoot = other.root;
        Node * FRestrict * myRoot = &this->root;
        // create, copy and progress
        while(otherRoot){
            (*myRoot) = new Node;
            (*myRoot)->target = otherRoot->target;

            myRoot = &(*myRoot)->next;
            otherRoot = otherRoot->next;
        }
        // End with null
        *myRoot = nullptr;
        this->size = other.size;
    }

public:
    typedef Object ValueType; /**< data type of data in FVector */

    /** Constructor (of an empty list) */
    FList() : root(nullptr) , size(0) {
    }

    /** Desctructor */
    virtual ~FList(){
        clear();
    }

    /**
        * Copy operator
        * This will clear the current list before copying
        * @param other the source list
        * @return the current list as a reference
        */
    FList& operator=(const FList& other){
        if(this != &other){
            clear();
            copy(other);
        }
        return *this;
    }

    /**
        * Copy constructor
        * @param other the source/original list
        */
    FList(const FList& other): root(nullptr) , size(0)  {
        copy(other);
    }

    /**
        * Copy operator
        * This will clear the current list before copying
        * @param other the source list
        * @return the current list as a reference
        */
    FList& operator=(FList&& other){
        if(&other != this){
            clear();
            root = other.root;
            size = other.size;
            other.root = nullptr;
            other.size = 0;
        }
        return *this;
    }

    /**
        * Copy constructor
        * @param other the source/original list
        */
    FList(FList&& other): root(nullptr) , size(0)  {
        root = other.root;
        size = other.size;
        other.root = nullptr;
        other.size = 0;
    }

    /**
        * To clear the list
        * Size is 0 after calling this function
        */
    void clear(){
        while(this->root){
            Node*const FRestrict next = this->root->next;
            delete this->root;
            this->root = next;
        }
        this->size = 0;
    }

    /**
        * Push an element in the head of the list
        * @param inObject the object to insert
        */
    void push(const Object& inObject){
        Node* newNode   = new Node;
        newNode->target = inObject;
        newNode->next 	= this->root;

        this->root 	= newNode;
        ++this->size;
    }

    /**
     * @brief pushEmpty
     * use default constructor
     */
    void pushEmpty(){
        Node* newNode   = new Node;
        newNode->next 	= this->root;

        this->root 	= newNode;
        ++this->size;
    }


    /**
        * To get head value (last pushed value)
        * the list is unchanged after this function
        * @param defaultValue as the returned value in case size == 0, equal Object() if no param as passed
        * @return first value if exists or defaultValue otherwise
        * FList does not check that size != 0
        */
    Object& head(/*Object& defaultValue = Object()*/){
        return this->root->target;
    }

    /**
    * To get head value as const
    * the list is unchanged after this function
    * @param defaultValue as the returned value in case size == 0, equal Object() if no param as passed
    * @return first value if exists or defaultValue otherwise
    * FList does not check that size != 0
    */
    const Object& head(/*const Object& defaultValue = Object()*/) const {
        return this->root->target;
    }

    /**
    * To remove the head value from the list
    * @warning you must check the list's size before calling this function!
    */
    void pop(){
        --this->size;
        Node*const FRestrict headNode  = this->root;
        this->root                     = this->root->next;
        delete headNode;
    }

    /**
        * To get the number of elements in the list
        * @return size
        */
    int getSize() const{
        return this->size;
    }

    /**
          * This iterator allow accessing list's elements
          * If you change the target list pointed by an iterator
          * you cannot used the iterator any more.
          * <code>
          * FList<int> values; <br>
          * // inserting ... <br>
          * FList<int>::BasicIterator iter(values); <br>
          * while( iter.hasNotFinished() ){ <br>
          *     iter.data() += 1; <br>
          *     iter.gotoNext(); <br>
          * } <br>
          * </code>
          */
    class BasicIterator {
    private:
        Node** iter; //< current node on the list

    public:
        /**
          * Constructor needs the target list
          * @param the list to iterate on
          */
        explicit BasicIterator(FList& list) : iter(&list.root) {
        }

        /** To gotoNext on the list */
        void gotoNext(){
            if( hasNotFinished() ){
                iter = &((*iter)->next);
                if( (*iter)->next ) Prefetch_Write0( (*iter)->next->next);
            }
        }

        /**
            * Current pointed value
            * current iterator must be valide (hasNotFinished()) to use this function
            */
        Object& data(){
            return (*iter)->target;
        }

        /**
            * Current pointed value
            * current iterator must be valide (hasNotFinished()) to use this function
            */
        const Object& data() const{
            return (*iter)->target;
        }

        /** Set the data */
        void setData(const Object& inData){
            (*iter)->target = inData;
        }

        /**
            * To know if an iterator is at the end of the list
            * @return true if the current iterator can gotoNext and access to value, else false
            */
        bool hasNotFinished() const{
            return (*iter) != 0;
        }

        /** Remove an element
              */
        void remove() {
            if( hasNotFinished() ){
                Node* temp = (*iter)->next;
                delete (*iter);
                (*iter) = temp;
            }
        }

    };

    /**
          * This iterator allow accessing list's elements
          * If you change the target list pointed by an iterator
          * you cannot used the iterator any more.
          * <code>
          * FList<int> values; <br>
          * // inserting ... <br>
          * FList<int>::ConstBasicIterator iter(values); <br>
          * while( iter.hasNotFinished() ){ <br>
          *     iter.data() += 1; <br>
          *     iter.gotoNext(); <br>
          * } <br>
          * </code>
          */
    class ConstBasicIterator {
    private:
        const Node* iter; //< current node on the list

    public:
        /**
              * Constructor needs the target list
              * @param the list to iterate on
              */
        explicit ConstBasicIterator(const FList& list) : iter(list.root){
        }

        /** to gotoNext on the list */
        void gotoNext(){
            if(this->iter){
                this->iter = this->iter->next;
                if(this->iter) Prefetch_Read0(this->iter->next);
            }
        }

        /**
            * Current pointed value
            * current iterator must be valide (hasNotFinished()) to use this function
            */
        Object data(){
            return this->iter->target;
        }

        /**
            * Current pointed value
            * current iterator must be valide (hasNotFinished()) to use this function
            */
        const Object& data() const{
            return this->iter->target;
        }

        /**
            * To know if an iterator is at the end of the list
            * @return true if the current iterator can gotoNext and access to value, else false
            */
        bool hasNotFinished() const{
            return iter;
        }
    };

};

#endif //FLIST_HPP

