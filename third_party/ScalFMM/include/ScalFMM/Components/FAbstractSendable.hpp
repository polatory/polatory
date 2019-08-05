// See LICENCE file at project root
#ifndef FABSTRACTSENDABLE_HPP
#define FABSTRACTSENDABLE_HPP

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FAbstractSendable
* To make your cells are usable in the mpi fmm, they must provide this interface.
*
* If used during M2M or M2L they will be serialize up (multipole) if for the L2L serialize down is used.
*/
class FAbstractSendable {
protected:
    /** Empty Destructor */
    virtual ~FAbstractSendable(){}

    ///////////////////////////////////////////////
    // For Upward pass
    ///////////////////////////////////////////////

    /** Save your data */
    template <class BufferWriterClass>
    void serializeUp(BufferWriterClass&) const{
        static_assert(sizeof(BufferWriterClass) == 0 , "Your class should implement serializeUp");
    }
    /** Retrieve your data */
    template <class BufferReaderClass>
    void deserializeUp(BufferReaderClass&){
        static_assert(sizeof(BufferReaderClass) == 0 , "Your class should implement deserializeUp");
    }

    virtual FSize getSavedSizeUp() const = 0;

    ///////////////////////////////////////////////
    // For Downward pass
    ///////////////////////////////////////////////

    /** Save your data */
    template <class BufferWriterClass>
    void serializeDown(BufferWriterClass&) const{
        static_assert(sizeof(BufferWriterClass) == 0 , "Your class should implement serializeDown");
    }
    /** Retrieve your data */
    template <class BufferReaderClass>
    void deserializeDown(BufferReaderClass&){
        static_assert(sizeof(BufferReaderClass) == 0 , "Your class should implement deserializeDown");
    }

    virtual FSize getSavedSizeDown() const = 0;
};


#endif //FABSTRACTSENDABLE_HPP


