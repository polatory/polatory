// See LICENCE file at project root
#ifndef FABSTRACTSERIALIZABLE_HPP
#define FABSTRACTSERIALIZABLE_HPP


/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FAbstractSerializable
*
* This proposes an interface to save and restore a class.
*
* To make your container are usable in the mpi fmm, they must provide this interface.
* This class is mainly called by the Sendable classes in order to store the attributes.
*/
class FAbstractSerializable {
protected:
    /**
     * Should save all the native data into the buffer
     */
    template <class BufferWriterClass>
    void save(BufferWriterClass&) const{
        static_assert(sizeof(BufferWriterClass) == 0 , "Your class should implement save");
    }

    /**
     * Should restore all the native data from the buffer
     * Warning! It should be done in the same order! and with the same size!
     */
    template <class BufferReaderClass>
    void restore(BufferReaderClass&){
        static_assert(sizeof(BufferReaderClass) == 0 , "Your class should implement restore");
    }

    virtual FSize getSavedSize() const = 0;
};

#endif // FABSTRACTSERIALIZABLE_HPP
