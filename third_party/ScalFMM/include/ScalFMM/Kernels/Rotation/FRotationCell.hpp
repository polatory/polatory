// See LICENCE file at project root
#ifndef FROTATIONCELL_HPP
#define FROTATIONCELL_HPP

#include "../../Utils/FComplex.hpp"
#include "../../Utils/FMemUtils.hpp"

#include "../../Extensions/FExtendCellType.hpp"

#include "../../Components/FBasicCell.hpp"


/** This class is a cell used for the rotation based kernel
  * The size of the multipole and local vector are based on a template
  * User should choose this parameter P carrefuly to match with the
  * P of the kernel.
  *
  * Multipole/Local vectors contain value as:
  * {0,0}{1,0}{1,1}...{P,P-1}{P,P}
  * So the size of such vector can be obtained by a suite:
  * (n+1)*n/2 => (P+2)*(P+1)/2
  */
template <class FReal, int P>
class FRotationCell : public FBasicCell, public FAbstractSendable {
protected:
    //< Size of multipole vector
    static const int MultipoleSize = ((P+2)*(P+1))/2; // Artimethique suite (n+1)*n/2
    //< Size of local vector
    static const int LocalSize = ((P+2)*(P+1))/2;     // Artimethique suite (n+1)*n/2

    //< Multipole vector (static memory)
    FComplex<FReal> multipole_exp[MultipoleSize]; //< For multipole extenssion
    //< Local vector (static memory)
    FComplex<FReal> local_exp[LocalSize];         //< For local extenssion

public:
    /** Default constructor
      * Put 0 in vectors
      */
    FRotationCell(){
    }

    /** Copy constructor
      * Copy the value in the vectors
      */
    FRotationCell(const FRotationCell& other){
        (*this) = other;
    }

    /** Default destructor */
    virtual ~FRotationCell(){
    }

    /** Copy operator
      * copies only the value in the vectors
      */
    FRotationCell& operator=(const FRotationCell& other) {
        FMemUtils::copyall(multipole_exp, other.multipole_exp, MultipoleSize);
        FMemUtils::copyall(local_exp, other.local_exp, LocalSize);
        return *this;
    }

    /** Get Multipole array */
    const FComplex<FReal>* getMultipole() const {
        return multipole_exp;
    }
    /** Get Local array */
    const FComplex<FReal>* getLocal() const {
        return local_exp;
    }

    /** Get Multipole array */
    FComplex<FReal>* getMultipole() {
        return multipole_exp;
    }
    /** Get Local array */
    FComplex<FReal>* getLocal() {
        return local_exp;
    }

    int getArraySize() const
    {
        return MultipoleSize;
    }


    /** Make it like the begining */
    void resetToInitialState(){
        for(int idx = 0 ; idx < MultipoleSize ; ++idx){
            multipole_exp[idx].setRealImag(FReal(0.0), FReal(0.0));
        }
        for(int idx = 0 ; idx < LocalSize ; ++idx){
            local_exp[idx].setRealImag(FReal(0.0), FReal(0.0));
        }
    }

    ///////////////////////////////////////////////////////
    // to extend FAbstractSendable
    ///////////////////////////////////////////////////////
    template <class BufferWriterClass>
    void serializeUp(BufferWriterClass& buffer) const{
        buffer.write(multipole_exp, MultipoleSize);
    }
    template <class BufferReaderClass>
    void deserializeUp(BufferReaderClass& buffer){
        buffer.fillArray(multipole_exp, MultipoleSize);
    }

    template <class BufferWriterClass>
    void serializeDown(BufferWriterClass& buffer) const{
        buffer.write(local_exp, LocalSize);
    }
    template <class BufferReaderClass>
    void deserializeDown(BufferReaderClass& buffer){
        buffer.fillArray(local_exp, LocalSize);
    }

    FSize getSavedSizeUp() const {
        return ((FSize) sizeof(FComplex<FReal>)) * (MultipoleSize);
    }

    FSize getSavedSizeDown() const {
        return ((FSize) sizeof(FComplex<FReal>)) * (LocalSize);
    }

    ///////////////////////////////////////////////////////
    // to extend Serializable
    ///////////////////////////////////////////////////////
    template <class BufferWriterClass>
    void save(BufferWriterClass& buffer) const{
        FBasicCell::save(buffer);
        buffer.write(multipole_exp, MultipoleSize);
        buffer.write(local_exp, LocalSize);
    }
    template <class BufferReaderClass>
    void restore(BufferReaderClass& buffer){
        FBasicCell::restore(buffer);
        buffer.fillArray(multipole_exp, MultipoleSize);
        buffer.fillArray(local_exp, LocalSize);
    }

    FSize getSavedSize() const {
        return FSize(((int) sizeof(FComplex<FReal>)) * (MultipoleSize + LocalSize)
                + FBasicCell::getSavedSize());
    }
};

template <class FReal, int P>
class FTypedRotationCell : public FRotationCell<FReal, P>, public FExtendCellType {
public:
    template <class BufferWriterClass>
    void save(BufferWriterClass& buffer) const{
        FRotationCell<FReal, P>::save(buffer);
        FExtendCellType::save(buffer);
    }
    template <class BufferReaderClass>
    void restore(BufferReaderClass& buffer){
        FRotationCell<FReal, P>::restore(buffer);
        FExtendCellType::restore(buffer);
    }
    void resetToInitialState(){
        FRotationCell<FReal, P>::resetToInitialState();
        FExtendCellType::resetToInitialState();
    }

    FSize getSavedSize() const {
        return FExtendCellType::getSavedSize() + FRotationCell<FReal, P>::getSavedSize();
    }
};

#endif // FROTATIONCELL_HPP
