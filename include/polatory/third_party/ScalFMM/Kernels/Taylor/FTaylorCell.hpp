// See LICENCE file at project root
#ifndef FTAYLORCELL_HPP
#define FTAYLORCELL_HPP

#include "../../Components/FBasicCell.hpp"
#include "../../Containers/FVector.hpp"
#include "../../Utils/FMemUtils.hpp"
#include "../../Extensions/FExtendCellType.hpp"

/**
 *@author Cyrille Piacibello
 *@class FTaylorCell
 *
 *This class is a cell used for the Taylor Expansion Kernel.
 *
 *
 */
template < class FReal, int P, int order>
class FTaylorCell : public FBasicCell, public FAbstractSendable {
protected:
    //Size of Multipole Vector
    static const int MultipoleSize = ((P+1)*(P+2)*(P+3))*order/6;
    //Size of Local Vector
    static const int LocalSize = ((P+1)*(P+2)*(P+3))*order/6;

    //Multipole vector
    FReal multipole_exp[MultipoleSize];
    //Local vector
    FReal local_exp[LocalSize];

public:
    /**
   *Default Constructor
   */
    FTaylorCell(){
        FMemUtils::memset(multipole_exp,0,MultipoleSize*sizeof(FReal(0)));
        FMemUtils::memset(local_exp,0,LocalSize*sizeof(FReal(0)));
    }

    //Get multipole Vector for setting
    FReal * getMultipole(void)
    {
        return multipole_exp;
    }

    //Get multipole Vector for reading
    const FReal * getMultipole(void) const
    {
        return multipole_exp;
    }

    //Get local Vector
    FReal * getLocal(void)
    {
        return local_exp;
    }

    //Get local Vector for reading
    const FReal * getLocal(void) const
    {
        return local_exp;
    }

    /** Make it like the begining */
    void resetToInitialState(){
        FMemUtils::memset(multipole_exp,0,MultipoleSize*sizeof(FReal(0)));
        FMemUtils::memset(local_exp,0,LocalSize*sizeof(FReal(0)));
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
        return ((FSize) sizeof(FReal) * (MultipoleSize));
    }

    FSize getSavedSizeDown() const {
        return ((FSize) sizeof(FReal) * (LocalSize));
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
        return FSize((sizeof(FReal) * (MultipoleSize + LocalSize)
                     + FBasicCell::getSavedSize()));
    }
};

template <class FReal, int P, int order>
class FTypedTaylorCell : public FTaylorCell<FReal, P,order>, public FExtendCellType {
public:
    template <class BufferWriterClass>
    void save(BufferWriterClass& buffer) const{
        FTaylorCell<FReal,P,order>::save(buffer);
        FExtendCellType::save(buffer);
    }
    template <class BufferReaderClass>
    void restore(BufferReaderClass& buffer){
        FTaylorCell<FReal,P,order>::restore(buffer);
        FExtendCellType::restore(buffer);
    }
    void resetToInitialState(){
        FTaylorCell<FReal,P,order>::resetToInitialState();
        FExtendCellType::resetToInitialState();
    }

    FSize getSavedSize() const {
        return FExtendCellType::getSavedSize() + FTaylorCell<FReal, P,order>::getSavedSize();
    }
};

#endif
