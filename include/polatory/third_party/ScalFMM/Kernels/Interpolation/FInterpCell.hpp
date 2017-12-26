// See LICENCE file at project root
#ifndef FINTERPCELL_HPP
#define FINTERPCELL_HPP


#include "FInterpTensor.hpp"
#include "Components/FBasicCell.hpp"
#include "Extensions/FExtendCellType.hpp"

#include "Utils/FComplex.hpp"

/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FInterpCell
 * Please read the license
 *
 * This class defines a cell that can be used in both Chebyshev and
 * Lagrange based FMM.
 *
 * PB: This class provides the storage and accessors for the transformed
 * expansion used in the Uniform FMM (in Fourier space, i.e. complex valued).
 *
 * PB: Beware! This class does not support yet the storage of the compressed
 * expansions used in the low rank version of the Chebyshev FMM algorithm.
 *
 * @param NVALS is the number of right hand side.
 */
template <class FReal, int ORDER, int NRHS = 1, int NLHS = 1, int NVALS = 1>
class FInterpCell : public FBasicCell, public FAbstractSendable
{
    static const int VectorSize = TensorTraits<ORDER>::nnodes;
    static const int TransformedVectorSize = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1);

    FReal multipole_exp[NRHS * NVALS * VectorSize]; //< Multipole expansion
    FReal     local_exp[NLHS * NVALS * VectorSize]; //< Local expansion
    // PB: Store multipole and local expansion in Fourier space
    FComplex<FReal> transformed_multipole_exp[NRHS * NVALS * TransformedVectorSize];
    FComplex<FReal>     transformed_local_exp[NLHS * NVALS * TransformedVectorSize];

public:
    FInterpCell(){
        memset(multipole_exp, 0, sizeof(FReal) * NRHS * NVALS * VectorSize);
        memset(local_exp, 0, sizeof(FReal) * NLHS * NVALS * VectorSize);
        memset(transformed_multipole_exp, 0,
               sizeof(FComplex<FReal>) * NRHS * NVALS * TransformedVectorSize);
        memset(transformed_local_exp, 0,
               sizeof(FComplex<FReal>) * NLHS * NVALS * TransformedVectorSize);
    }

    ~FInterpCell() {}

    /** Get Multipole */
    const FReal* getMultipole(const int inRhs) const
    {	return this->multipole_exp + inRhs*VectorSize;
    }
    /** Get Local */
    const FReal* getLocal(const int inRhs) const{
        return this->local_exp + inRhs*VectorSize;
    }

    /** Get Multipole */
    FReal* getMultipole(const int inRhs){
        return this->multipole_exp + inRhs*VectorSize;
    }
    /** Get Local */
    FReal* getLocal(const int inRhs){
        return this->local_exp + inRhs*VectorSize;
    }

    /** To get the leading dim of a vec */
    int getVectorSize() const{
        return VectorSize;
    }

    /** Get Transformed Multipole */
    const FComplex<FReal>* getTransformedMultipole(const int inRhs) const{
        return this->transformed_multipole_exp + inRhs*TransformedVectorSize;
    }
    /** Get Transformed Local */
    const FComplex<FReal>* getTransformedLocal(const int inRhs) const{
        return this->transformed_local_exp + inRhs*TransformedVectorSize;
    }

    /** Get Transformed Multipole */
    FComplex<FReal>* getTransformedMultipole(const int inRhs){
        return this->transformed_multipole_exp + inRhs*TransformedVectorSize;
    }
    /** Get Transformed Local */
    FComplex<FReal>* getTransformedLocal(const int inRhs){
        return this->transformed_local_exp + inRhs*TransformedVectorSize;
    }

    /** To get the leading dim of a vec */
    int getTransformedVectorSize() const{
        return TransformedVectorSize;
    }

    /** Make it like the begining */
    void resetToInitialState(){
        memset(multipole_exp, 0, sizeof(FReal) * NRHS * NVALS * VectorSize);
        memset(local_exp, 0, sizeof(FReal) * NLHS * NVALS * VectorSize);
        memset(transformed_multipole_exp, 0,
               sizeof(FComplex<FReal>) * NRHS * NVALS * TransformedVectorSize);
        memset(transformed_local_exp, 0,
               sizeof(FComplex<FReal>) * NLHS * NVALS * TransformedVectorSize);
    }

    ///////////////////////////////////////////////////////
    // to extend FAbstractSendable
    ///////////////////////////////////////////////////////
    template <class BufferWriterClass>
    void serializeUp(BufferWriterClass& buffer) const{
        buffer.write(multipole_exp, VectorSize*NVALS*NRHS);
        buffer.write(transformed_multipole_exp, TransformedVectorSize*NVALS*NRHS);
    }

    template <class BufferReaderClass>
    void deserializeUp(BufferReaderClass& buffer){
        buffer.fillArray(multipole_exp, VectorSize*NVALS*NRHS);
        buffer.fillArray(transformed_multipole_exp, TransformedVectorSize*NVALS*NRHS);
    }

    template <class BufferWriterClass>
    void serializeDown(BufferWriterClass& buffer) const{
        buffer.write(local_exp, VectorSize*NVALS*NLHS);
        buffer.write(transformed_local_exp, TransformedVectorSize*NVALS*NLHS);
    }

    template <class BufferReaderClass>
    void deserializeDown(BufferReaderClass& buffer){
        buffer.fillArray(local_exp, VectorSize*NVALS*NLHS);
        buffer.fillArray(transformed_local_exp, TransformedVectorSize*NVALS*NLHS);
    }

    ///////////////////////////////////////////////////////
    // to extend Serializable
    ///////////////////////////////////////////////////////
    template <class BufferWriterClass>
    void save(BufferWriterClass& buffer) const{
        FBasicCell::save(buffer);
        buffer.write(multipole_exp, VectorSize*NVALS*NRHS);
        buffer.write(transformed_multipole_exp, TransformedVectorSize*NVALS*NRHS);
        buffer.write(local_exp, VectorSize*NVALS*NLHS);
        buffer.write(transformed_local_exp, TransformedVectorSize*NVALS*NLHS);
    }

    template <class BufferReaderClass>
    void restore(BufferReaderClass& buffer){
        FBasicCell::restore(buffer);
        buffer.fillArray(multipole_exp, VectorSize*NVALS*NRHS);
        buffer.fillArray(transformed_multipole_exp, TransformedVectorSize*NVALS*NRHS);
        buffer.fillArray(local_exp, VectorSize*NVALS*NLHS);
        buffer.fillArray(transformed_local_exp, TransformedVectorSize*NVALS*NLHS);
    }

    static constexpr int GetSize(){
        return (NRHS+NLHS)*NVALS*VectorSize * (int) sizeof(FReal) + (NRHS+NLHS)*NVALS*TransformedVectorSize * (int) sizeof(FComplex<FReal>);
    }

    FSize getSavedSize() const {
        return FSize((NRHS+NLHS)*NVALS*VectorSize * (int) sizeof(FReal)
                     + (NRHS+NLHS)*NVALS*TransformedVectorSize * (int) sizeof(FComplex<FReal>)
                     + FBasicCell::getSavedSize());
    }

    FSize getSavedSizeUp() const {
        return FSize((NRHS)*NVALS*VectorSize * (int) sizeof(FReal)
                     + (NRHS)*NVALS*TransformedVectorSize * (int) sizeof(FComplex<FReal>));
    }

    FSize getSavedSizeDown() const {
        return FSize((NLHS)*NVALS*VectorSize * (int) sizeof(FReal)
                     + (NLHS)*NVALS*TransformedVectorSize * (int) sizeof(FComplex<FReal>));
    }

};

template <class FReal,int ORDER, int NRHS = 1, int NLHS = 1, int NVALS = 1>
class FTypedInterpCell : public FInterpCell<FReal,ORDER,NRHS,NLHS,NVALS>, public FExtendCellType {
public:
    template <class BufferWriterClass>
    void save(BufferWriterClass& buffer) const{
        FInterpCell<FReal,ORDER,NRHS,NLHS,NVALS>::save(buffer);
        FExtendCellType::save(buffer);
    }
    template <class BufferReaderClass>
    void restore(BufferReaderClass& buffer){
        FInterpCell<FReal,ORDER,NRHS,NLHS,NVALS>::restore(buffer);
        FExtendCellType::restore(buffer);
    }
    void resetToInitialState(){
        FInterpCell<FReal,ORDER,NRHS,NLHS,NVALS>::resetToInitialState();
        FExtendCellType::resetToInitialState();
    }

    FSize getSavedSize() const {
        return FExtendCellType::getSavedSize() + FInterpCell<FReal, ORDER,NRHS,NLHS,NVALS>::getSavedSize();
    }
};

#endif //FINTERPCELL_HPP
