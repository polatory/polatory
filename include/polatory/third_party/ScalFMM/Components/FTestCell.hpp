// See LICENCE file at project root
#ifndef FTESTCELL_HPP
#define FTESTCELL_HPP

#include <cstddef>
#include "FBasicCell.hpp"

/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * @class FBasicCell*
 * @brief This class is used in the FTestKernels, please look at this class to know how to customize a cell.
 *
 * This cell simply store the data when up/down.
 * It also shows how to be restored and saved, etc.
 */
class FTestCell : public FBasicCell, public FAbstractSendable  {
protected:
	// To store data during upward and downward pass
	long long int dataUp, dataDown;
public:
	FTestCell(): dataUp(0) , dataDown(0){
	}
	/** Default destructor */
	virtual ~FTestCell(){
	}
	/** When doing the upward pass */
	long long int getDataUp() const {
		return this->dataUp;
	}
	/** When doing the upward pass */
	void setDataUp(const long long int inData){
		this->dataUp = inData;
	}
	/** When doing the downard pass */
	long long int getDataDown() const {
		return this->dataDown;
	}
	/** When doing the downard pass */
	void setDataDown(const long long int inData){
		this->dataDown = inData;
	}

	/** Make it like the begining */
	void resetToInitialState(){
		this->dataDown = 0;
		this->dataUp = 0;
	}

	/////////////////////////////////////////////////

	/** Save the current cell in a buffer */
	template <class BufferWriterClass>
	void save(BufferWriterClass& buffer) const{
		FBasicCell::save(buffer);
		buffer << dataDown << dataUp;
	}

	/** Restore the current cell from a buffer */
	template <class BufferReaderClass>
	void restore(BufferReaderClass& buffer){
		FBasicCell::restore(buffer);
		buffer >> dataDown >> dataUp;
	}

    FSize getSavedSize() const {
        return FSize(sizeof(long long int))*2 + FBasicCell::getSavedSize();
    }

	/////////////////////////////////////////////////

	/** Serialize only up data in a buffer */
	template <class BufferWriterClass>
	void serializeUp(BufferWriterClass& buffer) const {
		buffer << this->dataUp;
	}
	/** Deserialize only up data in a buffer */
	template <class BufferReaderClass>
	void deserializeUp(BufferReaderClass& buffer){
		buffer >> this->dataUp;
	}

	/** Serialize only down data in a buffer */
	template <class BufferWriterClass>
	void serializeDown(BufferWriterClass& buffer) const {
		buffer << this->dataDown;
	}
	/** Deserialize only up data in a buffer */
	template <class BufferReaderClass>
	void deserializeDown(BufferReaderClass& buffer){
		buffer >> this->dataDown;
	}

    FSize getSavedSizeDown() const {
        return FSize(sizeof(long long int));
    }

    FSize getSavedSizeUp() const {
        return FSize(sizeof(long long int));
    }
};

#endif //FTESTCELL_HPP


