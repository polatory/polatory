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
#ifndef FBASICCELL_HPP
#define FBASICCELL_HPP

#include "../Utils/FGlobal.hpp"
#include "../Containers/FTreeCoordinate.hpp"
#include "FAbstractSerializable.hpp"
#include "FAbstractSendable.hpp"


/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FBasicCell
*
* This class defines a basic cell used for examples. It extends
* the mininum, only what is needed by FOctree and FFmmAlgorithm
* to make the things working.
* By using this extension it will implement the FAbstractCell without
* inheriting from it.
*
*
*/
class FBasicCell : public FAbstractSerializable {
    MortonIndex mortonIndex;    ///< Morton index (need by most elements)
    FTreeCoordinate coordinate; ///< The position
    std::size_t level;          ///< Level in tree

public:
    /** Default constructor */
    FBasicCell() : mortonIndex(0) {
    }

    /** Default destructor */
    virtual ~FBasicCell(){
    }

    std::size_t getLevel() const {
        return this->level;
    }

    void setLevel(std::size_t inLevel) {
        this->level = inLevel;
    }

    /** To get the morton index */
    MortonIndex getMortonIndex() const {
        return this->mortonIndex;
    }

    /** To set the morton index */
    void setMortonIndex(const MortonIndex inMortonIndex) {
        this->mortonIndex = inMortonIndex;
    }

    /** To get the position */
    const FTreeCoordinate& getCoordinate() const {
        return this->coordinate;
    }

    /** To set the position */
    void setCoordinate(const FTreeCoordinate& inCoordinate) {
        this->coordinate = inCoordinate;
    }

    /** To set the position from 3 FReals */
    void setCoordinate(const int inX, const int inY, const int inZ) {
        this->coordinate.setX(inX);
        this->coordinate.setY(inY);
        this->coordinate.setZ(inZ);
    }

    /** Save the current cell in a buffer */
    template <class BufferWriterClass>
    void save(BufferWriterClass& buffer) const{
        buffer << mortonIndex;
        coordinate.save(buffer);
    }
    /** Restore the current cell from a buffer */
    template <class BufferReaderClass>
    void restore(BufferReaderClass& buffer){
        buffer >> mortonIndex;
        coordinate.restore(buffer);
    }

    FSize getSavedSize() const {
        return FSize(sizeof(mortonIndex)) +  coordinate.getSavedSize();
    }

    /** Do nothing */
    void resetToInitialState(){
    }
};


#endif //FBASICCELL_HPP
