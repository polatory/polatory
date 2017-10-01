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


