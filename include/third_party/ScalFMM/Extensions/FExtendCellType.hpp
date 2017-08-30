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
#ifndef FEXTENDCELLTYPE_HPP
#define FEXTENDCELLTYPE_HPP

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FExtendCellType
* This class is an extenssion.
* It proposes a target/source extenssion for cell.
* Because cells may have child that contains only
* sources or targets (in Tsm system) then it is important
* to not compute for nothing.
*/
class FExtendCellType {
protected:
    /** Current type */
    bool containsTargets;
    bool containsSources;
    bool m2lDone;
    bool l2lDone;

public:
    /** Default constructor */
    FExtendCellType() : containsTargets(false), containsSources(false), m2lDone(false), l2lDone(false) {
    }

    /** Copy constructor */
    FExtendCellType(const FExtendCellType& other)
        : containsTargets(other.containsTargets)
        ,containsSources(other.containsSources)
        ,m2lDone(other.m2lDone)
        ,l2lDone(other.l2lDone){
    }

    /** Copy operator */
    FExtendCellType& operator=(const FExtendCellType& other) {
        this->containsTargets = other.containsTargets;
        this->containsSources = other.containsSources;
        this->m2lDone = other.m2lDone;
        this->l2lDone = other.l2lDone;
        return *this;
    }

    /** To know if a cell has sources */
    bool hasSrcChild() const {
        return containsSources;
    }

    /** To know if a cell has targets */
    bool hasTargetsChild() const {
        return containsTargets;
    }

    /** To set cell as sources container */
    void setSrcChildTrue() {
        containsSources = true;
    }

    /** To set cell as targets container */
    void setTargetsChildTrue() {
        containsTargets = true;
    }

    void setTargetsChildFalse()
    {
       containsTargets = false;
    }

    bool isM2LDone() const {
       return m2lDone;
    }

    bool isL2LDone() const {
       return l2lDone;
    }

    void setM2LDoneTrue() {
       m2lDone = true;
    }

    void setL2LDoneTrue() {
       l2lDone = true;
    }

public:
    /** Save current object */
    template <class BufferWriterClass>
    void save(BufferWriterClass& buffer) const {
        buffer << containsTargets;
        buffer << containsSources;
    }
    /** Retrieve current object */
    template <class BufferReaderClass>
    void restore(BufferReaderClass& buffer) {
        buffer >> containsTargets;
        buffer >> containsSources;
    }
    /** reset to unknown type */
    void resetToInitialState(){
        containsTargets = false;
        containsSources = false;
        m2lDone = false;
        l2lDone = false;
    }

    FSize getSavedSize() const {
        return FSize(sizeof(containsTargets) + sizeof(containsSources));
    }
};


#endif //FEXTENDCELLTYPE_HPP


