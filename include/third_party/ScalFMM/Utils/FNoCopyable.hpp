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
#ifndef FNOCOPYABLE_HPP
#define FNOCOPYABLE_HPP

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* This class has to be inherited to forbid copy
* @todo use C++0x ?
*/
class FNoCopyable {
private:
        /** Forbiden copy constructor */
        FNoCopyable(const FNoCopyable&) = delete;
        /** Forbiden copy operator */
        FNoCopyable& operator=(const FNoCopyable&) = delete;
protected:
        /** Empty constructor */
        FNoCopyable(){}
};

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* This class has to be inherited to forbid assignement
*/
class FNoAssignement {
private:
        /** Forbiden copy operator */
        FNoAssignement& operator=(const FNoAssignement&) = delete;
protected:
        /** Empty constructor */
        FNoAssignement(){}
};

#endif // FNOCOPYABLE_HPP
